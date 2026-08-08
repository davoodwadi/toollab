from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from tool_lab.config import ExperimentSpec
from tool_lab.experiment.environment import build_environment
from tool_lab.models import create_model_session
from tool_lab.storage import ResultWriter
from tool_lab.models.base import _to_serializable, AssistantResponse, ToolInvocation

import time

import random
from copy import deepcopy

class ExperimentRunner:
    def __init__(self, spec: ExperimentSpec, output_root: str = "results", verbose: bool = False) -> None:
        self.spec = spec
        self.output_root = output_root
        self.verbose = verbose 

    def run(self) -> dict[str, Any]:
        # print('self.spec.budget_type', self.spec.budget_type)
        # print('self.spec.inspect_cell_tool_cost', self.spec.inspect_cell_tool_cost)
        for index in range(self.spec.replications):
            writer = ResultWriter(
                output_root=self.output_root,
                experiment_name=self.spec.name,
                provider=self.spec.model.provider, 
                model_name=self.spec.model.model_name,
                metadata=self.spec.metadata,
            )
            if index==0:
                writer.write_config(self.spec.to_dict())
            
            record = self._run_single(index + 1, session_name = writer.session_name)
            writer.write_events(record)
 
        return {
            "session_dir": str(writer.session_dir),
        }

    def _run_single(self, replicate_index: int, session_name: str) -> dict[str, Any]:
        run_spec = deepcopy(self.spec)

        # 2. Apply Randomization if configured
        if run_spec.randomization:
            base_seed = run_spec.randomization.get("seed")
            # print(base_seed)
            if base_seed is not None:
                rng = random.Random(base_seed + replicate_index)
                print(f'randomizing with seed: {base_seed + replicate_index}')
            else:
                rng = random.Random()
                print('randomizing without seed')
            
            if run_spec.randomization.get("randomize_attributes_options"):
                rng.shuffle(run_spec.options)
                if run_spec.attributes:
                    rng.shuffle(run_spec.attributes)

            # 2b. Sample from placeholder pools
            for placeholder, pool in run_spec.cue_value_pools.items():
                if not pool:
                    continue
                
                matching_cues = [cue for cue in run_spec.cues if cue.value == placeholder]
                if not matching_cues:
                    continue
                
                if len(pool) >= len(matching_cues):
                    values = rng.sample(pool, len(matching_cues))
                else:
                    values = rng.choices(pool, k=len(matching_cues))
                
                for cue, new_val in zip(matching_cues, values):
                    cue.value = new_val

        # print('run_spec.cues', [c for c in run_spec.cues if c.attribute_id=='brand'])
        # 3. Rewrite option IDs to positional names (e.g. option_1)
        for i, option in enumerate(run_spec.options):
            original_id = option.metadata.get('original_id', option.id)
            position = i + 1
            new_id = f"option_{position}"
            new_name = option.display_name
            
            # Ensure the original ID is saved so we can map it back later
            option.metadata['original_id'] = original_id
            
            # Rewrite for the LLM
            option.id = new_id
            option.display_name = new_name
            
            # Update all cues that belonged to this option
            for cue in run_spec.cues:
                if cue.option_id == original_id:
                    cue.option_id = new_id
        # print(run_spec.options[0])
        # print(run_spec.cues[0])
        # exit()
        # print(run_spec.inspect_mode)
        # exit()    
        environment = build_environment(run_spec) 
        # print(environment)
        # exit()
        # if environment.attributes:
        #     print('environment.attributes', list(environment.attributes.keys()))
        # Based on the config, get the correct provider `session` with system_prompt, initial_user_message, and tools
        model_session = create_model_session(
            self.spec.model, 
            environment.build_system_prompt(), 
            environment.build_user_prompt(),
            environment=environment
        )
        
        # print('model_session.system_prompt', model_session.system_prompt)
        # print('model_session.initial_user_message', model_session.initial_user_message)
        # exit()
        started_at = datetime.now(timezone.utc).isoformat()
        forced_choice = False
        
        print("\n" + "+"*50)
        print(f'replicate_index: {replicate_index}')


        if self.verbose and replicate_index==1:
            num_options = len(run_spec.options)
            num_attributes = len(run_spec.attributes) if run_spec.attributes else 0
            total_cells = num_options * num_attributes
            print("\n" + "="*50)
            print("INITIALIZING EXPERIMENT")
            print("="*50)
            print(f'Model: {environment.spec.model.model_name}\n')
            print(f'Replications: {environment.spec.replications}\n')
            if environment.spec.budget_type=='usd':
                print(f"Budget: ${environment.spec.budget_usd}")
            elif environment.spec.budget_type=='tools':
                print(f"Budget: {environment.spec.budget_tools} Tools")
            elif environment.spec.budget_type=='tool_usd':
                print(f"Tool cost: ${environment.spec.inspect_tool_cost}")
            elif environment.spec.budget_type=='tokens':
                print(f"Budget: {environment.spec.budget_tokens} Tokens")
            elif environment.spec.budget_type=='points':
                print(f"Budget: {environment.spec.budget_points} Points")
            print('*'*50)
            print(f"System Prompt:\n{model_session.system_prompt}\n")
            print('*'*50)
            print(f"User Prompt:\n{model_session.initial_user_message}\n")
            print('*'*50)
            print(f"Num Options: {num_options}")
            print(f"Num Attributes: {num_attributes}")
            print(f"Total cells: {total_cells}")

            # exit()
            # print("="*50)
            # print("EXPERIMENT MATRIX:")
            # opt_ids = [opt.id for opt in run_spec.options]
            # header = f"{'Attribute':<25} | " + " | ".join([f"{opt:<20}" for opt in opt_ids])
            # print(header)
            # print("-" * len(header))
            
            # attributes = set([c.attribute_id for c in run_spec.cues])

            # for attr in attributes:
            #     row_str = f"{attr:<25} | "
            #     vals = []
            #     for opt in opt_ids:
            #         val = "N/A"
            #         for c in run_spec.cues:
            #             if c.option_id == opt and c.attribute_id == attr:
            #                 val = str(c.value)
            #                 break
            #         vals.append(f"{val:<20}")
            #     row_str += " | ".join(vals)
            #     print(row_str)
            # print("="*50 + "\n")
        # exit()

        survey_data = {}
        for iteration in range(self.spec.max_turns):
            # calls the LLM -> gets response (tool_call, content, reasoning) -> adds it to session.transcript
            # gets: LLM's response              
            assistant_response = model_session._call_model()
            environment.charge_model_turn(assistant_response) 
            # START: record assistant_response
            assistant_data = _to_serializable(assistant_response)

            environment._record_event( 
                kind='assistant',
                data=assistant_data,
            )
            # print('assistant_data', assistant_data)

            if assistant_response.tool_calls:
                tc = assistant_response.tool_calls[0]
                assistant_data["tool_name"] = tc.name
                assistant_data["tool_arguments"] = tc.arguments

                if assistant_data["tool_name"]=='submit_choice' and self.verbose:
                    print('+'*50)
                    print(assistant_response.reasoning)
                    print('+'*50)
            # END: record assistant_response
            
            print(environment._step_index)

            # START: force choice
            if (not assistant_response.tool_calls): 
                is_not_choice = True
            elif assistant_response.tool_calls[0].name != 'submit_choice':
                is_not_choice = True
            else:
                is_not_choice = False
            
            if (is_budget_exhausted(environment)) and (is_not_choice) and (not forced_choice):
                force_message = environment.forced_vote_message()
                model_session.add_force_message(force_message)
                forced_choice = True

                # print('setting forced choice')
                print('FORCING CHOICE', environment.budget_remaining_usd if environment.spec.budget_type=='usd' else environment.budget_remaining_tools if environment.spec.budget_type=='tools' else environment.budget_remaining_tokens if environment.spec.budget_type=='tokens' else environment.budget_remaining_points)
                continue

            if (is_budget_exhausted(environment)) and (is_not_choice) and (forced_choice):
                print('forced choice ACTIVATED')
                # forced choice -> still no choice - Missed forced choice
                break
            # END: force choice


            # START: check if model did not call any tools in this turn
            if not assistant_response.tool_calls:
                print('model did not call any tools in this turn')
                # model did not call any tools -> ask it to try again
                model_session.add_reminder(environment.reminder_message())
                # print([m['role'] for m in model_session.messages])
                continue
            # END: check if model did not call any tools in this turn

            # print([m for m in model_session.messages if m['role']=='assistant'])

            # START: execute tools
            tool_results_and_name = []
            for index, tc in enumerate(assistant_response.tool_calls):
                if index==0:
                    # execute first tool
                    print(f"Tool Name: {tc.name}")
                    print(f"Tool Arguments: {tc.arguments}")
                    tool_response = environment.execute_tool(
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        tool_call_id=tc.tool_call_id,
                    )
                    tool_results_and_name.append((tool_response, tc.name))

                    print(f"Tool Response: {tool_response.get('content')}")
                else:
                    # error the remaining tools, if any
                    # return error
                    tool_error = model_session._get_tool_error_one_tool_only(tc)
                    tool_results_and_name.append((tool_error, tc.name))
            
            model_session.add_tool_results(tool_results_and_name)

            if assistant_response.tool_calls[0].name == "submit_choice":
                print(f"Choice: {environment.choice}")
                if hasattr(model_session, 'administer_survey'):
                    survey_data = model_session.administer_survey()
                    print(survey_data)
                break
            print('*'*50)
            # END: execute tools
        print('+'*50)
        print(f'Total cost: ${environment.cumulative_cost_usd}')
        print('+'*50)
        # session ended -> record the response

        run_record = {
            "session_name": session_name,
            "experiment_name": self.spec.name,
            "inspect_model":self.spec.inspect_mode,
            "provider": self.spec.model.provider,
            "model_name": self.spec.model.model_name,
            'forced_choice': forced_choice,
            "choice": environment.choice,
            'budget_type':environment.spec.budget_type,
            'budget_max': getattr(environment.spec, f"budget_{environment.spec.budget_type}", None),
            
            "cumulative_cost_usd": environment.cumulative_cost_usd,
            "budget_remaining_usd": environment.budget_remaining_usd,
            
            "cumulative_cost_tools": environment.cumulative_cost_tools,
            "budget_remaining_tools": environment.budget_remaining_tools,
            
            "cumulative_cost_tool_usd": environment.cumulative_cost_tool_usd,

            "cumulative_cost_tokens": environment.cumulative_cost_tokens,
            "budget_remaining_tokens": environment.budget_remaining_tokens,
            
            "cumulative_cost_points": environment.cumulative_cost_points,
            "budget_remaining_points": environment.budget_remaining_points,
            
            **self.spec.metadata,
            'trace': environment.trace,
            "option_mapping": {opt.id: opt.metadata.get('original_id', opt.id) for opt in run_spec.options},
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            'system_prompt' : model_session.system_prompt,
            'user_message' : model_session.initial_user_message,
        }

        if environment.spec.budget_type=='tool_usd':
            run_record['inspect_tool_cost'] = environment.spec.inspect_tool_cost

        if survey_data:
            for k,v in survey_data.items():
                run_record[k] = v


        return run_record


def is_budget_exhausted(environment):
    if environment.spec.budget_type=='usd':
        return False
        # return environment.budget_remaining_usd<=0
    elif environment.spec.budget_type=='tools' and environment.spec.budget_tools>=0:
        return environment.budget_remaining_tools<0
    elif environment.spec.budget_type=='tools' and environment.spec.budget_tools<0:
        return False
    elif environment.spec.budget_type=='tokens':
        return environment.budget_remaining_tokens<0
    elif environment.spec.budget_type=='tool_usd':
        return False
    elif environment.spec.budget_type=='points':
        # return environment.budget_remaining_points<0
        return False
    else: 
        return
