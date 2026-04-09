from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from tool_lab.config import ExperimentSpec
from tool_lab.experiment.environment import build_environment
from tool_lab.models import create_model_session
from tool_lab.storage import ResultWriter
from tool_lab.models.base import _to_serializable

class ExperimentRunner:
    def __init__(self, spec: ExperimentSpec, output_root: str = "results", verbose: bool = False) -> None:
        self.spec = spec
        self.output_root = output_root
        self.verbose = verbose 

    def run(self) -> dict[str, Any]:
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
        seed = (self.spec.seed or 0) + replicate_index

        # build_environment: fixed or scrolling -> ToolLabEnvironment: with methods for _inspect_cell
        environment = build_environment(self.spec, seed)

        # Based on the config, get the correct provider `session` with system_prompt, initial_user_message, and tools
        model_session = create_model_session(
            self.spec.model,
            environment.build_system_prompt(), 
            environment.build_user_prompt(),
        )
        # print('model_session.system_prompt', model_session.system_prompt)
        # print('model_session.initial_user_message', model_session.initial_user_message)
        # exit()
        started_at = datetime.now(timezone.utc).isoformat()
        forced_choice = False
        
        if self.verbose:
            num_options = len(environment.options)
            num_attributes = len(environment.attributes)
            total_cells = num_options * num_attributes
            print("\n" + "="*50)
            print("INITIALIZING EXPERIMENT (MOCK MODE)")
            print("="*50)
            print(f"System Prompt:\n{model_session.system_prompt}\n")
            print(f"User Prompt:\n{model_session.initial_user_message}\n")
            print(f"Num Options: {num_options}")
            print(f"Num Attributes: {num_attributes}")
            print(f"Total cells: {total_cells}")
            if environment.spec.budget_type=='usd':
                print(f"Budget: ${environment.spec.budget_usd}")
            else:
                print(f"Budget: {environment.spec.budget_tools} Tools")
            print("="*50 + "\n")
        # print('model_session.contents', model_session.contents)
        # exit()

        for iteration in range(self.spec.max_turns):
            # calls the LLM -> gets response (tool_call, content, reasoning) -> adds it to session.transcript
            # gets: LLM's response             
            assistant_response = model_session._call_model()
            environment.charge_model_turn(assistant_response)
            # print('budget_remaining_tools', environment.budget_remaining_tools, 'budget_remaining_usd', environment.budget_remaining_usd)
            # START: record assistant_response
            assistant_data = _to_serializable(assistant_response)
            if assistant_response.tool_calls:
                tc = assistant_response.tool_calls[0]
                assistant_data["tool_name"] = tc.name
                assistant_data["tool_arguments"] = tc.arguments
            environment._record_event(
                kind='assistant',
                data=assistant_data,
            )
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
                model_session.messages.append({'role': 'user', 'content': force_message})
                forced_choice = True
                # print('setting forced choice')
                print('FORCING CHOICE', environment.budget_remaining_usd if environment.spec.budget_type=='usd' else environment.budget_remaining_tools)
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
                model_session.messages.append({'role':'user','content':environment.reminder_message()})
                print([m['role'] for m in model_session.messages])
                continue
            # END: check if model did not call any tools in this turn

            # print([m for m in model_session.messages if m['role']=='assistant'])

            # execute first tool
            tool_call = assistant_response.tool_calls[0]
            tool_response = environment.execute_tool(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                tool_call_id=tool_call.tool_call_id,
            )
            
            model_session.add_tool_response(tool_response, tool_name=tool_call.name)

            
            print(tool_call.name)
            print(tool_response['content'])
            # print('opened_cues', environment.opened_cues)
            if tool_call.name == "submit_choice":
                break
            # error the remaining tools if any
            if len(assistant_response.tool_calls)>1:
                for tool_call in assistant_response.tool_calls[1:]:
                    # return error
                    tool_error = model_session._get_tool_error_one_tool_only(tool_call)
                    model_session.add_tool_response(tool_error, tool_name=tool_call.name)
            print('*'*50)

        # session ended -> record the response
        run_record = {
            "session_name": session_name,
            "experiment_name": self.spec.name,
            "provider": self.spec.model.provider,
            "model_name": self.spec.model.model_name,
            'forced_choice': forced_choice,
            "choice": environment.choice,
            "cumulative_cost_usd": environment.cumulative_cost_usd,
            "budget_remaining_usd": environment.budget_remaining_usd,
            "cumulative_cost_tools": environment.cumulative_cost_tools,
            "budget_remaining_tools": environment.budget_remaining_tools,
            **self.spec.metadata,
            'trace': environment.trace,
            "seed": seed,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        return run_record


def is_budget_exhausted(environment):
    if environment.spec.budget_type=='usd':
        return environment.budget_remaining_usd<=0
    elif environment.spec.budget_type=='tools':
        return environment.budget_remaining_tools<0
    else: 
        return
