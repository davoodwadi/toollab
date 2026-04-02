from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from tool_lab.config import ExperimentSpec
from tool_lab.experiment.environment import build_environment
from tool_lab.models import create_model_session
from tool_lab.models.base import ToolResultMessage
from tool_lab.storage import ResultWriter
from tool_lab.experiment.tools import BUILTIN_TOOL_DEFINITIONS
from tool_lab.models.base import _to_serializable

class ExperimentRunner:
    def __init__(self, spec: ExperimentSpec, output_root: str = "results") -> None:
        self.spec = spec
        self.output_root = output_root 

    def run(self) -> dict[str, Any]:
        for index in range(self.spec.replications):
            writer = ResultWriter(
                output_root=self.output_root,
                experiment_name=self.spec.name,
                provider=self.spec.model.provider, 
                model_name=self.spec.model.model_name,
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
        # print(environment.spec.keys())
        # print(environment.spec.to_dict().keys())
        # print('*'*50)
        # print(environment.spec.to_dict().get('options'))
        # print('*'*50)
        # check config requested tools against available tool (BUILTIN_TOOL_DEFINITIONS) and return `tools` list
        # print('BUILTIN_TOOL_DEFINITIONS', BUILTIN_TOOL_DEFINITIONS)
        tools = [value for name, value in BUILTIN_TOOL_DEFINITIONS.items() if name in environment.spec.tools]
        # print('tools', tools[0])

        # print('*'*50)
        # Based on the config, get the correct provider `session` with system_prompt, initial_user_message, and tools
        model_session = create_model_session(
            self.spec.model,
            environment.build_system_prompt(),
            environment.build_user_prompt(),
            tools,
        )
        # print('build_system_prompt',environment.build_system_prompt())
        # print('build_user_prompt',environment.build_user_prompt())
        started_at = datetime.now(timezone.utc).isoformat()

        forced_choice = False
        for _ in range(self.spec.max_turns):
            # calls the LLM -> gets response (tool_call, content, reasoning) -> adds it to session.transcript
            # gets: LLM's response 
            
            assistant_response = model_session._call_model()
            model_session.messages.append(assistant_response.raw)
            # add to environment.cumulative_cost_usd and .model_turns
            cost = environment.get_model_cost(assistant_response)
            environment.apply_model_cost(cost)

            environment._record_event(
                kind='assistant', 
                tool_name=None,
                cost=cost,
                data=_to_serializable(assistant_response)
            )
            
            print(environment._step_index)
            print('cumulative_cost_usd:', round(environment.cumulative_cost_usd, 3))
            print('budget_remaining_usd:', round(environment.budget_remaining_usd, 3))
            print('*'*50)

            # START: force choice
            if (not assistant_response.tool_calls):
                is_not_choice = True
            elif assistant_response.tool_calls[0].name != 'submit_choice':
                is_not_choice = True
            else:
                is_not_choice = False
            
            if (environment.budget_remaining_usd<=0) and (is_not_choice) and (not forced_choice):
                force_message = environment.forced_vote_message()
                model_session.messages.append({'role': 'user', 'content': force_message})
                forced_choice = True
                # print('setting forced choice')
                environment._record_event(
                    kind='setting_forced_choice', 
                    tool_name=None,
                    cost=None,
                    data={"force_message":force_message}
                )
                continue

            if (environment.budget_remaining_usd<=0) and (is_not_choice) and (forced_choice):
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
            model_session.messages.append(tool_response)
            
            # error the remaining tools if any
            if len(assistant_response.tool_calls)>1:
                for tool_call in assistant_response.tool_calls[1:]:
                    # return error
                    tool_error = model_session._get_tool_error(tool_call)
                    model_session.messages.append(tool_error)

        # session ended -> record the response
        run_record = {
            "session_name": session_name,
            "experiment_name": self.spec.name,
            "provider": self.spec.model.provider,
            "model_name": self.spec.model.model_name,
            "matrix_mode": self.spec.matrix_mode,
            "choice": environment.choice,
            "cumulative_cost_usd": environment.cumulative_cost_usd,
            "budget_remaining_usd": environment.budget_remaining_usd,
            'trace': environment.trace,
            "seed": seed,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        return run_record




