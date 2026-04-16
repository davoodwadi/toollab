from __future__ import annotations

from argparse import ArgumentParser
import json

from tool_lab.config import MatrixMode, load_experiment_spec
from tool_lab.runner import ExperimentRunner
from tool_lab.storage import read_jsonl

 
def main() -> None:
    parser = ArgumentParser(prog="tool-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
 
    run_parser = subparsers.add_parser("run", help="Run a Tool-Lab experiment")
    run_parser.add_argument("--config", required=True, help="Path to a YAML or JSON experiment spec")
    run_parser.add_argument("--provider", help="Model provider override: openai, anthropic, google, or mock")
    run_parser.add_argument("--model", help="Model name override")
    run_parser.add_argument(
        "--matrix-mode",
        choices=["fixed", "scrolling"],
        help="Override the experiment interface mode",
    )
    run_parser.add_argument("--replications", type=int, help="Override replication count")
    run_parser.add_argument("--budget_tools", type=int, help="Override max tool calls")
    run_parser.add_argument("--api-key-env", help="Override the API key environment variable name")
    run_parser.add_argument("--output-root", default="results", help="Directory for result artifacts")
    run_parser.add_argument("--mock", action="store_true", help="Run a local mock trial to verify the design")
    run_parser.add_argument("--verbose", action="store_true", help="Show more logs")

    grid_parser = subparsers.add_parser("grid", help="Run a grid of Tool-Lab experiments")
    grid_parser.add_argument("--config", required=True, help="Path to a YAML or JSON experiment spec")
    grid_parser.add_argument("--models", nargs="+", required=True, help="List of provider:model combinations (e.g. openai:gpt-5.4-mini llamacpp:default)")
    grid_parser.add_argument("--budget_tools", nargs="+", type=int, help="List of max tool calls to iterate over (e.g. 10 20 30)")
    grid_parser.add_argument(
        "--matrix-mode",
        choices=["fixed", "scrolling"],
        help="Override the experiment interface mode",
    )
    grid_parser.add_argument("--replications", type=int, help="Override replication count")
    grid_parser.add_argument("--api-key-env", help="Override the API key environment variable name")
    grid_parser.add_argument("--output-root", default="results", help="Directory for result artifacts")
    grid_parser.add_argument("--mock", action="store_true", help="Run a local mock trial to verify the design")
    grid_parser.add_argument("--verbose", action="store_true", help="Show more logs")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Recompute summary statistics from an existing runs.jsonl file"
    )
    summarize_parser.add_argument("--runs", required=True, help="Path to runs.jsonl")

    args = parser.parse_args()

    if args.command == "run":
        mock_overrides = {}
        if args.mock:
            mock_overrides = {
                "provider": "mock",
                "model_name": "mock-v1",
                "replications": 1
            }
        # print()
        spec = load_experiment_spec(args.config).with_runtime_overrides(
            matrix_mode=args.matrix_mode,
            provider=args.provider or mock_overrides.get("provider"),
            model_name=args.model or mock_overrides.get("model_name"),
            replications=args.replications or mock_overrides.get("replications"), 
            budget_tools=args.budget_tools,
            api_key_env=args.api_key_env,
        )
        runner = ExperimentRunner(
            spec, 
            output_root=args.output_root, 
            verbose=args.verbose or args.mock)
        result = runner.run()
        # print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "grid":
        mock_overrides = {}
        if args.mock:
            mock_overrides = {
                "replications": 1
            }
        print(f'Running a grid over: \n{args.models}\ntool calls: \n{args.budget_tools}\n')
        for model_str in args.models:
            if ":" not in model_str:
                print(f"Error: Invalid model format '{model_str}'. Must be provider:model_name")
                continue
            provider, model_name = model_str.split(":", 1)
            if args.mock:
                provider = "mock"
                model_name = "mock-v1"
            if args.budget_tools:
                for budget in args.budget_tools:
                    print(f"\n{'='*50}")
                    print(f"Running grid cell: provider={provider}, model={model_name}, budget_tools={budget}")
                    print(f"{'='*50}\n")
                    # continue
                    try:
                        spec = load_experiment_spec(args.config).with_runtime_overrides(
                            matrix_mode=args.matrix_mode,
                            provider=provider,
                            model_name=model_name,
                            budget_tools=budget,
                            replications=args.replications or mock_overrides.get("replications"),
                            api_key_env=args.api_key_env,
                        )
                        runner = ExperimentRunner(
                            spec, 
                            output_root=args.output_root, 
                            verbose=args.verbose or args.mock
                        )
                        runner.run()
                    except Exception as e:
                        import traceback
                        print(f"Error running grid cell ({provider}:{model_name}, budget={budget}): {e}")
                        traceback.print_exc()
            else:
                print(f"\n{'='*50}")
                print(f"Running grid cell: provider={provider}, model={model_name}")
                print(f"{'='*50}\n")
                try:
                    spec = load_experiment_spec(args.config).with_runtime_overrides(
                        matrix_mode=args.matrix_mode,
                        provider=provider,
                        model_name=model_name,
                        replications=args.replications or mock_overrides.get("replications"),
                        api_key_env=args.api_key_env,
                    )
                    runner = ExperimentRunner(
                        spec, 
                        output_root=args.output_root, 
                        verbose=args.verbose or args.mock
                    )
                    runner.run()
                except Exception as e:
                    import traceback
                    print(f"Error running grid cell ({provider}:{model_name}, budget={budget}): {e}")
                    traceback.print_exc()


        return

    if args.command == "summarize":
        records = read_jsonl(args.runs)


if __name__ == "__main__":
    main()
