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
    run_parser.add_argument("--api-key-env", help="Override the API key environment variable name")
    run_parser.add_argument("--output-root", default="results", help="Directory for result artifacts")
    run_parser.add_argument("--mock", action="store_true", help="Run a local mock trial to verify the design")
    run_parser.add_argument("--verbose", action="store_true", help="Show more logs")

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
        
        spec = load_experiment_spec(args.config).with_runtime_overrides(
            matrix_mode=args.matrix_mode,
            provider=args.provider or mock_overrides.get("provider"),
            model_name=args.model or mock_overrides.get("model_name"),
            replications=args.replications or mock_overrides.get("replications"), 
            api_key_env=args.api_key_env,
        )
        runner = ExperimentRunner(
            spec, 
            output_root=args.output_root, 
            verbose=args.verbose or args.mock)
        result = runner.run()
        # print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "summarize":
        records = read_jsonl(args.runs)


if __name__ == "__main__":
    main()
