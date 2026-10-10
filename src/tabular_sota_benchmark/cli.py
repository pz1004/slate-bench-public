from __future__ import annotations

import argparse
from pathlib import Path

from tabular_sota_benchmark.config import load_config
from tabular_sota_benchmark.paper import write_results_section
from tabular_sota_benchmark.runner import run_experiment
from tabular_sota_benchmark.stats import summarize_results
from tabular_sota_benchmark.verify import verify_all


def reference_method_from_config(config: dict) -> str:
    selector_config = config.get("selector", {})
    return selector_config.get(
        "reference_method_name",
        selector_config.get(
            "ensemble_method_name",
            selector_config.get("selector_method_name", "meta_selector_ensemble"),
        ),
    )


def extra_reference_methods_from_config(config: dict) -> list[str]:
    selector_config = config.get("selector", {})
    value = selector_config.get("extra_reference_method_names", [])
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _print_verification(results: list[object]) -> bool:
    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.step}: {result.verification}")
        if not result.passed:
            print(f"  Reason: {result.reason}")
            all_passed = False
    return all_passed


def command_run_experiment(args: argparse.Namespace) -> int:
    path = run_experiment(args.config)
    print(f"Wrote results: {path}")
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result_path = Path(config["experiment"]["output_dir"]) / "results.csv"
    outputs = summarize_results(
        result_path,
        config["experiment"]["analysis_dir"],
        reference_method=reference_method_from_config(config),
        extra_reference_methods=extra_reference_methods_from_config(config),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


def command_write_paper(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    path = write_results_section(
        config["experiment"]["analysis_dir"],
        paper_dir="paper",
        reference_method=reference_method_from_config(config),
    )
    print(f"Wrote paper results: {path}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    passed = _print_verification(verify_all(args.config))
    return 0 if passed else 1


def command_run_loop(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    reference_method = reference_method_from_config(config)
    for attempt in range(int(args.max_restarts) + 1):
        if attempt:
            print(f"Restart attempt {attempt} of {args.max_restarts}")
        result_path = run_experiment(args.config)
        print(f"Wrote results: {result_path}")
        summarize_results(
            result_path,
            config["experiment"]["analysis_dir"],
            reference_method=reference_method,
            extra_reference_methods=extra_reference_methods_from_config(config),
        )
        print(f"Wrote analysis: {config['experiment']['analysis_dir']}")
        paper_path = write_results_section(
            config["experiment"]["analysis_dir"],
            paper_dir="paper",
            reference_method=reference_method,
        )
        print(f"Wrote paper results: {paper_path}")
        passed = _print_verification(verify_all(args.config))
        if passed:
            return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tabular benchmark paper workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_experiment_parser = subparsers.add_parser("run-experiment")
    run_experiment_parser.add_argument("--config", default="configs/smoke.yaml")
    run_experiment_parser.set_defaults(func=command_run_experiment)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--config", default="configs/smoke.yaml")
    analyze_parser.set_defaults(func=command_analyze)

    paper_parser = subparsers.add_parser("write-paper")
    paper_parser.add_argument("--config", default="configs/smoke.yaml")
    paper_parser.set_defaults(func=command_write_paper)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--config", default="configs/smoke.yaml")
    verify_parser.set_defaults(func=command_verify)

    loop_parser = subparsers.add_parser("run-loop")
    loop_parser.add_argument("--config", default="configs/smoke.yaml")
    loop_parser.add_argument("--max-restarts", type=int, default=0)
    loop_parser.set_defaults(func=command_run_loop)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
