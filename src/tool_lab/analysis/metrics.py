from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any


NUMERIC_RUN_METRICS = [
    "attribute_based_rate",
    "comparability_score",
    "cost_per_unique_cue_usd",
    "distinct_attributes_inspected",
    "model_turn_count",
    "orientation_score",
    "systematic_transition_rate",
    "total_cost_usd",
    "unique_cues_inspected",
    "unique_cues_per_option_avg",
]


def compute_run_metrics(run_record: dict[str, Any]) -> dict[str, Any]:
    trace = run_record["trace"]
    inspect_events = [event for event in trace if event.get("kind") == "inspect" and event.get("status") == "ok"]
    correct_options = set(run_record["correct_options"])
    eligible_options = set(run_record["options"].keys())
    unique_inspections = {}
    by_attribute: dict[str, set[str]] = defaultdict(set)
    by_option: dict[str, set[str]] = defaultdict(set)

    candidate_based_count = 0
    attribute_based_count = 0
    candidate_feasible_count = 0
    attribute_feasible_count = 0
    systematic_count = 0
    revisit_count = 0
    transitions = 0

    for event in inspect_events:
        unique_inspections.setdefault(event["item_id"], event)
        by_attribute[event["attribute_id"]].add(event["option_id"])
        by_option[event["option_id"]].add(event["item_id"])
        transition_type = event.get("transition_type")
        if transition_type != "first":
            transitions += 1
        if transition_type == "attribute_based":
            attribute_based_count += 1
            systematic_count += 1
        elif transition_type == "candidate_based":
            candidate_based_count += 1
            systematic_count += 1
        elif transition_type == "revisit":
            revisit_count += 1

        if event.get("candidate_transition_feasible"):
            candidate_feasible_count += 1
        if event.get("attribute_transition_feasible"):
            attribute_feasible_count += 1

    unique_events = list(unique_inspections.values())
    distinct_attributes = len({event["attribute_id"] for event in unique_events})
    unique_cues = len(unique_events)
    unique_per_option_avg = (
        sum(len(by_option[option_id]) for option_id in eligible_options) / max(len(eligible_options), 1)
    )

    comparable_attributes = sum(
        1 for option_ids in by_attribute.values() if option_ids == eligible_options
    )
    comparability = comparable_attributes / max(len(by_attribute), 1)

    candidate_rate = candidate_based_count / max(candidate_feasible_count, 1)
    attribute_rate = attribute_based_count / max(attribute_feasible_count, 1)
    orientation = candidate_rate - attribute_rate
    systematic_rate = systematic_count / max(transitions, 1)
    total_cost = float(run_record["cumulative_cost_usd"])
    cost_per_unique_cue = total_cost / max(unique_cues, 1)

    return {
        "correct": run_record["choice"] in correct_options if run_record["choice"] else False,
        "distinct_attributes_inspected": distinct_attributes,
        "unique_cues_inspected": unique_cues,
        "unique_cues_per_option_avg": round(unique_per_option_avg, 4),
        "comparability_score": round(comparability, 4),
        "candidate_based_rate": round(candidate_rate, 4),
        "attribute_based_rate": round(attribute_rate, 4),
        "orientation_score": round(orientation, 4),
        "systematic_transition_rate": round(systematic_rate, 4),
        "revisit_count": revisit_count,
        "inspect_count": len(inspect_events),
        "model_turn_count": len(run_record["model_turns"]),
        "total_cost_usd": round(total_cost, 8),
        "cost_per_unique_cue_usd": round(cost_per_unique_cue, 8),
    }


def aggregate_run_records(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not run_records:
        return {}

    _attach_depth_composite(run_records)
    metrics_list = [record["metrics"] for record in run_records]
    vote_distribution = Counter(record["choice"] or "no_choice" for record in run_records)
    correctness_rate = mean(1.0 if metrics["correct"] else 0.0 for metrics in metrics_list)
    summary = {
        "experiment_name": run_records[0]["experiment_name"],
        "provider": run_records[0]["provider"],
        "model_name": run_records[0]["model_name"],
        "matrix_mode": run_records[0]["matrix_mode"],
        "replications": len(run_records),
        "correctness_rate": round(correctness_rate, 4),
        "vote_distribution": dict(vote_distribution),
        "stop_reasons": dict(Counter(record["stop_reason"] for record in run_records)),
    }

    for metric_name in NUMERIC_RUN_METRICS + ["candidate_based_rate", "depth_composite_z"]:
        values = [float(record["metrics"].get(metric_name, 0.0)) for record in run_records]
        summary[f"mean_{metric_name}"] = round(mean(values), 6)
        summary[f"std_{metric_name}"] = round(pstdev(values), 6)

    return summary


def _attach_depth_composite(run_records: list[dict[str, Any]]) -> None:
    keys = [
        "distinct_attributes_inspected",
        "unique_cues_inspected",
        "unique_cues_per_option_avg",
    ]
    means = {key: mean(float(record["metrics"][key]) for record in run_records) for key in keys}
    stds = {key: pstdev(float(record["metrics"][key]) for record in run_records) for key in keys}

    for record in run_records:
        z_scores = []
        for key in keys:
            value = float(record["metrics"][key])
            scale = stds[key]
            z_scores.append(0.0 if scale == 0 else (value - means[key]) / scale)
        record["metrics"]["depth_composite_z"] = round(sum(z_scores) / len(z_scores), 6)
