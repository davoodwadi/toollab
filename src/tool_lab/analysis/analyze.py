from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd
import yaml


def load_events(results_file: str | Path) -> pd.DataFrame:
    """Load events.jsonl file into a single DataFrame."""
    path = Path(results_file)
    frames = []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def assistant_events(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["kind"] == "assistant"].copy()


def inspect_events(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["kind"] == "tool") & (df["tool_name"] == "inspect_cell")].copy()


def choice_events(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["kind"] == "tool") & (df["tool_name"] == "submit_choice")].copy()


def _mean_run_length(transitions: pd.Series, label: str) -> float:
    run_lengths = []
    current = 0
    for transition in transitions:
        if transition == label:
            current += 1
            continue
        if current:
            run_lengths.append(current)
            current = 0
    if current:
        run_lengths.append(current)
    return float(sum(run_lengths) / len(run_lengths)) if run_lengths else float("nan")


def _load_analysis_config(config: dict[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if config is None or isinstance(config, dict):
        return config
    path = Path(config)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _option_scores_from_config(config: dict[str, Any] | None) -> pd.DataFrame:
    if not config:
        return pd.DataFrame(columns=["option_id", "option_score"])

    option_scores = pd.DataFrame(config.get("options", []))
    if not option_scores.empty:
        option_scores = option_scores[["id", "base_score"]].rename(
            columns={"id": "option_id", "base_score": "option_score"}
        )
    else:
        option_scores = pd.DataFrame(columns=["option_id", "option_score"])

    cue_scores = pd.DataFrame(config.get("cues", []))
    if not cue_scores.empty and {"option_id", "normative_score"}.issubset(cue_scores.columns):
        cue_scores = (
            cue_scores.groupby("option_id", as_index=False)
            .agg(option_score=("normative_score", "sum"))
        )
    else:
        cue_scores = pd.DataFrame(columns=["option_id", "option_score"])

    if option_scores.empty and cue_scores.empty:
        return pd.DataFrame(columns=["option_id", "option_score"])

    combined = pd.concat([option_scores, cue_scores], ignore_index=True)
    return combined.groupby("option_id", as_index=False).agg(option_score=("option_score", "sum"))


def _merge_choice_score_metrics(
    metrics: pd.DataFrame,
    df: pd.DataFrame,
    config: dict[str, Any] | str | Path | None,
) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["choice_score"] = pd.Series(pd.NA, index=metrics.index, dtype="Float64")
    metrics["best_option_score"] = pd.Series(pd.NA, index=metrics.index, dtype="Float64")
    metrics["choice_regret"] = pd.Series(pd.NA, index=metrics.index, dtype="Float64")
    metrics["attention_to_best_options"] = pd.Series(pd.NA, index=metrics.index, dtype="Float64")
    metrics["examined_best_option"] = pd.Series(pd.NA, index=metrics.index, dtype="boolean")
    metrics["chose_best_option"] = pd.Series(pd.NA, index=metrics.index, dtype="boolean")

    option_scores = _option_scores_from_config(_load_analysis_config(config))
    if option_scores.empty:
        return metrics

    choice_df = choice_events(df)[["session_name", "option_id"]].rename(
        columns={"option_id": "choice_option_id"}
    )
    choice_scores = choice_df.merge(
        option_scores.rename(columns={"option_id": "choice_option_id", "option_score": "choice_score"}),
        on="choice_option_id",
        how="left",
    )

    best_option_score = float(option_scores["option_score"].max())
    best_option_ids = set(
        option_scores.loc[option_scores["option_score"] == best_option_score, "option_id"]
    )
    inspect_df = inspect_events(df)
    if inspect_df.empty:
        best_attention = pd.DataFrame(columns=["session_name", "attention_to_best_options", "examined_best_option"])
    else:
        best_attention = (
            inspect_df.assign(is_best_option=inspect_df["option_id"].isin(best_option_ids))
            .groupby("session_name", as_index=False)
            .agg(
                attention_to_best_options=("is_best_option", "mean"),
                examined_best_option=("is_best_option", "any"),
            )
        )

    metrics = metrics.merge(
        choice_scores[["session_name", "choice_score"]].rename(
            columns={"choice_score": "choice_score_computed"}
        ),
        on="session_name",
        how="left",
    )
    metrics = metrics.merge(
        best_attention.rename(
            columns={
                "attention_to_best_options": "attention_to_best_options_computed",
                "examined_best_option": "examined_best_option_computed",
            }
        ),
        on="session_name",
        how="left",
    )
    metrics["choice_score"] = metrics.pop("choice_score_computed").astype("Float64")
    metrics["attention_to_best_options"] = metrics.pop(
        "attention_to_best_options_computed"
    ).astype("Float64")
    metrics["examined_best_option"] = metrics.pop("examined_best_option_computed").astype(
        "boolean"
    )
    metrics["best_option_score"] = best_option_score
    metrics["choice_regret"] = best_option_score - metrics["choice_score"]
    metrics["chose_best_option"] = (
        metrics["choice_score"].notna() & (metrics["choice_score"] == metrics["best_option_score"])
    ).astype("boolean")
    return metrics


def get_mouselab_metrics(
    df: pd.DataFrame,
    *,
    config: dict[str, Any] | str | Path | None = None,
    total_options: int | None = None,
    total_attributes: int | None = None,
) -> pd.DataFrame:
    """Compute per-session Mouselab process metrics from inspect events."""
    sessions = df[["session_name"]].drop_duplicates().copy()
    inspect_df = inspect_events(df)
    if inspect_df.empty:
        sessions = sessions.assign(
            acquisitions=0,
            reacquisitions=0,
            reacquisition_rate=pd.Series(dtype="float64"),
            n_options_examined=0,
            n_attributes_examined=0,
            unique_cells_examined=0,
            search_depth=pd.Series(dtype="float64"),
            alternative_transitions=0,
            attribute_transitions=0,
            mean_within_option_run_length=pd.Series(dtype="float64"),
            mean_within_attribute_run_length=pd.Series(dtype="float64"),
            payne_index=pd.Series(dtype="float64"),
        )
        return _merge_choice_score_metrics(sessions, df, config)

    inspect_df = inspect_df.sort_values(["session_name", "step_index"]).copy()
    inspect_df["cell_id"] = inspect_df["option_id"] + "::" + inspect_df["attribute_id"]

    inferred_options = int(inspect_df["option_id"].nunique())
    inferred_attributes = int(inspect_df["attribute_id"].nunique())
    matrix_cells = (total_options or inferred_options) * (total_attributes or inferred_attributes)

    metrics = (
        inspect_df.groupby("session_name")
        .agg(
            acquisitions=("tool_name", "size"),
            reacquisitions=("is_revisit", lambda s: int(s.fillna(False).sum())),
            n_options_examined=("option_id", "nunique"),
            n_attributes_examined=("attribute_id", "nunique"),
            unique_cells_examined=("cell_id", "nunique"),
        )
        .reset_index()
    )
    metrics["reacquisition_rate"] = metrics["reacquisitions"] / metrics["acquisitions"].where(
        metrics["acquisitions"] != 0
    )
    metrics["search_depth"] = metrics["unique_cells_examined"] / matrix_cells if matrix_cells else float("nan")

    qualifying = inspect_df[inspect_df["transition"].isin(["alternative", "attribute"])]
    transition_counts = (
        qualifying.groupby(["session_name", "transition"])
        .size()
        .unstack(fill_value=0)
        .rename(
            columns={
                "alternative": "alternative_transitions",
                "attribute": "attribute_transitions",
            }
        )
        .reset_index()
    )

    run_lengths = (
        inspect_df.groupby("session_name")["transition"]
        .agg(
            mean_within_option_run_length=lambda s: _mean_run_length(s, "alternative"),
            mean_within_attribute_run_length=lambda s: _mean_run_length(s, "attribute"),
        )
        .reset_index()
    )

    metrics = sessions.merge(metrics, on="session_name", how="left")
    metrics = metrics.merge(transition_counts, on="session_name", how="left")
    metrics = metrics.merge(run_lengths, on="session_name", how="left").fillna(
        {
            "acquisitions": 0,
            "reacquisitions": 0,
            "n_options_examined": 0,
            "n_attributes_examined": 0,
            "unique_cells_examined": 0,
            "alternative_transitions": 0,
            "attribute_transitions": 0,
        }
    )
    int_columns = [
        "acquisitions",
        "reacquisitions",
        "n_options_examined",
        "n_attributes_examined",
        "unique_cells_examined",
        "alternative_transitions",
        "attribute_transitions",
    ]
    metrics[int_columns] = metrics[int_columns].astype("int64")
    denominator = metrics["alternative_transitions"] + metrics["attribute_transitions"]
    metrics["payne_index"] = (
        metrics["alternative_transitions"] - metrics["attribute_transitions"]
    ) / denominator.where(denominator != 0)
    return _merge_choice_score_metrics(metrics, df, config)


def get_payne_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the classical Payne index per session from inspect transitions."""
    metrics = get_mouselab_metrics(df)
    return metrics[
        [
            "session_name",
            "alternative_transitions",
            "attribute_transitions",
            "payne_index",
        ]
    ].copy()


def session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per session with outcome and cost totals."""
    inspection_counts = (
        inspect_events(df)
        .groupby("session_name")
        .size()
        .rename("n_inspections")
        .reset_index()
    )
    agg = (
        df.groupby("session_name")
        .agg(
            experiment_name=("experiment_name", "first"),
            provider=("provider", "first"),
            model_name=("model_name", "first"),
            choice=("choice", "first"),
            forced_choice=("forced_choice", "first"),
            seed=("seed", "first"),
            started_at=("started_at", "first"),
            finished_at=("finished_at", "first"),
            total_steps=("step_index", "max"),
            cost_usd=("cumulative_cost_usd", "max"),
            remaining_usd=("budget_remaining_usd", "min"),
        )
        .reset_index()
    )
    agg = agg.merge(inspection_counts, on="session_name", how="left")
    agg["n_inspections"] = agg["n_inspections"].fillna(0).astype("int64")
    return agg

if __name__ == '__main__':

    results_dir = Path('/home/dw/github/toollab/results/consumer-choice-mock-mock-v2')
    results_file = results_dir/'events.jsonl'
    config_file = results_dir/'config.json'

    df_raw = load_events(results_file)
    df = df_raw.copy().sort_values(['session_name', 'step_index'])
    # print(f"{len(df)} events across {df['session_name'].nunique()} sessions")
    print(df)
    print(df.columns)
    print(session_summary(df)[['session_name', 'choice', 'forced_choice', 'n_inspections', 'cost_usd']].to_string())
    
    revisits = df.groupby('session_name').agg(
        total_steps=('step_index', max),
        n_revisits=('is_revisit', 'sum')
    )
    # print(revisits)

    transitions = df.groupby('session_name').agg(
        transition=('transition', 'value_counts'),
    )
    print(transitions)

    mouselab_metrics = get_mouselab_metrics(df, config=config_file)
    print(mouselab_metrics.columns)
    print(mouselab_metrics[['session_name', 'acquisitions', 'reacquisitions']])
    print(mouselab_metrics[['session_name', 'n_options_examined', 'n_attributes_examined', 'unique_cells_examined']])
    print(mouselab_metrics[['session_name', 'reacquisition_rate', 'search_depth', 'alternative_transitions', 'attribute_transitions']])
    print(mouselab_metrics[['session_name', 'mean_within_option_run_length', 'mean_within_attribute_run_length']])
    print(mouselab_metrics[['session_name', 'payne_index']])
    print(mouselab_metrics[['session_name', 'choice_score', 'best_option_score', 'choice_regret']])
    print(mouselab_metrics[['session_name', 'attention_to_best_options', 'examined_best_option', 'chose_best_option']])
