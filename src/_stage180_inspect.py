import pandas as pd

root = "experiments/trustmeteo_qc/tables/"
summary = pd.read_csv(
    root + "stage180_noaa_station_budget_row_tail_audit_summary.csv"
)
event = pd.read_csv(
    root + "stage180_noaa_station_budget_row_tail_audit_event_aggregate.csv"
)
station_type = pd.read_csv(
    root + "stage180_noaa_station_budget_row_tail_audit_station_type_detail.csv"
)
station = pd.read_csv(
    root + "stage180_noaa_station_budget_row_tail_audit_station_detail.csv"
)
print("SUMMARY")
print(
    summary.groupby(["q", "allocation_rule"])[
        [
            "selected_attack_rows",
            "attack_recall",
            "selected_normal_rate",
        ]
    ]
    .mean()
    .reindex(
        pd.MultiIndex.from_product(
            [[0.995, 0.999], ["size", "uniform", "risk"]],
            names=["q", "allocation_rule"],
        )
    )
    .to_string()
)
print("\nEVENT AGGREGATE")
print(event.to_string(index=False))
print("\nTYPE RECALL")
type_summary = (
    station_type.groupby(["q", "allocation_rule", "attack_type"])[
        ["attack_rows", "selected_attack_rows"]
    ]
    .sum()
    .reset_index()
)
type_summary["recall"] = (
    type_summary["selected_attack_rows"] / type_summary["attack_rows"]
)
print(
    type_summary.pivot(
        index=["q", "attack_type"],
        columns="allocation_rule",
        values="recall",
    ).to_string()
)
print("\nSTATION COUNT")
print(
    f"station_detail={len(station)}, "
    f"station_type_detail={len(station_type)}"
)
