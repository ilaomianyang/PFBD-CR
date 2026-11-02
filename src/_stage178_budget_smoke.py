import sys

import numpy as np

sys.path.insert(0, r"experiments\trustmeteo_qc\src")
import stage178_noaa_station_calibrated_alarm_budget as s  # noqa: E402


def main() -> int:
    for n in [1, 2, 3, 10, 11, 100]:
        values = np.arange(n, dtype=float)
        for budget in range(n + 1):
            mask = s.top_k_mask(values, budget)
            assert int(mask.sum()) == budget
    stations = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 2)
    value_sets = [
        np.array([1, 2, 3, 9, 8, 7, 6, 5, 4, 0], dtype=float),
        np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=float),
        np.ones(10, dtype=float),
        np.array([100, 99, 98, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    ]
    for budget in range(len(stations) + 1):
        for values in value_sets:
            for rule in s.ALLOCATION_RULES:
                mask = s.station_budget_mask(
                    stations, values, budget, rule=rule
                )
                assert int(mask.sum()) == budget
                assert set(mask.tolist()).issubset({False, True})
    print("budget smoke PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
