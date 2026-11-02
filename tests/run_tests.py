from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    detectors = load_module("test_detectors", HERE / "test_detectors.py")
    pipeline = load_module("test_stage09_pipeline", HERE / "test_stage09_pipeline.py")
    tests = [
        detectors.test_all_detectors_fit_score_predict,
        detectors.test_spatial_detector_responds_to_single_station_outlier,
        detectors.test_seasonal_residual_cusum_is_finite,
        pipeline.test_fusion_uses_validation_only_for_threshold_and_weights,
        pipeline.test_recovery_candidates_mask_predictions_and_select_valid_weights,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} core TrustMeteo-QC tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
