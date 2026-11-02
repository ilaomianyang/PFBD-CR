"""Compare measured Stage503 single-step resources with the manuscript's
Table 2 claims.

The manuscript claims (Table 2):
  * deep LSTM-AE:  >100 ms single-step latency, >200 MB resident memory
  * Anomaly Transformer: >200 ms, >500 MB

The measurements below are for *this* implementation: 16 features, window 24,
batch 1, one torch thread, i5-14600KF.  They are not measurements of the
original papers' full models, so a mismatch is reported as "not reproduced
under this protocol", never as proof that the manuscript is wrong.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
LABELS = {
    "lstm_ae": "LSTM-AE",
    "usad": "USAD",
    "timesnet": "TimesNet",
    "anomaly_transformer": "Anomaly Transformer",
}
CLAIMS = {
    "lstm_ae": (100.0, 200.0),
    "anomaly_transformer": (200.0, 500.0),
}


def main() -> int:
    path = TABLES / "stage503_resources_raw.csv"
    if not path.exists():
        path = TABLES / "stage503_deep_baseline_resources.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run the resource probe first")
    df = pd.read_csv(path)
    key = "method" if "method" in df.columns else "model"
    rows = []
    for _, r in df.iterrows():
        model = r[key]
        claim_ms, claim_mb = CLAIMS.get(model, (float("nan"), float("nan")))
        peak = float(r["peak_wset_mb"])
        param = float(r["param_mb"])
        latency_ok = bool(peak == peak and claim_ms == claim_ms and float(r["latency_mean_ms"]) > claim_ms)
        memory_ok = bool(peak == peak and claim_mb == claim_mb and peak > claim_mb)
        rows.append({
            "model": model,
            "label": LABELS.get(model, model),
            "latency_mean_ms": float(r["latency_mean_ms"]),
            "latency_p50_ms": float(r["latency_p50_ms"]),
            "latency_p95_ms": float(r["latency_p95_ms"]),
            "latency_max_ms": float(r["latency_max_ms"]),
            "param_mb": param,
            "peak_wset_mb": peak,
            "rss_before_model_mb": float(r["rss_before_model_mb"]),
            "rss_after_model_mb": float(r["rss_after_model_mb"]),
            "paper_latency_claim_ms": claim_ms,
            "paper_memory_claim_mb": claim_mb,
            "latency_claim_reproduced": latency_ok,
            "memory_claim_reproduced_process_peak": memory_ok,
            "latency_ratio_measured_over_claim": (
                float(r["latency_mean_ms"]) / claim_ms if claim_ms == claim_ms else float("nan")
            ),
            "memory_ratio_process_over_claim": (
                peak / claim_mb if claim_mb == claim_mb else float("nan")
            ),
            "hardware": r.get("processor", ""),
            "platform": r.get("platform", ""),
            "threads": int(r.get("threads", 1)),
            "length": int(r.get("length", 24)),
            "features": int(r.get("features", 16)),
        })
    out = pd.DataFrame(rows)
    family = {
        "lstm_ae": "deep_reconstruction",
        "usad": "deep_reconstruction",
        "timesnet": "deep_convolution",
        "anomaly_transformer": "deep_transformer",
    }
    implementation = {
        "lstm_ae": "PyTorch LSTM-AE (h64/z16, denoise 0.1)",
        "usad": "PyTorch USAD (latent 8, min-max input)",
        "timesnet": "PyTorch TimesNet (d32, top-3 periods)",
        "anomaly_transformer": "PyTorch Anomaly Transformer (d64, 4 heads, 1 layer)",
    }
    out["family"] = out["model"].map(family)
    out["implementation"] = out["model"].map(implementation)
    out["status"] = "OK"
    out["parameter_count"] = out["param_mb"] * (1024.0 ** 2) / 4.0
    out["method"] = out["model"]
    out.to_csv(TABLES / "stage503_deep_baseline_resources.csv", index=False, encoding="utf-8-sig")
    out.to_csv(TABLES / "stage506_resource_claims.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Stage503 单步时延与内存：对 Table 2 声称的实测核对",
        "",
        "> 口径：batch=1，16 维特征，窗口 24，PyTorch 单线程；warmup 200 步，计时 2000 步；进程峰值工作集含 Python/PyTorch 运行时。",
        "> 硬件：Intel Core i5-14600KF（非论文中的 i7-12700H）。",
        "",
        "| 模型 | 均值 ms | P95 ms | 最大 ms | 参数 MB | 进程峰值 MB | 论文时延声称 | 论文内存声称 | 时延是否复现 | 进程峰值是否超过声称 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for _, r in out.iterrows():
        lat_claim = "—" if r["paper_latency_claim_ms"] != r["paper_latency_claim_ms"] else f">{r['paper_latency_claim_ms']:.0f} ms"
        mem_claim = "—" if r["paper_memory_claim_mb"] != r["paper_memory_claim_mb"] else f">{r['paper_memory_claim_mb']:.0f} MB"
        lat_flag = "是" if r["latency_claim_reproduced"] else "否"
        mem_flag = "是" if r["memory_claim_reproduced_process_peak"] else "否"
        lines.append(
            f"| {r['label']} | {r['latency_mean_ms']:.3f} | {r['latency_p95_ms']:.3f} | "
            f"{r['latency_max_ms']:.3f} | {r['param_mb']:.4f} | {r['peak_wset_mb']:.1f} | "
            f"{lat_claim} | {mem_claim} | {lat_flag} | {mem_flag} |"
        )
    lines += [
        "",
        "## 结论（按实测口径）",
        "",
        "- **LSTM-AE**：单步均值约 {:.2f} ms，远低于“>100 ms”；进程峰值 {:.0f} MB，超过“>200 MB”，但其中绝大部分是 Python/PyTorch 运行时，参数本身仅 {:.4f} MB。".format(
            float(out.loc[out["model"].eq("lstm_ae"), "latency_mean_ms"].iloc[0]),
            float(out.loc[out["model"].eq("lstm_ae"), "peak_wset_mb"].iloc[0]),
            float(out.loc[out["model"].eq("lstm_ae"), "param_mb"].iloc[0]),
        ),
        "- **Anomaly Transformer**：单步均值约 {:.2f} ms，远低于“>200 ms”；进程峰值 {:.0f} MB，低于“>500 MB”。".format(
            float(out.loc[out["model"].eq("anomaly_transformer"), "latency_mean_ms"].iloc[0]),
            float(out.loc[out["model"].eq("anomaly_transformer"), "peak_wset_mb"].iloc[0]),
        ),
        "- 因此，在 **本实现/本硬件/本窗口** 下，Table 2 的 `>100 ms`、`>200 ms`、`>500 MB` 没有被复现；只有 LSTM-AE 的进程峰值 >200 MB 成立，但不能把它解释为模型权重常驻。",
        "- 可能原因：原论文的模型规模、窗口长度、批处理/框架、部署设备与本文不同；必须把这组数当作“本实现实测”，不能反过来宣称原论文数字错误。",
        "- 与 Table 13 的本系统 5.053 ms / 83.5 MB 相比：本实现的深度基线单步仍为毫秒级，说明“深度模型必然 >100 ms”这一对比矩阵论断缺乏实测支持；论文应改为“在本实现与给定硬件下”的限定表述，或引用原论文的原始测量条件。",
        "",
    ]
    (REPORTS / "stage506_resource_claims.md").write_text("\n".join(lines), encoding="utf-8")
    print(out[["model", "latency_mean_ms", "latency_p95_ms", "param_mb", "peak_wset_mb",
               "paper_latency_claim_ms", "paper_memory_claim_mb",
               "latency_claim_reproduced", "memory_claim_reproduced_process_peak"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
