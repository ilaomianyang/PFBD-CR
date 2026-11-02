"""Stage503: deep anomaly-detection baselines on the frozen V6 protocol.

Adds LSTM-AE, USAD, TimesNet and a compact Anomaly Transformer to the exact
16-dimensional feature matrix, provider splits and validation-only
FAR = 0.0065 calibration used by the V6 locked evaluation (Stage349).

Design constraints
------------------
* Same feature space as Table 3/4: 3 detector components + 13 morphology
  features (``stage248.feature_matrix``).
* Same provider split, same V6 attack generator, same five seeds.
* Thresholds use valid normal rows only; test labels are evaluation-only.
* No point adjustment, no test-time threshold selection.
* Deep models are trained once per provider on clean training rows; the
  attack injection only changes the validation/test streams.
* Scoring is causal: a window ending at row t uses rows <= t only.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL_PATH = ROOT / "confirmatory_package" / "simulated_attack_v6" / "protocol_freeze.json"

SEEDS = [3407, 6257, 7919, 104729, 20260715]
FAR = 0.0065
VALID_OFFSET = 1010001
TEST_OFFSET = 1020001
LENGTH = 24
N_FEATURES = 16
BATCH = 256
LR = 1e-3
CLIP = 10.0
MODEL_SEED = 20260922
MODEL_EPOCHS = {"lstm_ae": 30, "usad": 10, "timesnet": 20, "anomaly_transformer": 20}
LSTM_DENOISE = 0.1

FAMILY = {
    "lstm_ae": "deep_reconstruction",
    "usad": "deep_reconstruction",
    "timesnet": "deep_reconstruction",
    "anomaly_transformer": "deep_reconstruction",
}
LABEL_CN = {
    "lstm_ae": "LSTM-AE 重建",
    "usad": "USAD",
    "timesnet": "TimesNet",
    "anomaly_transformer": "Anomaly Transformer",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LSTMAE(nn.Module):
    def __init__(self, c: int = N_FEATURES, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(c, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.decoder = nn.LSTM(latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.encoder(x)
        z = self.to_latent(hn[-1])
        z_seq = z.unsqueeze(1).expand(-1, x.size(1), -1)
        h, _ = self.decoder(z_seq)
        return self.out(h)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        rec = ((self.forward(x) - x) ** 2).mean(dim=-1)
        return rec[:, -1]


class USAD(nn.Module):
    """USAD with the official three-linear-layer encoder/decoders.

    The inputs are min-max scaled to [0, 1] per provider using training
    windows only, matching the official USAD setup; the decoders therefore end
    in Sigmoid and the loss is MSE in the scaled space.
    """

    def __init__(self, length: int, c: int = N_FEATURES, latent: int = 8):
        super().__init__()
        w = length * c
        self.encoder = nn.Sequential(
            nn.Linear(w, w // 2), nn.ReLU(True),
            nn.Linear(w // 2, w // 4), nn.ReLU(True),
            nn.Linear(w // 4, latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent, w // 4), nn.ReLU(True),
            nn.Linear(w // 4, w // 2), nn.ReLU(True),
            nn.Linear(w // 2, w), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent, w // 4), nn.ReLU(True),
            nn.Linear(w // 4, w // 2), nn.ReLU(True),
            nn.Linear(w // 2, w), nn.Sigmoid(),
        )
        self.register_buffer("x_min", torch.zeros(1, c))
        self.register_buffer("x_max", torch.ones(1, c))

    def fit_scaler(self, x: np.ndarray) -> None:
        flat = np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
        lo = flat.min(axis=0)
        hi = flat.max(axis=0)
        span = np.where(hi - lo < 1e-6, 1.0, hi - lo)
        self.x_min.copy_(torch.from_numpy(lo).view(1, -1))
        self.x_max.copy_(torch.from_numpy(lo + span).view(1, -1))

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_min) / (self.x_max - self.x_min + 1e-6)

    def forward(self, x: torch.Tensor):
        flat = self._scale(x).reshape(x.size(0), -1)
        z = self.encoder(flat)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3

    def training_step(self, batch: torch.Tensor, epoch: int):
        w1, w2, w3 = self.forward(batch)
        flat = self._scale(batch).reshape(batch.size(0), -1)
        n = float(max(epoch, 1))
        loss1 = (1.0 / n) * F.mse_loss(flat, w1) + (1.0 - 1.0 / n) * F.mse_loss(flat, w3)
        loss2 = (1.0 / n) * F.mse_loss(flat, w2) - (1.0 - 1.0 / n) * F.mse_loss(flat, w3)
        return loss1, loss2

    def score(self, x: torch.Tensor) -> torch.Tensor:
        w1, w2, _ = self.forward(x)
        flat = self._scale(x).reshape(x.size(0), -1)
        e1 = ((w1 - flat) ** 2).reshape(x.size(0), x.size(1), -1).mean(dim=-1)[:, -1]
        e2 = ((w2 - flat) ** 2).reshape(x.size(0), x.size(1), -1).mean(dim=-1)[:, -1]
        return 0.5 * e1 + 0.5 * e2


class InceptionBlock(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv3 = nn.Conv2d(channels, hidden, 3, padding=1)
        self.conv5 = nn.Conv2d(channels, hidden, 5, padding=2)
        self.proj = nn.Conv2d(3 * hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.conv1(x), self.conv3(x), self.conv5(x)], dim=1)
        return x + F.gelu(self.proj(y))


class TimesNet(nn.Module):
    """Compact TimesNet: fixed dominant periods from train data.

    Periods are estimated once from the clean training windows and frozen for
    validation/test scoring, so no test-time information enters the model.
    """

    def __init__(self, length: int, c: int = N_FEATURES, d: int = 32, periods: tuple[int, ...] | None = None):
        super().__init__()
        self.length = int(length)
        self.periods = tuple(periods or (length, max(2, length // 2), max(2, length // 3)))
        self.embed = nn.Linear(c, d)
        self.blocks = nn.ModuleList([InceptionBlock(d, d) for _ in self.periods])
        self.weights = nn.Parameter(torch.zeros(len(self.periods)))
        self.out = nn.Linear(d, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        h = self.embed(x)
        outs: list[torch.Tensor] = []
        for period, block in zip(self.periods, self.blocks):
            n = int(math.ceil(length / period))
            pad = n * period - length
            xp = F.pad(h, (0, 0, 0, pad))
            xp = xp.reshape(b, n, period, -1).permute(0, 3, 1, 2)
            y = block(xp).permute(0, 2, 3, 1).reshape(b, n * period, -1)[:, :length, :]
            outs.append(y)
        w = torch.softmax(self.weights, dim=0)
        return self.out(sum(w[i] * outs[i] for i in range(len(outs))))

    def score(self, x: torch.Tensor) -> torch.Tensor:
        rec = ((self.forward(x) - x) ** 2).mean(dim=-1)
        return rec[:, -1]


def _kl_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-9)
    q = q.clamp_min(1e-9)
    return (p * (p.log() - q.log())).sum(dim=-1)


class AssociationAttention(nn.Module):
    def __init__(self, d: int, heads: int, length: int):
        super().__init__()
        self.heads = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.sigma = nn.Parameter(torch.ones(heads))
        dist = torch.arange(length, dtype=torch.float32)
        self.register_buffer("dist2", (dist[:, None] - dist[None, :]) ** 2)

    def forward(self, x: torch.Tensor):
        b, length, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, length, self.heads, self.dh).transpose(1, 2)
        k = k.view(b, length, self.heads, self.dh).transpose(1, 2)
        v = v.view(b, length, self.heads, self.dh).transpose(1, 2)
        series = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), dim=-1)
        prior = torch.exp(
            -self.dist2[None, None] / (2.0 * self.sigma[None, :, None, None] ** 2 + 1e-6)
        )
        prior = prior / prior.sum(dim=-1, keepdim=True)
        out = (series @ v).transpose(1, 2).reshape(b, length, d)
        return self.proj(out), series, prior


class ATLayer(nn.Module):
    def __init__(self, d: int, heads: int, length: int):
        super().__init__()
        self.attn = AssociationAttention(d, heads, length)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor):
        y, series, prior = self.attn(self.norm1(x))
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x, series, prior


class AnomalyTransformer(nn.Module):
    """Compact Anomaly Transformer (Xu et al., ICLR 2022).

    Prior association is a learnable Gaussian kernel over relative positions;
    series association is the attention map.  Training uses the association
    discrepancy as a minimax regulariser; scoring multiplies the reconstruction
    error by exp(-discrepancy), the streaming analogue of the paper's
    ``softmax(-discrepancy)`` weighting.
    """

    def __init__(self, length: int, c: int = N_FEATURES, d: int = 64, heads: int = 4, layers: int = 1):
        super().__init__()
        self.length = int(length)
        self.embed = nn.Linear(c, d)
        pe = torch.zeros(length, d)
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))
        self.encoder = nn.ModuleList([ATLayer(d, heads, length) for _ in range(layers)])
        self.decoder = ATLayer(d, heads, length)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, c)

    def forward(self, x: torch.Tensor):
        h = self.embed(x) + self.pe
        series_list = []
        prior_list = []
        for layer in self.encoder:
            h, series, prior = layer(h)
            series_list.append(series)
            prior_list.append(prior)
        h, _, _ = self.decoder(h)
        return self.out(self.norm(h)), series_list, prior_list

    def score(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, series_list, prior_list = self.forward(x)
        rec = ((x_hat - x) ** 2).mean(dim=-1)[:, -1]
        disc = torch.zeros(x.size(0), device=x.device)
        for series, prior in zip(series_list, prior_list):
            prior = prior / prior.sum(dim=-1, keepdim=True)
            disc = disc + 0.5 * (
                _kl_rows(series, prior.detach()).mean(dim=(1, 2))
                + _kl_rows(prior.detach(), series).mean(dim=(1, 2))
            )
        disc = disc / max(len(series_list), 1)
        return rec * torch.exp(-disc)


def dominant_periods(x: np.ndarray, k: int = 3) -> tuple[int, ...]:
    length = x.shape[1]
    signal = np.asarray(x, dtype=np.float64).mean(axis=(0, 2))
    amplitude = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(length, d=1.0)
    order = np.argsort(amplitude[1:])[::-1] + 1
    periods: list[int] = []
    for idx in order:
        if freqs[idx] <= 0:
            continue
        period = int(round(1.0 / freqs[idx]))
        if 2 <= period <= length and period not in periods:
            periods.append(period)
        if len(periods) >= k:
            break
    if not periods:
        periods = [length]
    return tuple(periods)


def build_model(
    name: str,
    length: int,
    c: int = N_FEATURES,
    periods: tuple[int, ...] | None = None,
) -> nn.Module:
    if name == "lstm_ae":
        return LSTMAE(c=c)
    if name == "usad":
        return USAD(length=length, c=c)
    if name == "timesnet":
        return TimesNet(length=length, c=c, periods=periods)
    if name == "anomaly_transformer":
        return AnomalyTransformer(length=length, c=c)
    raise ValueError(f"unknown model: {name}")


# ---------------------------------------------------------------------------
# Sequence windows
# ---------------------------------------------------------------------------


def station_groups(frame) -> list[tuple[int, int]]:
    ids = frame["station_id"].astype(str).to_numpy()
    if len(ids) == 0:
        return []
    change = np.r_[True, ids[1:] != ids[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(ids)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _windows(n: int, length: int):
    if n < length:
        return np.empty((0, length), dtype=int)
    return np.lib.stride_tricks.sliding_window_view(np.arange(n, dtype=int), length)


def collect_train_windows(matrix: np.ndarray, frame, length: int, limit: int, seed: int) -> np.ndarray:
    split = frame["split"].eq("train").to_numpy()
    obs = frame["observation_present"].to_numpy(dtype=bool)
    groups = station_groups(frame)
    rng = np.random.default_rng(int(seed))
    per_station = max(1, int(math.ceil(limit / max(len(groups), 1))))
    chunks: list[np.ndarray] = []
    for start, end in groups:
        n = end - start
        idx = _windows(n, length)
        if idx.shape[0] == 0:
            continue
        gpos = start + idx
        keep = split[gpos].all(axis=1) & obs[gpos].all(axis=1)
        cand = idx[keep]
        if cand.shape[0] == 0:
            continue
        if cand.shape[0] > per_station:
            chosen = rng.choice(cand.shape[0], size=per_station, replace=False)
            cand = cand[chosen]
        chunks.append(np.asarray(matrix[start + cand], dtype=np.float32))
    if not chunks:
        raise RuntimeError("no training windows available")
    x = np.concatenate(chunks, axis=0)
    return np.clip(x, -CLIP, CLIP).astype(np.float32)


def iter_split_windows(matrix: np.ndarray, frame, split: str, length: int):
    split_mask = frame["split"].eq(split).to_numpy()
    obs = frame["observation_present"].to_numpy(dtype=bool)
    for start, end in station_groups(frame):
        idx = _windows(end - start, length)
        if idx.shape[0] == 0:
            continue
        gpos = start + idx
        keep = split_mask[gpos[:, -1]] & obs[gpos].all(axis=1)
        cand = idx[keep]
        if cand.shape[0] == 0:
            continue
        x = np.clip(np.asarray(matrix[start + cand], dtype=np.float32), -CLIP, CLIP)
        yield start + cand[:, -1], x


def score_split(model: nn.Module, matrix: np.ndarray, frame, split: str, length: int, batch: int = 2048) -> np.ndarray:
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for end_positions, x in iter_split_windows(matrix, frame, split, length):
            parts: list[np.ndarray] = []
            for i in range(0, len(x), batch):
                xb = torch.from_numpy(x[i : i + batch])
                parts.append(model.score(xb).cpu().numpy().astype(np.float64))
            scores[end_positions] = np.concatenate(parts)
    return scores


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    name: str,
    x: np.ndarray,
    epochs: int,
    seed: int,
    periods: tuple[int, ...] | None = None,
) -> nn.Module:
    epochs = int(epochs) if int(epochs) > 0 else int(MODEL_EPOCHS.get(name, 20))
    torch.manual_seed(int(seed))
    model = build_model(name, x.shape[1], periods=periods)
    model.train()
    rng = np.random.default_rng(int(seed))
    n = len(x)
    if name == "usad":
        model.fit_scaler(x)
        opt1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=LR)
        opt2 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=LR)
        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n)
            for i in range(0, n, BATCH):
                batch = torch.from_numpy(x[perm[i : i + BATCH]])
                loss1, loss2 = model.training_step(batch, epoch)
                opt1.zero_grad()
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt1.step()
                loss1, loss2 = model.training_step(batch, epoch)
                opt2.zero_grad()
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt2.step()
            print(f"    {name} epoch {epoch}/{epochs}", flush=True)
        model.eval()
        return model

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, BATCH):
            batch = torch.from_numpy(x[perm[i : i + BATCH]])
            opt.zero_grad()
            if name == "anomaly_transformer":
                x_hat, series_list, prior_list = model(batch)
                rec = F.mse_loss(x_hat, batch)
                series_loss = 0.0
                prior_loss = 0.0
                for series, prior in zip(series_list, prior_list):
                    prior = prior / prior.sum(dim=-1, keepdim=True)
                    series_loss = series_loss + (
                        _kl_rows(series, prior.detach()).mean()
                        + _kl_rows(prior.detach(), series).mean()
                    )
                    prior_loss = prior_loss + (
                        _kl_rows(prior, series.detach()).mean()
                        + _kl_rows(series.detach(), prior).mean()
                    )
                series_loss = series_loss / max(len(series_list), 1)
                prior_loss = prior_loss / max(len(prior_list), 1)
                loss = rec - 3.0 * series_loss + 3.0 * prior_loss
            elif name == "lstm_ae":
                noisy = batch + LSTM_DENOISE * torch.randn_like(batch)
                loss = F.mse_loss(model(noisy), batch)
            else:
                loss = F.mse_loss(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(batch)
        print(f"    {name} epoch {epoch}/{epochs} loss={total / n:.6f}", flush=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def normal_quantile_threshold(scores: np.ndarray, normal_mask: np.ndarray, far: float) -> float:
    """Quantile threshold over finite normal-row scores only.

    Stage207's ``far_threshold`` takes a pandas Series; the deep baseline
    scores are plain arrays, so use the same rule locally.
    """
    values = np.asarray(scores, dtype=float)[np.asarray(normal_mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("no finite normal scores for calibration")
    return float(np.quantile(values, 1.0 - float(far)))


def scenario_rows(frame, pred: np.ndarray, scores: np.ndarray, source: str, seed: int, method: str) -> list[dict]:
    rows: list[dict] = []
    attack = frame["is_attack"].to_numpy(dtype=bool)
    scenarios = frame["attack_scenario"].astype(str).to_numpy()
    for name in sorted(set(scenarios[attack])) :
        mask = attack & (scenarios == name)
        if not mask.any():
            continue
        work = frame.loc[mask, ["attack_event_id"]].copy()
        work["predicted"] = pred[mask]
        grouped = work.groupby("attack_event_id", sort=True)["predicted"].any()
        tp = int(pred[mask].sum())
        fn = int(mask.sum()) - tp
        fp = int((pred & ~attack).sum())
        tn = int((~pred & ~attack).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        y_true = attack[mask]
        y_score = scores[mask]
        finite = np.isfinite(y_score)
        if finite.any() and y_true[finite].any() and (~y_true[finite]).any():
            from sklearn.metrics import average_precision_score, roc_auc_score

            roc_auc = float(roc_auc_score(y_true[finite], y_score[finite]))
            average_precision = float(average_precision_score(y_true[finite], y_score[finite]))
        else:
            roc_auc = float("nan")
            average_precision = float("nan")
        rows.append({
            "source": source,
            "seed": int(seed),
            "method": method,
            "scenario": name,
            "rows": int(len(frame)),
            "attack_rows": int(mask.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "row_recall": float(recall),
            "f1": float(f1),
            "false_alarm_rate": float(fp / max(int((~attack).sum()), 1)),
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "detected_rows": tp,
            "events": int(len(grouped)),
            "detected_events": int(grouped.sum()),
            "event_coverage": float(grouped.mean()) if len(grouped) else 0.0,
            "mean_score_attack": float(np.nanmean(y_score)) if np.isfinite(y_score).any() else float("nan"),
        })
    return rows


def eval_split(
    source: str,
    seed: int,
    method: str,
    frame,
    scores: np.ndarray,
    threshold: float,
    metrics_fn,
    event_fn,
) -> dict:
    obs = frame["observation_present"].to_numpy(dtype=bool)
    pred = obs & np.isfinite(scores) & (scores >= threshold)
    met = metrics_fn(frame["is_attack"].to_numpy(dtype=bool), pred, scores)
    ev = event_fn(frame, pred)
    return {
        "source": source,
        "seed": int(seed),
        "method": method,
        "threshold": float(threshold),
        **{k: met[k] for k in ("rows", "attack_rows", "precision", "recall", "f1", "false_alarm_rate", "roc_auc", "average_precision")},
        **{k: ev[k] for k in ("events", "detected_events", "event_coverage", "row_recall")},
    }


def evaluate_provider(
    source: str,
    models: list[str],
    seeds: list[int],
    epochs: int,
    window_limit: int,
    length: int,
    frozen: dict,
    scenarios: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    from stage207_dwd_controlled_attack_evaluation import detector_frame, event_metrics, far_threshold, fit_detectors, metrics, score_components
    from stage240_nonstuck_balanced_selection import PROTOCOLS, make_frame
    from stage244_cross_provider_nonstuck_candidate_features import candidate_features
    from stage248_cross_provider_fair_baseline_audit import feature_matrix
    import stage320_simulated_attack_confirmatory_evaluation as simbase

    protocol = load_json(PROTOCOLS[source])
    print(f"[{source}] loading frame", flush=True)
    frame = detector_frame(make_frame(source, protocol)).reset_index(drop=True)
    detectors = fit_detectors(frame, protocol)
    train_normal = (frame["split"].eq("train") & frame["observation_present"]).to_numpy(dtype=bool)
    print(f"[{source}] scoring detector components", flush=True)
    components = score_components(frame, detectors)
    candidates = candidate_features(frame, train_normal)
    matrix, names = feature_matrix(components, candidates, train_normal)
    print(f"[{source}] feature matrix {matrix.shape}, {len(names)} features", flush=True)
    x_train = collect_train_windows(matrix, frame, length, window_limit, MODEL_SEED)
    print(f"[{source}] train windows {x_train.shape}", flush=True)
    periods = dominant_periods(x_train, k=3) if "timesnet" in models else None
    if periods:
        print(f"[{source}] TimesNet periods {periods}", flush=True)

    trained: dict[str, nn.Module] = {}
    for name in models:
        started = time.perf_counter()
        print(f"[{source}] training {name}", flush=True)
        model = train_model(name, x_train, epochs, MODEL_SEED, periods=periods)
        trained[name] = model
        model_dir = TABLES / "stage503_models"
        model_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_dir / f"{source.replace(' ', '_')}_{name}.pt")
        print(f"[{source}] {name} trained in {time.perf_counter() - started:.1f}s", flush=True)

    detail: list[dict] = []
    scenarios_out: list[dict] = []
    for seed in seeds:
        print(f"[{source}] seed {seed}: generating attacks", flush=True)
        valid_blocks = simbase.make_simulated_blocks(frame, frozen, int(seed), "valid", VALID_OFFSET)
        test_blocks = simbase.make_simulated_blocks(frame, frozen, int(seed), "test", TEST_OFFSET)
        valid = simbase.inject_simulated_attack(frame, valid_blocks, scenarios).reset_index(drop=True)
        test = simbase.inject_simulated_attack(frame, test_blocks, scenarios).reset_index(drop=True)

        thresholds: dict[str, float] = {}
        for attacked, split in ((valid, "valid"), (test, "test")):
            print(f"[{source}] seed {seed} {split}: features + scoring", flush=True)
            comp = score_components(attacked, detectors)
            cand = candidate_features(attacked, train_normal)
            mat, _ = feature_matrix(comp, cand, train_normal)
            split_frame = attacked.loc[attacked["split"].eq(split)].reset_index(drop=True)
            split_mask = attacked["split"].eq(split).to_numpy()
            scores_by_model = {name: score_split(model, mat, attacked, split, length)[split_mask] for name, model in trained.items()}
            normal_mask = (split_frame["observation_present"] & ~split_frame["is_attack"]).to_numpy(dtype=bool)
            for name, scores in scores_by_model.items():
                # Calibrate on valid normal rows only; reuse that threshold on test.
                if split == "valid":
                    thresholds[name] = normal_quantile_threshold(scores, normal_mask, FAR)
                threshold = thresholds[name]
                row = eval_split(source, seed, name, split_frame, scores, threshold, metrics, event_metrics)
                row["split"] = split
                if split == "valid":
                    row["valid_only"] = True
                detail.append(row)
                if split == "test":
                    scenarios_out.extend(scenario_rows(split_frame, (split_frame["observation_present"] & np.isfinite(scores) & (scores >= threshold)).to_numpy(), scores, source, seed, name))
                print(
                    f"[{source}] seed {seed} {split} {name}: "
                    f"f1={row['f1']:.4f} cov={row['event_coverage']:.4f} far={row['false_alarm_rate']:.4f}",
                    flush=True,
                )
    return detail, scenarios_out, [{"source": source, "periods": periods, "train_windows": int(len(x_train))}]


# ---------------------------------------------------------------------------
# Resource probe (run in a subprocess)
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    import psutil

    return float(psutil.Process().memory_info().rss) / (1024.0 ** 2)


def _peak_wset_mb() -> float:
    import ctypes
    import ctypes.wintypes as wt

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD),
            ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PMC()
    counters.cb = ctypes.sizeof(PMC)
    kernel32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        raise RuntimeError("GetProcessMemoryInfo failed")
    return float(counters.PeakWorkingSetSize) / (1024.0 ** 2)


def resource_probe(name: str, length: int) -> int:
    import platform

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    periods = None
    if name == "timesnet":
        periods = (length, max(2, length // 2), max(2, length // 3))
    rss_before = _rss_mb()
    model = build_model(name, length, periods=periods)
    model.eval()
    params = int(sum(p.numel() for p in model.parameters()))
    rss_after = _rss_mb()
    x = torch.randn(1, length, N_FEATURES)
    with torch.inference_mode():
        for _ in range(200):
            model(x)
        times = np.empty(2000, dtype=np.float64)
        for i in range(2000):
            started = time.perf_counter_ns()
            model(x)
            times[i] = (time.perf_counter_ns() - started) / 1e6
    payload = {
        "method": name,
        "length": int(length),
        "features": N_FEATURES,
        "params": params,
        "param_mb": params * 4.0 / (1024.0 ** 2),
        "latency_mean_ms": float(times.mean()),
        "latency_p50_ms": float(np.quantile(times, 0.50)),
        "latency_p95_ms": float(np.quantile(times, 0.95)),
        "latency_max_ms": float(times.max()),
        "rss_before_model_mb": rss_before,
        "rss_after_model_mb": rss_after,
        "peak_wset_mb": _peak_wset_mb(),
        "threads": 1,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    print("RESOURCE_JSON " + json.dumps(payload))
    return 0


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------


def aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["precision", "recall", "f1", "false_alarm_rate", "roc_auc", "average_precision", "event_coverage", "row_recall"]
    summary = detail.groupby("method", sort=False)[metric_cols].mean().reset_index()
    summary["runs"] = detail.groupby("method", sort=False).size().to_numpy()
    summary["family"] = summary["method"].map(FAMILY).fillna("existing")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=["MeteoSwiss", "KNMI", "ECCC", "NOAA ISD"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--models", nargs="+", default=["lstm_ae", "usad", "timesnet", "anomaly_transformer"])
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--windows", type=int, default=20000)
    parser.add_argument("--length", type=int, default=LENGTH)
    parser.add_argument("--tag", default="")
    parser.add_argument("--skip-resources", action="store_true")
    parser.add_argument("--resource-only", action="store_true")
    parser.add_argument("--model", default="lstm_ae")
    args = parser.parse_args()

    if args.resource_only:
        return resource_probe(args.model, args.length)

    import pandas as pd
    from stage207_dwd_controlled_attack_evaluation import event_metrics, metrics  # noqa: F401
    import stage320_simulated_attack_confirmatory_evaluation as simbase
    from stage326_simulated_attack_v2_evaluation import _internal_type

    simbase._scenario_base_type = _internal_type
    frozen = load_json(PROTOCOL_PATH)
    scenarios = {str(item["name"]): item for item in frozen["attack_scenarios"]}
    tag = f"_{args.tag}" if args.tag else ""
    started = time.perf_counter()
    all_detail: list[dict] = []
    all_scenarios: list[dict] = []
    provider_meta: list[dict] = []

    for source in args.providers:
        detail, scen, meta = evaluate_provider(
            source=source,
            models=args.models,
            seeds=args.seeds,
            epochs=args.epochs,
            window_limit=args.windows,
            length=args.length,
            frozen=frozen,
            scenarios=scenarios,
        )
        all_detail.extend(detail)
        all_scenarios.extend(scen)
        provider_meta.extend(meta)
        pd.DataFrame(detail).to_csv(TABLES / f"stage503_detail_{source.replace(' ', '_')}{tag}.csv", index=False)

    detail_df = pd.DataFrame(all_detail)
    detail_df.to_csv(TABLES / f"stage503_deep_baseline_detail{tag}.csv", index=False)
    scen_df = pd.DataFrame(all_scenarios)
    scen_df.to_csv(TABLES / f"stage503_deep_baseline_scenario{tag}.csv", index=False)
    summary = aggregate(detail_df[detail_df["split"].eq("test")])
    summary.to_csv(TABLES / f"stage503_deep_baseline_summary{tag}.csv", index=False)

    # Merge with the frozen Stage349 summary for a Table-4-shaped view.
    old_path = TABLES / "stage349_v6_locked_evaluation_detail.csv"
    combined = summary[["method", "precision", "row_recall", "f1", "event_coverage", "average_precision", "false_alarm_rate", "roc_auc"]].copy()
    if old_path.exists():
        old = pd.read_csv(old_path)
        old_summary = old.groupby("method", sort=False)[
            ["precision", "row_recall", "f1", "event_coverage", "average_precision", "false_alarm_rate", "roc_auc"]
        ].mean().reset_index()
        combined = pd.concat([old_summary, combined], ignore_index=True)
    combined.to_csv(TABLES / f"stage503_main_results_combined{tag}.csv", index=False)

    resources = None
    if not args.skip_resources:
        resource_rows = []
        for name in args.models:
            print(f"[resource] probing {name}", flush=True)
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--resource-only", "--model", name, "--length", str(args.length)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(proc.stderr[-2000:], file=sys.stderr)
                continue
            lines = [line for line in proc.stdout.splitlines() if line.startswith("RESOURCE_JSON ")]
            if lines:
                resource_rows.append(json.loads(lines[-1][len("RESOURCE_JSON "):]))
        resources = pd.DataFrame(resource_rows)
        resources.to_csv(TABLES / f"stage503_deep_baseline_resources{tag}.csv", index=False)

    # LaTeX rows for the main results table.
    latex_lines = []
    for _, row in summary.iterrows():
        latex_lines.append(
            f"{LABEL_CN.get(row['method'], row['method'])} & "
            f"{row['precision']:.4f} & {row['row_recall']:.4f} & {row['f1']:.4f} & "
            f"{row['event_coverage']:.4f} & {row['average_precision']:.4f} & "
            f"{100.0 * row['false_alarm_rate']:.3f} & {row['roc_auc']:.4f} \\\\"
        )
    (TABLES / f"stage503_main_results_rows{tag}.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    if resources is not None and len(resources):
        res_lines = []
        for _, row in resources.iterrows():
            res_lines.append(
                f"{LABEL_CN.get(row['method'], row['method'])} & {int(row['params'])} & "
                f"{row['latency_mean_ms']:.3f} & {row['latency_p50_ms']:.3f} & {row['latency_p95_ms']:.3f} & "
                f"{row['peak_wset_mb']:.1f} & {row['param_mb']:.3f} \\\\"
            )
        (TABLES / f"stage503_resource_rows{tag}.tex").write_text("\n".join(res_lines) + "\n", encoding="utf-8")

    report = [
        "# Stage503 深度异常检测基线（V6 同特征空间）",
        "",
        f"- 状态：`EXPLORATORY_POST_V6`（V6 测试集已检视，本阶段不替代 V6 主结果）",
        f"- 协议：`{frozen['protocol_id']}`，目标 FAR=`{FAR}`（阈值只在 valid normal 行上校准，原样用于 test），seeds=`{args.seeds}`",
        f"- 特征：16 维（3 组件 + 13 形态），窗口长度 `{args.length}`",
        f"- 训练：每机构在干净 train 行上训练一次，epochs=`{args.epochs}`，windows=`{args.windows}`；test FAR 为实测值",
        f"- 耗时：`{time.perf_counter() - started:.1f}s`",
        "",
        "## 汇总（provider × seed 宏平均，test）",
        "",
        "| 方法 | 精确率 | 行召回 | F1 | 事件覆盖 | AP | FAR | ROC-AUC | 运行数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        report.append(
            f"| {LABEL_CN.get(row['method'], row['method'])} | {row['precision']:.4f} | {row['row_recall']:.4f} | "
            f"{row['f1']:.4f} | {row['event_coverage']:.4f} | {row['average_precision']:.4f} | "
            f"{100.0 * row['false_alarm_rate']:.3f}% | {row['roc_auc']:.4f} | {int(row['runs'])} |"
        )
    if resources is not None and len(resources):
        report += [
            "",
            "## 单步推理资源（CPU 单线程，batch=1）",
            "",
            "| 方法 | 参数量 | 均值 ms | P50 ms | P95 ms | 进程峰值 MB | 参数 MB |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in resources.iterrows():
            report.append(
                f"| {LABEL_CN.get(row['method'], row['method'])} | {int(row['params'])} | {row['latency_mean_ms']:.3f} | "
                f"{row['latency_p50_ms']:.3f} | {row['latency_p95_ms']:.3f} | {row['peak_wset_mb']:.1f} | {row['param_mb']:.3f} |"
            )
        report += [
            "",
            "> 进程峰值包含 Python + PyTorch 运行时；参数 MB 是模型权重本身。两者必须在论文中分开表述。",
        ]
    report += [
        "",
        "## 边界",
        "",
        "- 仅使用既有四机构数据与 V6 冻结攻击场景；不新增数据、不新增机构。",
        "- 阈值只用 valid normal 行标定；test 标签只用于评价。",
        "- 深度模型在同一条目内共享同一 16 维特征矩阵与同一条目切分。",
        "- V6 测试集已检视，本阶段结果为探索性补充，不得改写为确认性结果。",
    ]
    (REPORTS / f"stage503_deep_anomaly_baselines{tag}.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[done] {time.perf_counter() - started:.1f}s")
    print(summary.to_string(index=False))
    if resources is not None:
        print(resources.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
