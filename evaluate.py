"""
evaluate.py  –  SGMSE+ Speech De-Reverberation – Evaluation Script
===================================================================
Samsung Spatial Hackathon – Task 2
Team: Harsh Singhal (23BEC1021), Sailee Allyadwar (22BCE5211),
      Navomi S. Ramesh (22BCE1855)

Computes the metrics reported in the paper for matched clean/enhanced pairs:
  • PESQ   (Perceptual Evaluation of Speech Quality, wideband, range −0.5…4.5)
  • ESTOI  (Extended Short-Time Objective Intelligibility,  range 0…1)
  • SI-SDR (Scale-Invariant Signal-to-Distortion Ratio,     dB)
  • SI-SIR / SI-SAR (via fast-bss-eval BSS decomposition,  dB)
  • DNSMOS (Microsoft Deep Noise Suppression MOS,           range 1…5)

Usage
-----
    python evaluate.py \\
        --clean      ./data/test/clean/ \\
        --enhanced   ./enhanced_wavs/ \\
        --output     results.csv

The script pairs files by stem name (filename without extension).
Enhanced files must share the same name as their clean references.

Optional
--------
  --reverberant   Folder of reverberant (input) files. When supplied,
                  baseline metrics (unprocessed) are also printed for
                  comparison alongside the enhanced metrics.
  --sr            Sampling rate expected for all files (default: 16000 Hz).
                  Note: PESQ only supports 8000 or 16000 Hz.
"""

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torchaudio
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate speech de-reverberation quality metrics"
    )
    parser.add_argument("--clean", type=str, required=True,
                        help="Folder of clean reference .wav files")
    parser.add_argument("--enhanced", type=str, required=True,
                        help="Folder of de-reverberated (enhanced) .wav files")
    parser.add_argument("--reverberant", type=str, default=None,
                        help="Optional: folder of reverberant (unprocessed) .wav files")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Expected sampling rate in Hz (default: 16000)")
    parser.add_argument("--output", type=str, default="results.csv",
                        help="Output CSV file path (default: results.csv)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio_numpy(path: str, target_sr: int) -> np.ndarray:
    """
    Load an audio file as a 1-D float32 numpy array.

    The file is down-mixed to mono and resampled to target_sr if needed.
    """
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        import torchaudio.transforms as T
        waveform = T.Resample(sr, target_sr)(waveform)
    return waveform.squeeze().numpy().astype(np.float32)


def align(ref: np.ndarray, est: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trim both arrays to the length of the shorter one."""
    n = min(len(ref), len(est))
    return ref[:n], est[:n]


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """
    Wideband PESQ score (ITU-T P.862.2).
    Requires sr == 16000 Hz for wideband mode.
    """
    if sr not in (8000, 16000):
        print(f"  [PESQ SKIP] Unsupported sample rate {sr} Hz (need 8000 or 16000).")
        return math.nan
    mode = "wb" if sr == 16000 else "nb"
    try:
        from pesq import pesq  # type: ignore
        return float(pesq(sr, ref, deg, mode))
    except Exception as exc:
        print(f"  [PESQ ERROR] {exc}")
        return math.nan


def compute_estoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Extended STOI (Taal et al., range 0…1, higher is better)."""
    try:
        from pystoi import stoi  # type: ignore
        return float(stoi(ref, deg, sr, extended=True))
    except Exception as exc:
        print(f"  [ESTOI ERROR] {exc}")
        return math.nan


def compute_si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (Le Roux et al., 2019).
    Higher is better; ideal is +∞.
    """
    ref = ref - ref.mean()
    est = est - est.mean()
    dot = np.dot(ref, est)
    ref_energy = np.dot(ref, ref) + 1e-8
    scaling = dot / ref_energy
    projection = scaling * ref
    noise = est - projection
    si_sdr_val = 10.0 * np.log10(
        (np.dot(projection, projection) + 1e-8) /
        (np.dot(noise, noise) + 1e-8)
    )
    return float(si_sdr_val)


def compute_si_sir_sar(
    ref: np.ndarray, est: np.ndarray
) -> Tuple[float, float]:
    """
    SI-SIR and SI-SAR using fast-bss-eval BSS-Eval decomposition.
    Falls back to (SI-SDR, SI-SDR) if the package is unavailable.

    Returns (si_sir, si_sar).
    """
    try:
        from fast_bss_eval import si_bss_eval_sources  # type: ignore
        # Expects shape (n_src, n_samples)
        _, sir, sar, _ = si_bss_eval_sources(
            ref[np.newaxis, :].astype(np.float64),
            est[np.newaxis, :].astype(np.float64),
        )
        return float(sir.item()), float(sar.item())
    except ImportError:
        si_sdr_v = compute_si_sdr(ref, est)
        print("  [SI-SIR/SAR] fast-bss-eval not installed – using SI-SDR as proxy.")
        return si_sdr_v, si_sdr_v
    except Exception as exc:
        print(f"  [SI-SIR/SAR ERROR] {exc}")
        return math.nan, math.nan


def compute_dnsmos(audio: np.ndarray, sr: int) -> float:
    """
    Overall DNSMOS score using the Microsoft DNSMOS ONNX model.

    The model expects ~9 s of 16 kHz audio and returns three sub-scores:
      SIG (signal quality), BAK (background noise), OVRL (overall).
    This function returns OVRL.

    Download the ONNX file from:
        https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
    and place it at:
        <script_dir>/dnsmos/sig_bak_ovr.onnx
    """
    DNSMOS_INPUT_LEN_S = 9.01
    onnx_path = Path(__file__).parent / "dnsmos" / "sig_bak_ovr.onnx"

    try:
        import onnxruntime as ort  # type: ignore

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"DNSMOS model not found: {onnx_path}\n"
                "Download from https://github.com/microsoft/DNS-Challenge"
            )

        desired = int(DNSMOS_INPUT_LEN_S * sr)
        if len(audio) < desired:
            audio_in = np.pad(audio, (0, desired - len(audio)))
        else:
            audio_in = audio[:desired]

        sess = ort.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        scores = sess.run(None, {input_name: audio_in[np.newaxis, :].astype(np.float32)})
        return float(scores[0][0][2])   # index 2 = OVRL

    except (FileNotFoundError, ImportError) as exc:
        print(f"  [DNSMOS UNAVAILABLE] {exc}")
        return math.nan
    except Exception as exc:
        print(f"  [DNSMOS ERROR] {exc}")
        return math.nan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIELDS = ["file", "pesq", "estoi", "si_sdr", "si_sir", "si_sar", "dnsmos"]


def compute_all_metrics(
    ref: np.ndarray, est: np.ndarray, sr: int
) -> Dict[str, float]:
    """Return a dict of all metric values for a single (ref, est) pair."""
    si_sir, si_sar = compute_si_sir_sar(ref, est)
    return {
        "pesq":   compute_pesq(ref, est, sr),
        "estoi":  compute_estoi(ref, est, sr),
        "si_sdr": compute_si_sdr(ref, est),
        "si_sir": si_sir,
        "si_sar": si_sar,
        "dnsmos": compute_dnsmos(est, sr),
    }


def _nanmean(values: List[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else math.nan


def summarise(rows: List[Dict]) -> Dict[str, str]:
    """Return a summary row with per-metric mean ± std over all rows."""
    summary: Dict[str, str] = {"file": "MEAN ± STD"}
    for key in FIELDS[1:]:
        vals = [float(r[key]) for r in rows]
        finite = [v for v in vals if not math.isnan(v)]
        if finite:
            summary[key] = f"{np.mean(finite):.4f} ± {np.std(finite):.4f}"
        else:
            summary[key] = "nan"
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    clean_dir    = Path(args.clean)
    enhanced_dir = Path(args.enhanced)
    rev_dir      = Path(args.reverberant) if args.reverberant else None

    # Validate directories
    for d in filter(None, [clean_dir, enhanced_dir, rev_dir]):
        if not d.is_dir():
            print(f"[ERROR] Directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    # Build stem → path mappings
    def index_wavs(folder: Path) -> Dict[str, Path]:
        return {f.stem: f for f in sorted(folder.glob("**/*.wav"))}

    clean_idx    = index_wavs(clean_dir)
    enhanced_idx = index_wavs(enhanced_dir)
    rev_idx      = index_wavs(rev_dir) if rev_dir else {}

    common = sorted(set(clean_idx) & set(enhanced_idx))
    if not common:
        print(
            "[ERROR] No matching filenames found between clean and enhanced folders.\n"
            "        Make sure the .wav files share the same stem names.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Matched pairs : {len(common)}")
    if rev_dir:
        rev_common = set(common) & set(rev_idx)
        print(f"Baseline pairs: {len(rev_common)}  (reverberant baseline)\n")

    # ── Evaluation loop ─────────────────────────────────────────────────────
    rows:     List[Dict] = []
    rev_rows: List[Dict] = []

    for stem in tqdm(common, desc="Evaluating", unit="file"):
        ref = load_audio_numpy(str(clean_idx[stem]),    args.sr)
        est = load_audio_numpy(str(enhanced_idx[stem]), args.sr)
        ref_a, est_a = align(ref, est)

        m = compute_all_metrics(ref_a, est_a, args.sr)
        row = {"file": stem, **{k: f"{v:.4f}" for k, v in m.items()}}
        rows.append(row)

        # Baseline (reverberant vs clean) if provided
        if rev_dir and stem in rev_idx:
            rev = load_audio_numpy(str(rev_idx[stem]), args.sr)
            ref_r, rev_r = align(ref, rev)
            m_rev = compute_all_metrics(ref_r, rev_r, args.sr)
            rev_rows.append({"file": stem, **{k: f"{v:.4f}" for k, v in m_rev.items()}})

    # ── Write CSV ────────────────────────────────────────────────────────────
    summary = summarise(rows)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary)

    print(f"\nResults written → {args.output}")

    # ── Console summary ───────────────────────────────────────────────────────
    sep = "=" * 58
    print(f"\n{sep}\nEnhanced – Summary ({len(rows)} files)\n{sep}")
    for k, v in summary.items():
        if k != "file":
            print(f"  {k.upper():<10}: {v}")

    if rev_rows:
        rev_summary = summarise(rev_rows)
        print(f"\n{sep}\nBaseline (Reverberant, unprocessed) – Summary ({len(rev_rows)} files)\n{sep}")
        for k, v in rev_summary.items():
            if k != "file":
                print(f"  {k.upper():<10}: {v}")


if __name__ == "__main__":
    main()
