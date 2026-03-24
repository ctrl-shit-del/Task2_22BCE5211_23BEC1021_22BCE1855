"""
prepare_data.py  –  Dataset Preparation for SGMSE+ Speech De-Reverberation
===========================================================================
Samsung Spatial Hackathon – Task 2
Team: Harsh Singhal (23BEC1021), Sailee Allyadwar (22BCE5211),
      Navomi S. Ramesh (22BCE1855)

This script
  1. Scans clean LibriSpeech utterances (.flac / .wav)
  2. Resamples each utterance to 16 kHz
  3. Convolves it with a randomly selected ARNI Room Impulse Response (RIR)
     to synthesise reverberant speech
  4. Writes clean + reverberant pairs to disk in train / val / test splits
  5. Produces per-split CSV manifests listing audio paths and durations

Expected input layout
---------------------
  <librispeech_root>/
    train-clean-100/   (or any LibriSpeech subset)
    dev-clean/
    test-clean/

  <arni_root>/
    *.wav   (single flat folder or nested sub-folders – both work)

Output layout
-------------
  <out_dir>/
    train/
      clean/           reverberant/
    val/
      clean/           reverberant/
    test/
      clean/           reverberant/
    manifest_train.csv
    manifest_val.csv
    manifest_test.csv

Usage
-----
    python prepare_data.py \\
        --librispeech ./LibriSpeech \\
        --arni        ./ARNI \\
        --output      ./data \\
        --sr          16000 \\
        --val-ratio   0.1 \\
        --test-ratio  0.1 \\
        --seed        42
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torchaudio
from scipy.signal import fftconvolve
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reverberant-speech dataset for SGMSE+ (LibriSpeech + ARNI)"
    )
    parser.add_argument("--librispeech", type=str, required=True,
                        help="Root of the LibriSpeech dataset")
    parser.add_argument("--arni", type=str, required=True,
                        help="Root of the ARNI RIR dataset")
    parser.add_argument("--output", type=str, default="./data",
                        help="Output root directory (default: ./data)")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Target sampling rate in Hz (default: 16000)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction reserved for validation (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Fraction reserved for testing (default: 0.1)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total utterances (useful for quick debugging)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_resample(path: Path, target_sr: int) -> np.ndarray:
    """
    Load an audio file (wav / flac), down-mix to mono, and resample to
    target_sr.  Returns a 1-D float32 numpy array.
    """
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy().astype(np.float32)


def convolve_rir(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """
    Convolve `clean` speech with a Room Impulse Response `rir` to produce
    reverberant speech.

    The output is trimmed to the original length of `clean` (causal trim).
    Amplitude is normalised so the peak of the reverberant signal matches
    the peak of the clean signal, preventing clipping.
    """
    reverberant = fftconvolve(clean, rir)[:len(clean)]
    peak = np.max(np.abs(reverberant))
    if peak > 1e-8:
        reverberant = reverberant * (np.max(np.abs(clean)) / peak)
    return reverberant.astype(np.float32)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_audio_files(root: str, extensions: Tuple[str, ...] = (".wav", ".flac")) -> List[Path]:
    """Recursively collect all audio files with the given extensions."""
    root_path = Path(root)
    if not root_path.is_dir():
        print(f"[ERROR] Directory not found: {root_path}", file=sys.stderr)
        sys.exit(1)
    files = sorted(p for p in root_path.rglob("*") if p.suffix.lower() in extensions)
    return files


def make_unique_stem(path: Path) -> str:
    """
    Create a unique output filename from a LibriSpeech path.

    LibriSpeech can have identical filenames (e.g., 0001.flac) across
    different speaker / chapter sub-directories.  Using the two parent
    directories avoids collisions: <speaker>_<chapter>_<filename>.
    """
    parts = path.parts
    if len(parts) >= 3:
        return f"{parts[-3]}_{parts[-2]}_{path.stem}"
    return path.stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.output)

    # ── Discover files ───────────────────────────────────────────────────────
    print("Scanning LibriSpeech …")
    speech_files = find_audio_files(args.librispeech)

    print("Scanning ARNI RIR dataset …")
    rir_files = find_audio_files(args.arni, extensions=(".wav",))

    if not speech_files:
        print(f"[ERROR] No audio files found under: {args.librispeech}", file=sys.stderr)
        sys.exit(1)
    if not rir_files:
        print(f"[ERROR] No .wav RIR files found under: {args.arni}", file=sys.stderr)
        sys.exit(1)

    print(f"  Speech files : {len(speech_files)}")
    print(f"  RIR files    : {len(rir_files)}")

    if args.max_samples:
        speech_files = speech_files[: args.max_samples]

    # ── Train / val / test split ─────────────────────────────────────────────
    random.shuffle(speech_files)
    n       = len(speech_files)
    n_test  = max(1, int(n * args.test_ratio))
    n_val   = max(1, int(n * args.val_ratio))
    n_train = n - n_val - n_test

    splits = {
        "train": speech_files[:n_train],
        "val":   speech_files[n_train: n_train + n_val],
        "test":  speech_files[n_train + n_val:],
    }
    print(f"\n  train={n_train}  val={n_val}  test={n_test}\n")

    # ── Process each split ───────────────────────────────────────────────────
    for split_name, files in splits.items():
        clean_dir = out_root / split_name / "clean"
        rev_dir   = out_root / split_name / "reverberant"
        clean_dir.mkdir(parents=True, exist_ok=True)
        rev_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = []

        for speech_path in tqdm(files, desc=f"[{split_name:5s}]", unit="file"):
            rir_path = random.choice(rir_files)

            try:
                clean = load_resample(speech_path, args.sr)
                rir   = load_resample(rir_path,    args.sr)
            except Exception as exc:
                tqdm.write(f"  [SKIP] {speech_path.name}: {exc}")
                continue

            reverberant = convolve_rir(clean, rir)

            out_stem  = make_unique_stem(speech_path) + ".wav"
            clean_out = clean_dir / out_stem
            rev_out   = rev_dir   / out_stem

            sf.write(str(clean_out), clean,      args.sr, subtype="PCM_16")
            sf.write(str(rev_out),   reverberant, args.sr, subtype="PCM_16")

            manifest_rows.append({
                "clean":       str(clean_out),
                "reverberant": str(rev_out),
                "rir":         str(rir_path),
                "duration_s":  f"{len(clean) / args.sr:.3f}",
            })

        # Write manifest CSV
        manifest_path = out_root / f"manifest_{split_name}.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["clean", "reverberant", "rir", "duration_s"]
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        tqdm.write(f"  Manifest → {manifest_path}  ({len(manifest_rows)} entries)")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
