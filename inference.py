"""
inference.py  –  SGMSE+ Speech De-Reverberation – Inference Script
===================================================================
Samsung Spatial Hackathon – Task 2
Team: Harsh Singhal (23BEC1021), Sailee Allyadwar (22BCE5211),
      Navomi S. Ramesh (22BCE1855)

Runs a trained SGMSE+ checkpoint on reverberant .wav files and
produces de-reverberated output using the Predictor-Corrector sampler.

Pre-requisite
-------------
Clone and install the SGMSE codebase before running:
    git clone https://github.com/sp-uhh/sgmse
    cd sgmse && pip install -e .

Usage – single file
-------------------
    python inference.py \\
        --ckpt sgmse_task2_light.ckpt \\
        --input reverberant.wav \\
        --output enhanced.wav

Usage – batch (folder)
-----------------------
    python inference.py \\
        --ckpt sgmse_task2_light.ckpt \\
        --input ./reverberant_wavs/ \\
        --output ./enhanced_wavs/ \\
        --N 30 --corrector-steps 1 --snr 0.5

Options
-------
  --ckpt              Path to the model checkpoint (.ckpt)
  --input             Input .wav file, or folder of .wav files
  --output            Output .wav file, or output folder
  --sr                Target sampling rate in Hz          (default: 16000)
  --N                 Number of reverse-SDE steps         (default: 30)
  --corrector-steps   Corrector iterations per predictor  (default: 1)
  --snr               Langevin corrector SNR parameter    (default: 0.5)
  --device            'cuda' or 'cpu' (auto-detected if omitted)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SGMSE+ Speech De-Reverberation – Inference"
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the model checkpoint (.ckpt)")
    parser.add_argument("--input", type=str, required=True,
                        help="Reverberant .wav file or folder of .wav files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .wav file or output folder")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Target sampling rate in Hz (default: 16000)")
    parser.add_argument("--N", type=int, default=30,
                        help="Number of reverse-SDE steps (default: 30)")
    parser.add_argument("--corrector-steps", type=int, default=1,
                        help="Corrector iterations per predictor step (default: 1)")
    parser.add_argument("--snr", type=float, default=0.5,
                        help="SNR for the Langevin corrector (default: 0.5)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="Compute device (auto-detected if omitted)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int) -> torch.Tensor:
    """
    Load a .wav / .flac file, down-mix to mono, and resample to target_sr.

    Returns
    -------
    torch.Tensor  shape (1, T), float32
    """
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:          # stereo → mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sr
        )
        waveform = resampler(waveform)
    return waveform  # (1, T)


def save_audio(path: str, waveform: torch.Tensor, sr: int) -> None:
    """Save a (1, T) waveform tensor as a 16-bit PCM .wav file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(path, waveform.cpu(), sr)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device):
    """
    Load the SGMSE+ ScoreModel from a PyTorch Lightning checkpoint.

    The ScoreModel wraps the NCSN++ backbone and the OUVE SDE.
    Requires the `sgmse` package to be installed (see module docstring).

    Returns the model in eval mode, moved to `device`.
    """
    if not Path(ckpt_path).exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from sgmse.model import ScoreModel  # type: ignore
    except ModuleNotFoundError:
        print(
            "[ERROR] The `sgmse` package is not installed.\n"
            "        Clone and install it first:\n"
            "            git clone https://github.com/sp-uhh/sgmse\n"
            "            cd sgmse && pip install -e .",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    model = ScoreModel.load_from_checkpoint(
        ckpt_path, base_dir="", map_location=device
    )
    model.eval()
    model.to(device)
    print(f"  Model loaded on {device}.\n")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def enhance_waveform(
    model,
    y: torch.Tensor,
    N: int,
    corrector_steps: int,
    snr: float,
    device: torch.device,
) -> torch.Tensor:
    """
    De-reverberate a single waveform using the reverse-time SDE
    Predictor-Corrector sampler.

    Parameters
    ----------
    model           : loaded ScoreModel
    y               : reverberant waveform, shape (1, T)
    N               : number of reverse-SDE steps
    corrector_steps : Langevin corrector iterations per predictor step
    snr             : corrector SNR hyper-parameter
    device          : torch.device

    Returns
    -------
    torch.Tensor  shape (1, T), float32, on CPU
    """
    y = y.to(device)
    with torch.no_grad():
        # model.enhance() handles STFT → score model → iSTFT internally
        x_hat = model.enhance(
            y,
            N=N,
            corrector_steps=corrector_steps,
            snr=snr,
        )
    if x_hat.dim() == 1:
        x_hat = x_hat.unsqueeze(0)
    return x_hat.cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = load_model(args.ckpt, device)

    # ── File collection ─────────────────────────────────────────────────────
    input_path  = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        file_pairs = [(input_path, output_path)]
    elif input_path.is_dir():
        wav_files = sorted(input_path.glob("**/*.wav"))
        if not wav_files:
            print(f"[ERROR] No .wav files found in: {input_path}", file=sys.stderr)
            sys.exit(1)
        output_path.mkdir(parents=True, exist_ok=True)
        file_pairs = [
            (f, output_path / f.relative_to(input_path))
            for f in wav_files
        ]
    else:
        print(f"[ERROR] Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Files   : {len(file_pairs)}")
    print(f"Steps N : {args.N}   Corrector : {args.corrector_steps}   SNR : {args.snr}\n")

    # ── Inference loop ──────────────────────────────────────────────────────
    t_start = time.time()
    for src, dst in tqdm(file_pairs, desc="Enhancing", unit="file"):
        y = load_audio(str(src), args.sr)
        x_hat = enhance_waveform(
            model, y,
            N=args.N,
            corrector_steps=args.corrector_steps,
            snr=args.snr,
            device=device,
        )
        save_audio(str(dst), x_hat, args.sr)

    elapsed = time.time() - t_start
    print(f"\nDone. {len(file_pairs)} file(s) processed in {elapsed:.1f} s.")
    print(f"Output : {output_path}")


if __name__ == "__main__":
    main()
