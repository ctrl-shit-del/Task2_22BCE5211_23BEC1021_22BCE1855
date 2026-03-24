"""
train.py  –  SGMSE+ Speech De-Reverberation – Training Script
==============================================================
Samsung Spatial Hackathon – Task 2
Team: Harsh Singhal (23BEC1021), Sailee Allyadwar (22BCE5211),
      Navomi S. Ramesh (22BCE1855)

Trains or fine-tunes a SGMSE+ (Score-Based Generative Model for Speech
Enhancement) model for the task of speech de-reverberation.

Architecture
------------
  Backbone : NCSN++ (Noise Conditional Score Network ++, Song et al.)
  SDE      : OUVE  (Ornstein-Uhlenbeck Variance Exploding, Richter et al.)
  Sampler  : Predictor-Corrector (Euler-Maruyama + Langevin) at inference

Pre-requisite
-------------
Clone and install the SGMSE codebase before running:
    git clone https://github.com/sp-uhh/sgmse
    cd sgmse && pip install -e .

Usage – train from scratch
---------------------------
    python train.py \\
        --base-dir  ./data \\
        --epochs    4 \\
        --batch-size 4 \\
        --lr        1e-4

Usage – resume from checkpoint
-------------------------------
    python train.py \\
        --base-dir      ./data \\
        --resume-from   epoch=4-step=10000.ckpt \\
        --epochs        8

Outputs
-------
  ./lightning_logs/   –  TensorBoard event files
  ./checkpoints/      –  Saved .ckpt files (top-k by valid_loss + last)
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SGMSE+ training script for speech de-reverberation"
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Root data folder with train/ and val/ sub-dirs")
    parser.add_argument("--format", type=str, default="default",
                        help="Dataset format tag passed to SpecsDataModule (default: 'default')")

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument("--backbone", type=str, default="ncsnpp",
                        choices=["ncsnpp", "dcunet", "dccrn"],
                        help="Score-network backbone (default: ncsnpp)")
    parser.add_argument("--sde", type=str, default="ouve",
                        choices=["ouve", "bbed", "vesde"],
                        help="Stochastic Differential Equation type (default: ouve)")

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=4,
                        help="Maximum number of training epochs (default: 4)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per device (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices, e.g. '0' or '0,1'. "
                             "Omit to train on CPU.")

    # ── Checkpointing & logging ───────────────────────────────────────────────
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints (default: ./checkpoints)")
    parser.add_argument("--log-dir", type=str, default="./lightning_logs",
                        help="TensorBoard log directory (default: ./lightning_logs)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a .ckpt file to resume training from")
    parser.add_argument("--save-top-k", type=int, default=3,
                        help="Number of best checkpoints to keep (default: 3)")
    parser.add_argument("--log-every-n-steps", type=int, default=10,
                        help="Log metrics every N training steps (default: 10)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Import check ──────────────────────────────────────────────────────────
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
        from pytorch_lightning.loggers import TensorBoardLogger
        from sgmse.model import ScoreModel            # type: ignore
        from sgmse.data_module import SpecsDataModule  # type: ignore
    except ImportError as exc:
        print(
            f"[ImportError] {exc}\n\n"
            "Install the SGMSE package first:\n"
            "    git clone https://github.com/sp-uhh/sgmse\n"
            "    cd sgmse && pip install -e .\n",
            file=sys.stderr,
        )
        sys.exit(1)

    pl.seed_everything(42, workers=True)

    # ── DataModule ────────────────────────────────────────────────────────────
    data_module = SpecsDataModule(
        base_dir=args.base_dir,
        format=args.format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            print(f"[ERROR] Checkpoint not found: {args.resume_from}", file=sys.stderr)
            sys.exit(1)
        print(f"Resuming from checkpoint: {args.resume_from}")
        model = ScoreModel.load_from_checkpoint(
            args.resume_from, map_location="cpu"
        )
    else:
        # Instantiate with NCSN++ backbone (default SGMSE+ configuration).
        # backbone_kwargs are passed through to the NCSNpp module.
        model = ScoreModel(
            backbone=args.backbone,
            sde=args.sde,
            lr=args.lr,
            backbone_kwargs=dict(
                # Channel multipliers from the default SGMSE NCSN++ config
                # (see sgmse/backbones/ncsnpp.py – NCSNpp defaults)
                ch_mult=(1, 1, 2, 2, 2, 2, 2),
            ),
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="epoch={epoch}-step={step}",
        save_top_k=args.save_top_k,
        monitor="valid_loss",
        mode="min",
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="sgmse_dereverberation",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    if args.gpus is not None:
        gpu_ids   = [int(g.strip()) for g in args.gpus.split(",")]
        accelerator = "gpu"
        devices     = gpu_ids
    else:
        accelerator = "cpu"
        devices     = 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True,
        gradient_clip_val=1.0,   # stabilises NCSN++ score-matching loss
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(
        f"\n{sep}\n"
        f"SGMSE+ Training – Speech De-Reverberation\n"
        f"{sep}\n"
        f"  Backbone     : {args.backbone.upper()}\n"
        f"  SDE          : {args.sde.upper()}\n"
        f"  Epochs       : {args.epochs}\n"
        f"  Batch size   : {args.batch_size}\n"
        f"  Learning rate: {args.lr}\n"
        f"  Accelerator  : {accelerator}  {devices}\n"
        f"  Data root    : {args.base_dir}\n"
        f"  Ckpt dir     : {args.ckpt_dir}\n"
        f"  Log dir      : {args.log_dir}\n"
        f"{sep}\n"
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_from,   # None → train from scratch
    )

    best = checkpoint_callback.best_model_path
    print(f"\nTraining complete.")
    print(f"Best checkpoint : {best}")
    print(f"Last checkpoint : {checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    main()
