"""
Train Torus English Diffusion on TinyStories
==============================================
Jason (theory) + Claude (implementation) — 2026

Usage:
    python train_torus_english.py

    # Resume from checkpoint
    python train_torus_english.py --resume checkpoints/torus_english_step_5000.pt

    # Custom settings
    python train_torus_english.py --batch_size 16 --lr 3e-4
"""

import torch
import torch.nn.functional as F
import os
import sys
import time
import json
import argparse
import gc
from pathlib import Path

from torus_english_diffusion import (
    TorusEnglishDiffusion,
    TorusEnglishDiffusionConfig,
)


# ════════════════════════════════════════════════════════════
# DATASET — TinyStories
# ════════════════════════════════════════════════════════════

class TinyStoriesDataset:
    """
    Streams TinyStories text, encodes with our vocabulary,
    yields fixed-length chunks for training.
    """

    def __init__(self, vocab, seq_len=256, data_dir='./data'):
        self.vocab = vocab
        self.seq_len = seq_len
        self.data_dir = data_dir
        self._texts = []
        self._load()

    def _load(self):
        """Load TinyStories from HuggingFace datasets or local file."""
        local_path = os.path.join(self.data_dir, 'tinystories.txt')

        if os.path.exists(local_path):
            print(f"  Loading TinyStories from {local_path}...")
            with open(local_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Split on story boundaries
            self._texts = [s.strip() for s in text.split('<|endoftext|>') if s.strip()]
            print(f"  Loaded {len(self._texts)} stories from local file")
            return

        # Download from HuggingFace
        print(f"  Downloading TinyStories from HuggingFace...")
        try:
            from datasets import load_dataset
            ds = load_dataset('roneneldan/TinyStories', split='train',
                            trust_remote_code=True)
            self._texts = [row['text'] for row in ds if row.get('text')]
            print(f"  Downloaded {len(self._texts)} stories")

            # Cache locally
            os.makedirs(self.data_dir, exist_ok=True)
            with open(local_path, 'w', encoding='utf-8') as f:
                for text in self._texts:
                    f.write(text + '\n<|endoftext|>\n')
            print(f"  Cached to {local_path}")

        except ImportError:
            print("  ERROR: pip install datasets")
            print("  Or place tinystories.txt in ./data/")
            sys.exit(1)

    def get_batch(self, batch_size, device):
        """Get a random batch of encoded, padded sequences."""
        import random

        ids_batch = []
        mask_batch = []

        for _ in range(batch_size):
            # Pick random story
            text = random.choice(self._texts)

            # Encode
            tokens = self.vocab.encode(text, add_special=True)

            # Truncate or pad to seq_len
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len - 1] + [2]  # EOS

            # Pad
            pad_len = self.seq_len - len(tokens)
            mask = [True] * len(tokens) + [False] * pad_len
            tokens = tokens + [0] * pad_len  # PAD

            ids_batch.append(tokens)
            mask_batch.append(mask)

        ids = torch.tensor(ids_batch, dtype=torch.long, device=device)
        mask = torch.tensor(mask_batch, dtype=torch.bool, device=device)
        return ids, mask


# ════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════

def train(args):
    # ── Device ──
    # Auto-detect. No flags needed. Whatever hardware you have, use it.
    if torch.xpu.is_available():
        device = torch.device('xpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"  Device: {device}")

    # ── Model ──
    print(f"\n  Building model...")
    cfg = TorusEnglishDiffusionConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        num_timesteps=args.timesteps,
        consensus_path=args.consensus,
    )

    model = TorusEnglishDiffusion(cfg)
    model = model.to(device)

    params = model.count_parameters()
    print(f"\n  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # ── Optimizer ──
    # Plain Adam, not AdamW. Weight decay caused model collapses before.
    # The consensus geometry already constrains the embedding space.
    # We don't need a second regularizer fighting the first one.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )

    # Flat LR. The noise schedule IS the curriculum.
    # No warmup — the model starts on maximally noisy data (99% corruption)
    # which is already the easiest task (just predict the most common token).
    # Cosine annealing is comp sci dogma, not physics.

    # ── Dataset ──
    print(f"\n  Loading dataset...")
    dataset = TinyStoriesDataset(
        model.vocab, seq_len=args.seq_len, data_dir=args.data_dir)

    # ── Resume ──
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n  Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
        print(f"  Resumed at step {start_step}")

    # ── Checkpoints ──
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Train ──
    print(f"\n{'═' * 60}")
    print(f"  TRAINING")
    print(f"{'═' * 60}")
    print(f"  Steps: {start_step} → {args.max_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq len: {args.seq_len}")
    print(f"  LR: {args.lr}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"{'═' * 60}\n")

    model.train()
    loss_history = []
    acc_history = []
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        # Get batch
        ids, mask = dataset.get_batch(args.batch_size, device)

        # Forward + loss
        losses = model.compute_loss(ids, mask)
        loss = losses['total_loss']

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient health: monitor, don't blindly clip
        # Only clip if genuinely exploding (norm > 10), not as routine suppression
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        optimizer.step()

        # Track
        loss_val = loss.item()
        acc_val = losses['accuracy'].item()
        loss_history.append(loss_val)
        acc_history.append(acc_val)

        # ── Logging ──
        if step % args.log_every == 0 or step == start_step:
            elapsed = time.time() - t0
            steps_done = step - start_step + 1
            steps_per_sec = steps_done / max(elapsed, 1)
            lr_now = args.lr

            # Recent averages
            recent = min(100, len(loss_history))
            avg_loss = sum(loss_history[-recent:]) / recent
            avg_acc = sum(acc_history[-recent:]) / recent

            print(f"  step {step:>6d} | loss {loss_val:.4f} (avg {avg_loss:.4f}) "
                  f"| acc {acc_val:.1%} (avg {avg_acc:.1%}) "
                  f"| lr {lr_now:.2e} | {steps_per_sec:.1f} step/s")

        # ── Generation sample ──
        if step > 0 and step % args.sample_every == 0:
            model.eval()
            with torch.no_grad():
                result = model.generate(
                    seq_len=80, device=device, seed=step)
                text = model.vocab.decode(result['token_ids'][0].tolist())
                collapsed = result['collapse_mask'][0].float().mean().item()
            print(f"  ╔══ sample (seed={step}) ══╗")
            print(f"  ║ {text[:70]}")
            print(f"  ║ collapse: {collapsed:.0%}")
            print(f"  ╚{'═' * 30}╝")
            model.train()

        # ── Checkpoint ──
        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f'torus_english_step_{step}.pt')
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': {
                    'd_model': cfg.d_model,
                    'n_heads': cfg.n_heads,
                    'n_layers': cfg.n_layers,
                    'd_ff': cfg.d_ff,
                    'max_seq_len': cfg.max_seq_len,
                    'num_timesteps': cfg.num_timesteps,
                    'vocab_size': cfg.vocab_size,
                },
                'loss_history': loss_history,
                'acc_history': acc_history,
            }, ckpt_path)
            print(f"  ✓ saved {ckpt_path}")

    # ── Final save ──
    final_path = os.path.join(args.checkpoint_dir, 'torus_english_final.pt')
    torch.save({
        'step': args.max_steps,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_history': loss_history,
        'acc_history': acc_history,
    }, final_path)
    print(f"\n  ✓ Final checkpoint: {final_path}")

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  DONE — {args.max_steps - start_step} steps in {elapsed/60:.1f} min")
    print(f"  Final avg loss: {sum(loss_history[-100:])/100:.4f}")
    print(f"  Final avg acc:  {sum(acc_history[-100:])/100:.1%}")
    print(f"{'═' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Torus English Diffusion on TinyStories')

    # Model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--timesteps', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--consensus', type=str,
                        default='./consensus_coordinates.json')

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_steps', type=int, default=50000)

    # Data
    parser.add_argument('--data_dir', type=str, default='./data')

    # Logging
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=2000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    # Resume
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
