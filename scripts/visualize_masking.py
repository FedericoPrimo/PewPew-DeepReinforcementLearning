"""
Salva una figura che mostra l'effetto del masking sulla pipeline visiva.

Esempi:
  python scripts/visualize_masking.py
  python scripts/visualize_masking.py --output results/masking_example.png --seed 42
  python scripts/visualize_masking.py --output results/masking_example.png --mask-seed 123
  python scripts/visualize_masking.py --mask-size 28 28 --num-masks 3 --apply-probability 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.masking import RandomScreenMasker
from src.utils.config import load_config
from src.utils.seeding import set_global_seed

gym.register_envs(ale_py)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualizza il masking su un frame Atari.")
    parser.add_argument("--common", default="configs/config_common.yaml")
    parser.add_argument("--output", default="results/masking_example.png")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=None,
        help="Seed opzionale solo per la posizione delle maschere. Se omesso, cambia a ogni run.",
    )
    parser.add_argument("--steps", type=int, default=40, help="Step casuali prima di catturare il frame.")
    parser.add_argument("--mode", choices=["rgb", "grayscale"], default="rgb")
    parser.add_argument("--mask-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--num-masks", type=int, default=None)
    parser.add_argument("--apply-probability", type=float, default=1.0)
    parser.add_argument("--mask-value", choices=["zero", "mean"], default=None)
    return parser.parse_args()


def chw_to_hwc(frame: np.ndarray, frame_stack: int) -> np.ndarray:
    """Converte lo stack (3*frame_stack,H,W) in immagine HWC usando l'ultimo frame."""
    last = frame[-3:, :, :]
    image = np.transpose(last, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def step_randomly(env: gym.Env, steps: int) -> np.ndarray:
    obs, _ = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return obs


def main() -> int:
    args = parse_args()
    cfg = load_config(args.common)
    seed = args.seed if args.seed is not None else cfg.seed
    set_global_seed(seed)

    raw_env = gym.make(cfg.env.id, render_mode="rgb_array")
    raw_frame = step_randomly(raw_env, args.steps)

    preprocessing_kwargs = cfg.preprocessing.to_dict()
    preprocessing_kwargs["mode"] = args.mode
    preprocessor = ImagePreprocessor(**preprocessing_kwargs)
    processed = preprocessor.process(raw_frame)

    masking_kwargs = cfg.masking.to_dict()
    masking_kwargs["enabled"] = True
    masking_kwargs["apply_probability"] = args.apply_probability
    if args.mask_size is not None:
        masking_kwargs["mask_size"] = tuple(args.mask_size)
    if args.num_masks is not None:
        masking_kwargs["num_masks"] = args.num_masks
    if args.mask_value is not None:
        masking_kwargs["mask_value"] = args.mask_value

    if args.mask_seed is None:
        np.random.seed(None)
    else:
        np.random.seed(args.mask_seed)

    masker = RandomScreenMasker(**masking_kwargs)
    masked = masker.apply(processed)

    processed_img = chw_to_hwc(processed, preprocessing_kwargs["frame_stack"])
    masked_img = chw_to_hwc(masked, preprocessing_kwargs["frame_stack"])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(raw_frame)
    axes[0].set_title("Raw Atari frame")
    axes[1].imshow(processed_img)
    axes[1].set_title(f"Preprocessed ({args.mode})")
    axes[2].imshow(masked_img)
    axes[2].set_title("Masked input")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Masking: {masking_kwargs['num_masks']} masks, "
        f"size={tuple(masking_kwargs['mask_size'])}, value={masking_kwargs['mask_value']}",
        fontsize=11,
    )
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    raw_env.close()

    print(f"Figura salvata: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
