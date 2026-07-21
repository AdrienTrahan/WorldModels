import torch
import glob
import os
import re
from torch.distributions import Normal
from PIL import Image


def save_episode_gif(frames, path, duration=350):
    images = [Image.fromarray(frame).convert("P") for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def find_latest_checkpoint(checkpoint_dir="./checkpoints"):
    if not os.path.isdir(checkpoint_dir):
        return None, -1

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
    if not checkpoints:
        return None, -1

    def epoch_num(path):
        match = re.search(r"epoch_(\d+)\.pth", os.path.basename(path))
        return int(match.group(1)) if match else -1

    checkpoints = [(epoch_num(p), p) for p in checkpoints]
    checkpoints = [c for c in checkpoints if c[0] != -1]

    if not checkpoints:
        return None, -1

    latest_epoch, latest_path = max(checkpoints, key=lambda x: x[0])
    return latest_path, latest_epoch