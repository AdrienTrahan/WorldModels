import glob
import os
import re
from PIL import Image
from config import RANDOM_MAP, FROZEN_LAKE_MAP
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random
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

def get_map():
    if RANDOM_MAP:
        return generate_random_map(size=4, p=0.8)
        # return _randomize_start_goal(base_desc)
    else:
        return FROZEN_LAKE_MAP
    

def _randomize_start_goal(desc, max_attempts=200):
    grid = [list(row) for row in desc]
    nrow = len(grid)
    ncol = len(grid[0])

    for r in range(nrow):
        for c in range(ncol):
            if grid[r][c] in ("S", "G"):
                grid[r][c] = "F"

    open_cells = [(r, c) for r in range(nrow) for c in range(ncol) if grid[r][c] != "H"]

    start, goal = random.sample(open_cells, 2)
    for _ in range(max_attempts):
        start, goal = random.sample(open_cells, 2)
        if _reachable(grid, start, goal):
            break

    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"
    new_desc = ["".join(row) for row in grid]
    start_state = start[0] * ncol + start[1]
    goal_state = goal[0] * ncol + goal[1]
    return new_desc, start_state, goal_state


def _reachable(grid, start, goal):
    nrow = len(grid)
    ncol = len(grid[0])
    visited = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrow and 0 <= nc < ncol and (nr, nc) not in visited and grid[nr][nc] != "H":
                visited.add((nr, nc))
                stack.append((nr, nc))
    return False