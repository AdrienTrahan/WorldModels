import os
import random
import numpy as np
import torch
from config import (
    CACHE_PATH,
    NUM_EPISODES,
)
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class Episode:
    def __init__(self, goal):
        self.data = []
        self.goal = goal

    def add(self, obs, act, rew, done):
        self.data.append((obs, act, rew, done))

    def __len__(self):
        return len(self.data)


class DataAccumulator:
    def __init__(self):
        self.episodes = []

    def add(self, episode):
        self.episodes.append((episode.data, episode.goal))

    def __len__(self):
        return len(self.episodes)


class TransitionDataLoader:
    def __init__(self, accumulator, batch_size, shuffle=True):
        self.accumulator = accumulator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = []
        for ep_idx, (episode, goal) in enumerate(accumulator.episodes):
            for t in range(len(episode) - 1):
                self.index.append((ep_idx, t))

    def __len__(self):
        return (len(self.index) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.index)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield self._build_batch(batch_indices)

    def _build_batch(self, batch_indices):
        frame_batch, act_batch, next_frame_batch = [], [], []

        for idx in batch_indices:
            ep_idx, t = self.index[idx]
            episode, goal = self.accumulator.episodes[ep_idx]

            obs_t, act_t, _, _ = episode[t]
            obs_t1, _, _, _ = episode[t + 1]

            frame_batch.append(obs_t)
            act_batch.append(act_t)
            next_frame_batch.append(obs_t1)

        return {
            "frame": torch.as_tensor(np.stack(frame_batch), dtype=torch.float32),
            "action": torch.as_tensor(np.stack(act_batch), dtype=torch.float32),
            "next_frame": torch.as_tensor(np.stack(next_frame_batch), dtype=torch.float32),
        }


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


def _worker_single(episode_idx):
    seed = episode_idx * 10000 + 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    import gymnasium as gym
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    import torchvision.transforms.functional as TF
    from config import MAX_EPISODE_LEN

    def to_frame(image):
        frame = TF.rgb_to_grayscale(
            torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            num_output_channels=1,
        )
        return TF.resize(frame, (64, 64), antialias=True)

    base_desc = generate_random_map(size=4, p=0.8)
    desc, start_state, goal_state = _randomize_start_goal(base_desc)

    env = gym.make(
        "FrozenLake-v1",
        render_mode="rgb_array",
        desc=desc,
        is_slippery=False,
    )
    env.reset()
    env.unwrapped.s = start_state

    current_state = env.unwrapped.s
    env.unwrapped.s = goal_state
    goal_frame = to_frame(env.render())
    env.unwrapped.s = current_state

    episode = Episode(goal=goal_frame)

    fell_in_hole = False
    hole_frame = None

    for step_idx in range(MAX_EPISODE_LEN):
        if fell_in_hole:
            action = env.action_space.sample()
            action_onehot = torch.zeros(4, dtype=hole_frame.dtype)
            action_onehot[action] = 1.0
            episode.add(hole_frame, action_onehot, 0.0, True)
            continue

        frame = to_frame(env.render())
        action = env.action_space.sample()
        _, _reward, terminated, truncated, _ = env.step(action)

        if step_idx == MAX_EPISODE_LEN - 1:
            truncated = True

        action_onehot = torch.zeros(4, dtype=frame.dtype)
        action_onehot[action] = 1.0
        episode.add(frame, action_onehot, 0.0, False)

        if terminated or truncated:
            final_frame = to_frame(env.render())
            reached_goal = env.unwrapped.s == goal_state

            if terminated and not reached_goal:
                fell_in_hole = True
                hole_frame = final_frame
                zero_action = torch.zeros(4, dtype=final_frame.dtype)
                episode.add(final_frame, zero_action, 0.0, True)
                continue

            zero_action = torch.zeros(4, dtype=final_frame.dtype)
            episode.add(final_frame, zero_action, 0.0, terminated or truncated)
            if terminated:
                episode.add(final_frame, torch.zeros(4, dtype=final_frame.dtype), 0.0, terminated or truncated)
            break

    env.close()

    acc = DataAccumulator()
    acc.add(episode)
    return acc


def generate_training_data():
    if os.path.exists(CACHE_PATH):
        return torch.load(CACHE_PATH, weights_only=False)

    num_workers = min(cpu_count(), NUM_EPISODES)

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_worker_single, range(NUM_EPISODES)),
            total=NUM_EPISODES,
            desc="Generating episodes"
        ))

    merged = DataAccumulator()
    for acc in results:
        merged.episodes.extend(acc.episodes)

    torch.save(merged, CACHE_PATH)
    return merged