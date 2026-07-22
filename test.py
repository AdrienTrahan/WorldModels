import torch
import gymnasium as gym
from model import Thinker
from config import EMBED_SIZE
from utils import find_latest_checkpoint
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from main import _get_frame 

device = torch.device("cpu")
thinker = Thinker(d=EMBED_SIZE).to(device)
checkpoint_path, epoch = find_latest_checkpoint("./checkpoints")
thinker.load_state_dict(torch.load(checkpoint_path, map_location=device))
thinker.eval()


env = gym.make("FrozenLake-v1", render_mode="rgb_array", desc=generate_random_map(size=4, p=0.8), is_slippery=False)
env.reset()

with torch.no_grad():
    embs = []
    for test_state in [0, 1, 4, 5, 9, 15]:  # 15 = goal, should look very different
        env.unwrapped.s = test_state
        frame = _get_frame(env, device)
        emb = thinker.encoder(frame)
        embs.append(emb.squeeze(0))
        print(f"state={test_state} emb[:5]={emb.squeeze(0)[:5]}")

    embs = torch.stack(embs)
    print("\npairwise embedding distances across DIFFERENT STATES:")
    print(torch.cdist(embs, embs))