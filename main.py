import torch
import torch.nn.functional as F
import gymnasium as gym
import torchvision.transforms.functional as TF
from dataset import TransitionDataLoader, generate_training_data
from model import Thinker
from config import NUM_EPOCHS, EMBED_SIZE, BATCH_SIZE, LAMBDA, LEARNING_RATE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_episode_gif, find_latest_checkpoint, get_map

def _get_frame(env, device):
    image = env.render()
    frame = TF.rgb_to_grayscale(
        torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
        num_output_channels=1,
    )
    frame = TF.resize(frame, (64, 64), antialias=True)
    return frame.unsqueeze(0).to(device)

@torch.no_grad()
def plan_action(thinker, emb, goal_emb, depth=6, action_dim=4, device=None):
    device = device or emb.device

    seqs = torch.cartesian_prod(*[torch.arange(action_dim, device=device) for _ in range(depth)])
    if depth == 1:
        seqs = seqs.unsqueeze(-1)
    num_seqs = seqs.shape[0]

    cur_emb = emb.expand(num_seqs, -1)
    
    for t in range(depth):
        actions = F.one_hot(seqs[:, t], num_classes=action_dim).to(cur_emb.dtype)
        diff = thinker.predictor(cur_emb, actions)
        cur_emb = cur_emb + diff

    final_dists = F.mse_loss(cur_emb, goal_emb.expand(num_seqs, -1), reduction="none").mean(dim=-1)
    best = torch.argmin(final_dists)
    return seqs[best].tolist()


@torch.no_grad()
def run(thinker, depth=8, save_gif_path=None, device=None):
    thinker.eval()
    device = device or next(thinker.parameters()).device

    desc = get_map()
    env = gym.make(
        "FrozenLake-v1",
        render_mode="rgb_array",
        desc=desc,
        is_slippery=False,
    )
    env.reset()

    goal_state = env.unwrapped.nrow * env.unwrapped.ncol - 1
    current_state = env.unwrapped.s

    env.unwrapped.s = goal_state
    goal_frame = _get_frame(env, device)
    env.unwrapped.s = current_state

    goal_emb = thinker.encoder(goal_frame)

    action_dim = env.action_space.n
    frames = []
    success = False

    frame = _get_frame(env, device)
    emb = thinker.encoder(frame)
    action_sequence = plan_action(thinker, emb, goal_emb, depth=depth, action_dim=action_dim, device=device)

    for step_idx in range(len(action_sequence)):
        frame = _get_frame(env, device)
        frames.append((frame.squeeze(0) * 255.0).byte().permute(1, 2, 0).squeeze(-1).cpu().numpy())

        _, _, terminated, truncated, _ = env.step(action_sequence[step_idx])

        if terminated or truncated:
            success = bool(env.unwrapped.s == goal_state)
            frame = _get_frame(env, device)
            frames.append((frame.squeeze(0) * 255.0).byte().permute(1, 2, 0).squeeze(-1).cpu().numpy())
            break

    env.close()

    if save_gif_path is not None and len(frames) > 0:
        save_episode_gif(frames, save_gif_path)

    return success, frames

if __name__ == "__main__":
    data_accumulator = generate_training_data()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    thinker = Thinker(d=EMBED_SIZE).to(device)

    checkpoint_path, start_epoch = find_latest_checkpoint("./checkpoints")
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path} (epoch {start_epoch})")
        thinker.load_state_dict(torch.load(checkpoint_path, map_location=device))
        start_epoch += 1
    else:
        start_epoch = 0

    optimizer = torch.optim.Adam(
        list(thinker.encoder.parameters()) + list(thinker.predictor.parameters()),
        lr=LEARNING_RATE,
    )

    writer = SummaryWriter()
    global_step = 0

    for i in range(start_epoch, NUM_EPOCHS):
        print("Epoch", i)
        dataloader = TransitionDataLoader(
            data_accumulator,
            batch_size=BATCH_SIZE,
        )

        for batch in tqdm(dataloader):
            frame = batch["frame"].to(device)
            action = batch["action"].to(device)
            next_frame = batch["next_frame"].to(device)

            mse_loss, visreg_loss = thinker.pred_loss(frame, action, next_frame)
            loss = LAMBDA * visreg_loss + (1 - LAMBDA) * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/visreg', visreg_loss.item(), global_step)
            writer.add_scalar('loss/pred', mse_loss.item(), global_step)
            writer.add_scalar('loss/total', loss.item(), global_step)
            global_step += 1
        
        run(thinker, save_gif_path=f"./gifs/epoch_{i}.gif", device=device)
        torch.save(thinker.state_dict(), f"./checkpoints/epoch_{i}.pth")

    writer.close()