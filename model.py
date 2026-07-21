import torch
import torch.nn as nn
from sigreg import MultiSubspaceSIGReg, VISReg, SIGReg
class Encoder(nn.Module):
    def __init__(self, d=32, in_channels=1, pool_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * pool_size * pool_size, d),
        )

    def forward(self, x, mask=None):
        x = self.net(x)
        x = x.flatten(1)
        x = self.fc(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
    

class Predictor(nn.Module):
    def __init__(
        self,
        d=32,
        action_dim=4,
        hidden_size=256
    ):
        super().__init__()
    
        self.seq = nn.Sequential(
            nn.Linear(d + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d),
        )

    def forward(self, x, act):
        x = torch.cat([x, act], dim=-1)
        x = self.seq(x)
        return x


    
class Thinker(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.encoder = Encoder(d=d)
        self.predictor = Predictor(d=d)
        # self.subjepa = MultiSubspaceSIGReg(embed_dim=d)
        # self.visreg = VISReg()
        self.sigreg = SIGReg()

    def pred(self, frame, action):
        emb = self.encoder(frame)
        diff = self.predictor(emb, action)
        return emb + diff
    
    def pred_loss(self, frame, action, next_frame):
        emb = self.encoder(frame)
        emb_t = self.encoder(next_frame)

        diff = self.predictor(emb, action)

        mse_loss = nn.functional.mse_loss(emb + diff, emb_t)
        # visreg_loss = self.visreg(emb.unsqueeze(0)) + self.visreg(emb_t.unsqueeze(0))
        # visreg_loss = self.subjepa(emb.unsqueeze(1)) + self.subjepa(emb_t.unsqueeze(1))
        visreg_loss = self.sigreg(emb.unsqueeze(0)) + self.sigreg(emb_t.unsqueeze(0))

        return mse_loss, visreg_loss
