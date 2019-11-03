import torch
from models.base_model import BaseModel
import torch.nn.functional as F

class TripleDense(torch.nn.Module, BaseModel):
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out

    @staticmethod
    def name():
        return "triple_dense"
