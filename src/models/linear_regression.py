import torch
from models.base_model import BaseModel

class LinearRegression(torch.nn.Module, BaseModel):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.linear = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        return self.linear(x)

    @staticmethod
    def name():
        return "linear_regression"
