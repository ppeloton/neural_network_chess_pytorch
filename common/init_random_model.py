import common
import torch

model = common.SupNetwork()

torch.save(model.state_dict(), "random_model.pth")

