import torch
import torch.nn as nn

from dplearn.cnn_model import CNNModel


PATH = './models/model.pth'

device = torch.device("cuda")
#model.to(device)
#torch.save(model.state_dict(), PATH)
model = CNNModel()
model.load_state_dict(torch.load(PATH))
model.eval()

print(print.state_dict())