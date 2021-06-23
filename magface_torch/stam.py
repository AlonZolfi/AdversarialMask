from utils import load_embedder
import torch

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = load_embedder('MagFace', 'weights/magface_resnet100.pth', device)
print('x')
