import utils
from config import BaseConfiguration
import torch
from torchvision import transforms
from nn_modules import LandmarkExtractor, FaceXZooProjector
from PIL import Image

cfg = BaseConfiguration()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

face_landmark_detector = utils.get_landmark_detector(cfg, device)
location_extractor = LandmarkExtractor(device, face_landmark_detector, cfg.img_size).to(device)
fxz_projector = FaceXZooProjector(device, cfg.img_size, cfg.patch_size).to(device)
img_t = transforms.ToTensor()(Image.open('../datasets/CASIA-WebFace_aligned/0000137/070.jpg')).unsqueeze(0).to(device)
mask_t = utils.load_mask(cfg, '../data/masks/face3.png', device)
applied = utils.apply_mask(location_extractor, fxz_projector, img_t, mask_t[:, :3], mask_t[:, 3], is_3d=True)
transforms.ToPILImage()(applied[0].cpu()).save('exmaple.png')
