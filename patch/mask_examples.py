import utils
from config import BaseConfiguration
import torch
from torchvision import transforms
from nn_modules import LandmarkExtractor, FaceXZooProjector
from PIL import Image

cfg = BaseConfiguration()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mask_t = utils.load_mask(cfg, '../data/masks/final_patch.png', device)
face_landmark_detector = utils.get_landmark_detector(cfg, device)
location_extractor = LandmarkExtractor(device, face_landmark_detector, cfg.img_size).to(device)
fxz_projector = FaceXZooProjector(device, cfg.img_size, cfg.patch_size).to(device)
img_t = transforms.ToTensor()(Image.open('../datasets/real_person_cropped/Alon Zolfi/20210701_100608.jpg')).unsqueeze(0).to(device)
applied = utils.apply_mask(location_extractor, fxz_projector, img_t, mask_t[:, :3], mask_t[:, 3])
transforms.ToPILImage()(applied[0].cpu()).show()
