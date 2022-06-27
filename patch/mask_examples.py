import utils
from config import BaseConfiguration
import torch
from torchvision import transforms
from nn_modules import LandmarkExtractor, FaceXZooProjector
from PIL import Image
import os
from pathlib import Path

cfg = BaseConfiguration()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

face_landmark_detector = utils.get_landmark_detector(cfg, device)
location_extractor = LandmarkExtractor(device, face_landmark_detector, cfg.img_size).to(device)
fxz_projector = FaceXZooProjector(device, cfg.img_size, cfg.patch_size).to(device)
img_t = transforms.ToTensor()(Image.open('../datasets/real_aligned/Daniel/IMG_20211116_150209.jpg')).unsqueeze(0).to(device)
person_id = 'Daniel1'
Path(os.path.join('..', 'outputs', person_id)).mkdir(parents=True, exist_ok=True)
transforms.ToPILImage()(img_t[0].cpu()).save(os.path.join('..', 'outputs', person_id, 'clean' + person_id + '.png'))
for mask_path, is_3d, save_name in [('final_patch.png', False, 'adv'),
                                    ('random.png', False, 'random'),
                                    ('blue.png', True, 'blue'),
                                    ('face1.png', True, 'male'),
                                    ('face3.png', True, 'female')]:
    mask_t = utils.load_mask(cfg, os.path.join('../data/masks', mask_path), device)
    uv_mask = mask_t[:, 3] if mask_t.shape[1] == 4 else None
    applied = utils.apply_mask(location_extractor, fxz_projector, img_t, mask_t[:, :3], uv_mask, is_3d=is_3d)
    transforms.ToPILImage()(applied[0].cpu()).save(os.path.join('..', 'outputs', person_id, save_name + person_id + '.png'))
