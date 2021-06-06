import sys
import os
import random
import warnings

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.nn import CosineSimilarity, functional as F
import torch.optim as optim

from config import patch_config_types
from nn_modules import PatchApplier, LocationExtractor, PatchTransformer, LandmarksApplier, Projector, ap, FaceXZooProjector
from utils import SplitDataset, CustomDataset, CustomDataset1, load_embedder

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if sys.base_prefix.__contains__('home/zolfi'):
    sys.path.append('/home/zolfi/AdversarialMask/patch')
    sys.path.append('/home/zolfi/AdversarialMask/facenet_pytorch')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_random_seed(seed_value=41)


class TrainPatch:
    def __init__(self, mode):
        self.config = patch_config_types[mode]()
        # custom_dataset = CustomDataset(img_dir=self.config.img_dir_train_val,
        #                                lab_dir=self.config.lab_dir_train_val,
        #                                max_lab=self.config.max_labels_per_img,
        #                                img_size=self.config.img_size,
        #                                transform=transforms.Compose(
        #                                    [transforms.Resize(self.config.img_size), transforms.ToTensor()]))

        custom_dataset = CustomDataset1(img_dir=self.config.img_dir,
                                        img_size=self.config.img_size,
                                        transform=transforms.Compose(
                                            [transforms.Resize(self.config.img_size), transforms.ToTensor()]))

        self.train_loader, self.val_loader, self.test_loader = SplitDataset(custom_dataset)(
            val_split=0.1,
            test_split=0.2,
            shuffle=True,
            batch_size=self.config.batch_size)

        self.embedder = load_embedder(self.config.embedder_name, self.config.embedder_weights_path, device)

        # self.patch_applier = PatchApplier(self.config.mask_points)
        self.location_extractor = LocationExtractor(device, self.config.img_size)
        # self.transformer = PatchTransformer(device, self.config.img_size, self.config.patch_size)
        # self.landmarks_applier = LandmarksApplier(self.config.mask_points)

        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size)
        self.cos_sim = CosineSimilarity()

        self.set_to_device()
        self.train_losses = []
        self.val_losses = []

    def set_to_device(self):
        # self.patch_applier = self.patch_applier.to(device)
        self.location_extractor = self.location_extractor.to(device)
        self.fxz_projector = self.fxz_projector.to(device)
        self.cos_sim = self.cos_sim.to(device)

    def train(self):
        # adv_patch_cpu = torch.zeros((1, 3, 100, 100), dtype=torch.float32)
        adv_patch_cpu = self.get_patch()
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        epoch_length = len(self.train_loader)
        prog_bar_desc = 'train-loss: {:.6}, '
        for epoch in range(self.config.epochs):
            train_loss = 0.0
            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=epoch_length)
            for i_batch, img_batch in progress_bar:
                loss = self.forward_step(img_batch, adv_patch_cpu)

                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                adv_patch_cpu.data.clamp_(0, 1)

                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss / (i_batch + 1)))

                if i_batch + 1 == epoch_length:
                    self.save_losses(epoch_length, train_loss)
                    val_loss = self.calc_validation(adv_patch_cpu)
                    prog_bar_desc += ', val-loss: {:.6}'
                    progress_bar.set_postfix_str(prog_bar_desc.format(train_loss, val_loss))

            scheduler.step(self.val_losses[-1])

    def calc_loss(self, clean_emb, patch_emb):
        distance = torch.mean(self.cos_sim(clean_emb, patch_emb))
        return distance

    def get_patch(self):
        import random
        patch = torch.zeros((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
        for i in range(patch.size()[2]):
            patch[0, 0, i, :] = random.randint(0, 255) / 255
            patch[0, 1, i, :] = random.randint(0, 255) / 255
            patch[0, 2, i, :] = random.randint(0, 255) / 255
        patch.requires_grad_(True)
        return patch

    def forward_step(self, img_batch, adv_patch_cpu):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            img_batch = img_batch.to(device)
            adv_patch = adv_patch_cpu.to(device)

            lab_batch, preds = self.location_extractor(img_batch)
            img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch)

            img_batch = F.interpolate(img_batch, size=112)
            img_batch_applied = F.interpolate(img_batch_applied, size=112)

            clean_emb = self.embedder(img_batch)
            patch_emb = self.embedder(img_batch_applied)

            loss = self.calc_loss(clean_emb, patch_emb)
            return loss

    def calc_validation(self, adv_patch_cpu):
        val_loss = 0.0
        with torch.no_grad():
            for i_batch, img_batch in self.val_loader:
                loss = self.forward_step(img_batch, adv_patch_cpu)
                val_loss += loss.item()
        if len(self.val_loader) > 0:
            val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(val_loss)

        return val_loss

    def save_losses(self, epoch_length, train_loss):
        train_loss /= epoch_length
        self.train_losses.append(train_loss)


def main():
    mode = 'private'
    # mode = 'cluster'
    patch_train = TrainPatch(mode)
    patch_train.train()


if __name__ == '__main__':
    main()
