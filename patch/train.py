import sys
import os
import random
import warnings
import time
import datetime
from pathlib import Path
import pickle

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.nn import CosineSimilarity, L1Loss, MSELoss
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader

from PIL import Image

import utils
import losses
from config import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation, NormalizeToArcFace
from test import Evaluator
from utils import SplitDataset, CustomDataset1, load_embedder, EarlyStopping
from landmark_detection.face_alignment.face_alignment import FaceAlignment, LandmarksType
from landmark_detection.pytorch_face_landmark.models import mobilefacenet

import warnings
warnings.simplefilter('ignore', UserWarning)

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is {}'.format(device))

if sys.base_prefix.__contains__('home/zolfi'):
    sys.path.append('/home/zolfi/AdversarialMask/patch')
    sys.path.append('/home/zolfi/AdversarialMask/arcface_torch')
    sys.path.append('/home/zolfi/AdversarialMask/face-alignment')
    sys.path.append('/home/zolfi/AdversarialMask/prnet')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_random_seed(seed_value=42)


class AdversarialMask:
    def __init__(self, mode):
        self.config = patch_config_types[mode]()

        if self.config.is_real_person:
            self.train_no_aug_loader, self.train_loader, self.val_loader, self.test_loader = utils.get_loaders_real_images(self.config)
        else:
            self.train_no_aug_loader, self.train_loader, self.val_loader, self.test_loader = utils.get_loaders(self.config)

        self.embedder = load_embedder(self.config.embedder_name, self.config.embedder_weights_path, device)

        face_landmark_detector = utils.get_landmark_detector(self.config, device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        # self.preds = self.load_landmarks()
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size).to(device)
        self.normalize_arcface = NormalizeToArcFace(device).to(device)
        self.total_variation = TotalVariation(device).to(device)
        self.dist_loss = losses.get_loss(self.config)

        self.train_losses = []
        self.dist_losses = []
        self.tv_losses = []
        self.val_losses = []

        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        if 'SLURM_JOBID' not in os.environ.keys():
            self.current_dir = "experiments/" + month_name + '/' + time.strftime("%d-%m-%Y") + '_' + time.strftime(
                "%H-%M-%S")
        else:
            self.current_dir = "experiments/" + month_name + '/' + time.strftime("%d-%m-%Y") + '_' + os.environ['SLURM_JOBID']
        self.create_folders()
        utils.save_class_to_file(self.config, self.current_dir)
        self.target_embedding = utils.get_person_embedding(self.config,  self.train_no_aug_loader, self, device)
        self.test_target_embedding = utils.get_person_embedding(self.config,  self.train_no_aug_loader, self, device, False)
        self.best_patch = None

    def create_folders(self):
        Path('/'.join(self.current_dir.split('/')[:2])).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/final_results').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/saved_patches').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/saved_similarities').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/losses').mkdir(parents=True, exist_ok=True)

    def train(self):
        adv_patch_cpu = self.get_patch(self.config.initial_patch)
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(current_dir=self.current_dir, patience=self.config.es_patience)
        epoch_length = len(self.train_loader)
        for epoch in range(self.config.epochs):
            train_loss = 0.0
            dist_loss = 0.0
            tv_loss = 0.0
            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=epoch_length)
            prog_bar_desc = 'train-loss: {:.6}, dist-loss: {:.6}, tv-loss: {:.6}, lr: {:.6}'
            for i_batch, (img_batch, img_names) in progress_bar:
                (b_loss, sep_loss), vars = self.forward_step(img_batch, adv_patch_cpu, img_names)

                train_loss += b_loss.item()
                dist_loss += sep_loss[0].item()
                tv_loss += sep_loss[1].item()

                optimizer.zero_grad()
                b_loss.backward()
                optimizer.step()

                adv_patch_cpu.data.clamp_(0, 1)

                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss / (i_batch + 1),
                                                                  dist_loss / (i_batch + 1),
                                                                  tv_loss / (i_batch + 1),
                                                                  optimizer.param_groups[0]["lr"]))

                if i_batch + 1 == epoch_length:
                    # self.calc_validation(adv_patch_cpu)
                    self.save_losses(epoch_length, train_loss, dist_loss, tv_loss)
                    # prog_bar_desc += ', val-loss: {:.6}'
                    progress_bar.set_postfix_str(prog_bar_desc.format(self.train_losses[-1],
                                                                      self.dist_losses[-1],
                                                                      self.tv_losses[-1],
                                                                      optimizer.param_groups[0]["lr"],))
                                                                      # self.val_losses[-1]))
                for var in vars + sep_loss:
                    del var
                del b_loss
                torch.cuda.empty_cache()

            if early_stop(self.train_losses[-1], adv_patch_cpu, epoch):
                self.best_patch = adv_patch_cpu
                break

            scheduler.step(self.train_losses[-1])

        self.best_patch = early_stop.best_patch
        self.save_final_objects()
        self.plot_train_val_loss()
        self.plot_separate_loss()

    def loss_fn(self, patch_emb, tv_loss):
        distance_loss = self.config.dist_weight * torch.mean(self.dist_loss(patch_emb, self.target_embedding))
        # distance_loss = self.config.dist_weight * torch.mean(F.relu(self.dist_loss(patch_emb, self.target_embedding)))
        tv_loss = self.config.tv_weight * tv_loss
        total_loss = distance_loss + tv_loss
        return total_loss, [distance_loss, tv_loss]

    def get_patch(self, p_type):
        if p_type == 'stripes':
            patch = torch.zeros((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
            for i in range(patch.size()[2]):
                patch[0, 0, i, :] = random.randint(0, 255) / 255
                patch[0, 1, i, :] = random.randint(0, 255) / 255
                patch[0, 2, i, :] = random.randint(0, 255) / 255
        elif p_type =='l_stripes':
            patch = torch.zeros((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
            for i in range(int(patch.size()[2]/16)):
                patch[0, 0, int(patch.size()[2]/16)*i:int(patch.size()[2]/16)*(i+1), :] = random.randint(0, 255) / 255
                patch[0, 1, int(patch.size()[2]/16)*i:int(patch.size()[2]/16)*(i+1), :] = random.randint(0, 255) / 255
                patch[0, 2, int(patch.size()[2]/16)*i:int(patch.size()[2]/16)*(i+1), :] = random.randint(0, 255) / 255
        elif p_type == 'random':
            patch = torch.rand((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
        elif p_type == 'white':
            patch = torch.ones((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
        elif p_type == 'body':
            patch = torch.zeros((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
            patch[:, 0] = 1
            patch[:, 1] = 0.8
            patch[:, 2] = 0.58
        elif p_type == 'black':
            patch = torch.zeros((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32) + 0.01
        uv_face = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))
        patch = patch * uv_face
        patch.requires_grad_(True)
        # transforms.ToPILImage()(patch.squeeze(0) * uv_face).save('random.png')
        # transforms.ToPILImage()(patch.squeeze(0) * uv_face).show()
        return patch

    def forward_step(self, img_batch, adv_patch_cpu, img_names, train=True):
        img_batch = img_batch.to(device)
        adv_patch = adv_patch_cpu.to(device)

        preds = self.location_extractor(img_batch)
        # preds = self.get_batch_landmarks(img_names)
        img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch)
        # img_batch_applied = torch.nn.functional.interpolate(img_batch_applied, 112)
        # normalized_batch = self.normalize_arcface(img_batch_applied, preds)
        patch_emb = self.embedder(img_batch_applied)

        tv_loss = self.total_variation(adv_patch, train)
        loss = self.loss_fn(patch_emb, tv_loss)

        return loss, [img_batch, adv_patch, img_batch_applied, patch_emb, tv_loss]

    def calc_validation(self, adv_patch_cpu):
        val_loss = 0.0
        with torch.no_grad():
            for img_batch, img_names in self.val_loader:
                img_batch = img_batch.to(device)
                (loss, _), _ = self.forward_step(img_batch, adv_patch_cpu, img_names, train=False)
                val_loss += loss.item()

                del img_batch, loss
                torch.cuda.empty_cache()
        val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(val_loss)

    def save_losses(self, epoch_length, train_loss, dist_loss, tv_loss):
        train_loss /= epoch_length
        dist_loss /= epoch_length
        tv_loss /= epoch_length
        self.train_losses.append(train_loss)
        self.dist_losses.append(dist_loss)
        self.tv_losses.append(tv_loss)

    def plot_train_val_loss(self):
        epochs = [x + 1 for x in range(len(self.train_losses))]
        plt.plot(epochs, self.train_losses, 'b', label='Training loss')
        if len(self.val_losses) > 0:
            plt.plot(epochs, self.val_losses, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(self.current_dir + '/final_results/train_val_loss_plt.png')
        plt.close()

    def plot_separate_loss(self):
        epochs = [x + 1 for x in range(len(self.train_losses))]
        weights = np.array([self.config.dist_weight, self.config.tv_weight])
        number_of_subplots = weights[weights > 0].astype(bool).sum()
        fig, axes = plt.subplots(nrows=1, ncols=number_of_subplots, figsize=(6 * number_of_subplots, 2 * number_of_subplots), squeeze=False)
        idx = 0
        for weight, train_loss, label in zip(weights, [self.dist_losses,  self.tv_losses], ['Distance loss', 'Total Variation loss']):
            if weight > 0:
                axes[0, idx].plot(epochs, train_loss, c='b', label='Train')
                axes[0, idx].set_xlabel('Epoch')
                axes[0, idx].set_ylabel('Loss')
                axes[0, idx].set_title(label)
                axes[0, idx].legend(loc='upper right')
                axes[0, idx].xaxis.set_major_locator(MaxNLocator(integer=True))
                idx += 1
        fig.tight_layout()
        plt.savefig(self.current_dir + '/final_results/separate_loss_plt.png')
        plt.close()

    def save_final_objects(self):
        alpha = transforms.ToTensor()(Image.open('../prnet/old_uv_templates/new_uv.png').convert('L'))
        final_patch = torch.cat([self.best_patch.squeeze(0), alpha])
        final_patch_img = transforms.ToPILImage()(final_patch.squeeze(0))
        final_patch_img.save(self.current_dir + '/final_results/final_patch.png', 'PNG')
        new_size = tuple(self.config.magnification_ratio * s for s in self.config.img_size)
        transforms.Resize(new_size)(final_patch_img).save(self.current_dir + '/final_results/final_patch_magnified.png', 'PNG')

        torch.save(self.best_patch, self.current_dir + '/final_results/final_patch_raw.pt')

        with open(self.current_dir + '/losses/train_losses', 'wb') as fp:
            pickle.dump(self.train_losses, fp)
        with open(self.current_dir + '/losses/val_losses', 'wb') as fp:
            pickle.dump(self.val_losses, fp)
        with open(self.current_dir + '/losses/dist_losses', 'wb') as fp:
            pickle.dump(self.dist_losses, fp)
        with open(self.current_dir + '/losses/tv_losses', 'wb') as fp:
            pickle.dump(self.tv_losses, fp)


def main():
    mode = 'private'
    # mode = 'cluster'
    adv_mask = AdversarialMask(mode)
    adv_mask.train()
    evaluator = Evaluator(adv_mask)
    evaluator.test()


if __name__ == '__main__':
    main()
