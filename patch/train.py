import sys
import os

if sys.base_prefix.__contains__('home/zolfi'):
    sys.path.append('/home/zolfi/AdversarialMask')
    sys.path.append('/home/zolfi/AdversarialMask/patch')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
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
print('device is {}'.format(device), flush=True)


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


class AdversarialMask:
    def __init__(self, config):
        self.config = config
        set_random_seed(seed_value=self.config.seed)

        self.train_no_aug_loader, self.train_loader = utils.get_train_loaders(self.config)

        self.embedders = load_embedder(self.config.train_embedder_names, device)

        face_landmark_detector = utils.get_landmark_detector(self.config, device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        # self.preds = self.load_landmarks()
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size).to(device)
        self.normalize_arcface = NormalizeToArcFace(device).to(device)
        self.total_variation = TotalVariation(device).to(device)
        self.dist_loss = losses.get_loss(self.config)

        self.train_losses_epoch = []
        self.train_losses_iter = []
        self.dist_losses = []
        self.tv_losses = []
        self.val_losses = []

        self.create_folders()
        utils.save_class_to_file(self.config, self.config.current_dir)
        self.target_embedding = utils.get_person_embedding(self.config, self.train_no_aug_loader, self.config.celeb_lab_mapper, self.location_extractor,
                                                           self.fxz_projector, self.embedders, device)
        self.best_patch = None

    def create_folders(self):
        Path('/'.join(self.config.current_dir.split('/')[:2])).mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/sim-boxes').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/pr-curves').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/stats/similarity').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/stats/average_precision').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/saved_preds').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/saved_patches').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/saved_similarities').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/losses').mkdir(parents=True, exist_ok=True)

    def train(self):
        adv_patch_cpu = self.get_patch(self.config.initial_patch)
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(current_dir=self.config.current_dir, patience=self.config.es_patience, init_patch=adv_patch_cpu)
        epoch_length = len(self.train_loader)
        for epoch in range(self.config.epochs):
            train_loss = 0.0
            dist_loss = 0.0
            tv_loss = 0.0
            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=epoch_length)
            prog_bar_desc = 'train-loss: {:.6}, dist-loss: {:.6}, tv-loss: {:.6}, lr: {:.6}'
            for i_batch, (img_batch, img_names, cls_id) in progress_bar:
                (b_loss, sep_loss), vars = self.forward_step(img_batch, adv_patch_cpu, img_names, cls_id)

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
                self.train_losses_iter.append(train_loss / (i_batch + 1))
                if i_batch + 1 == epoch_length:
                    # self.calc_validation(adv_patch_cpu)
                    self.save_losses(epoch_length, train_loss, dist_loss, tv_loss)
                    # prog_bar_desc += ', val-loss: {:.6}'
                    progress_bar.set_postfix_str(prog_bar_desc.format(self.train_losses_epoch[-1],
                                                                      self.dist_losses[-1],
                                                                      self.tv_losses[-1],
                                                                      optimizer.param_groups[0]["lr"], ))
                                                                      # self.val_losses[-1]))
                for var in vars + sep_loss:
                    del var
                del b_loss
                torch.cuda.empty_cache()
            if early_stop(self.train_losses_epoch[-1], adv_patch_cpu, epoch):
                self.best_patch = adv_patch_cpu
                break

            scheduler.step(self.train_losses_epoch[-1])
        self.best_patch = early_stop.best_patch
        self.save_final_objects()
        self.plot_train_val_loss(self.train_losses_epoch, 'Epoch')
        self.plot_train_val_loss(self.train_losses_iter, 'Iterations')
        self.plot_separate_loss()

    def loss_fn(self, patch_embs, tv_loss, cls_id):
        distance_loss = torch.empty(0, device=device)
        for target_embedding, (emb_name, patch_emb) in zip(self.target_embedding.values(), patch_embs.items()):
            target_embeddings = torch.index_select(target_embedding, index=cls_id, dim=0).squeeze(-2)
            distance = self.dist_loss(patch_emb, target_embeddings)
            single_embedder_dist_loss = torch.mean(distance).unsqueeze(0)
            distance_loss = torch.cat([distance_loss, single_embedder_dist_loss], dim=0)
        distance_loss = self.config.dist_weight * distance_loss.mean()
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
            size = 16
            for i in range(int(patch.size()[2]/size)):
                patch[0, 0, int(patch.size()[2]/size)*i:int(patch.size()[2]/size)*(i+1), :] = random.randint(0, 255) / 255
                patch[0, 1, int(patch.size()[2]/size)*i:int(patch.size()[2]/size)*(i+1), :] = random.randint(0, 255) / 255
                patch[0, 2, int(patch.size()[2]/size)*i:int(patch.size()[2]/size)*(i+1), :] = random.randint(0, 255) / 255
            for i in range(0, int(patch.size()[3]/size), 2):
                patch[0, 0, :, int(patch.size()[3]/size)*i:int(patch.size()[3]/size)*(i+1)] = random.randint(0, 255) / 255
                patch[0, 1, :, int(patch.size()[3]/size)*i:int(patch.size()[3]/size)*(i+1)] = random.randint(0, 255) / 255
                patch[0, 2, :, int(patch.size()[3]/size)*i:int(patch.size()[3]/size)*(i+1)] = random.randint(0, 255) / 255
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
        # transforms.ToPILImage()(patch.squeeze(0) * uv_face).save('../data/masks/random.png')
        # transforms.ToPILImage()(patch.squeeze(0) * uv_face).show()
        return patch

    def forward_step(self, img_batch, adv_patch_cpu, img_names, cls_id):
        img_batch = img_batch.to(device)
        adv_patch = adv_patch_cpu.to(device)
        cls_id = cls_id.to(device)

        preds = self.location_extractor(img_batch)

        img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch, do_aug=self.config.mask_aug)

        patch_embs = {}
        for embedder_name, emb_model in self.embedders.items():
            patch_embs[embedder_name] = emb_model(img_batch_applied)

        tv_loss = self.total_variation(adv_patch)
        loss = self.loss_fn(patch_embs, tv_loss, cls_id)

        return loss, [img_batch, adv_patch, img_batch_applied, patch_embs, tv_loss]

    def calc_validation(self, adv_patch_cpu):
        val_loss = 0.0
        with torch.no_grad():
            for img_batch, img_names, cls_id in self.val_loader:
                img_batch = img_batch.to(device)
                (loss, _), _ = self.forward_step(img_batch, adv_patch_cpu, img_names, cls_id, train=False)
                val_loss += loss.item()

                del img_batch, loss
                torch.cuda.empty_cache()
        val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(val_loss)

    def save_losses(self, epoch_length, train_loss, dist_loss, tv_loss):
        train_loss /= epoch_length
        dist_loss /= epoch_length
        tv_loss /= epoch_length
        self.train_losses_epoch.append(train_loss)
        self.dist_losses.append(dist_loss)
        self.tv_losses.append(tv_loss)

    def plot_train_val_loss(self, loss, loss_type):
        xticks = [x + 1 for x in range(len(loss))]
        plt.plot(xticks, loss, 'b', label='Training loss')
        # if len(self.val_losses) > 0:
        #     plt.plot(xticks, self.val_losses, 'r', label='Validation loss')
        plt.title('Training loss')
        plt.xlabel(loss_type)
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(self.config.current_dir + '/final_results/train_loss_' + loss_type.lower() + '_plt.png')
        plt.close()

    def plot_separate_loss(self):
        epochs = [x + 1 for x in range(len(self.train_losses_epoch))]
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
        plt.savefig(self.config.current_dir + '/final_results/separate_loss_plt.png')
        plt.close()

    def save_final_objects(self):
        alpha = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))
        final_patch = torch.cat([self.best_patch.squeeze(0), alpha])
        final_patch_img = transforms.ToPILImage()(final_patch.squeeze(0))
        final_patch_img.save(self.config.current_dir + '/final_results/final_patch.png', 'PNG')
        new_size = tuple(self.config.magnification_ratio * s for s in self.config.img_size)
        transforms.Resize(new_size)(final_patch_img).save(self.config.current_dir + '/final_results/final_patch_magnified.png', 'PNG')
        torch.save(self.best_patch, self.config.current_dir + '/final_results/final_patch_raw.pt')

        with open(self.config.current_dir + '/losses/train_losses', 'wb') as fp:
            pickle.dump(self.train_losses_epoch, fp)
        with open(self.config.current_dir + '/losses/val_losses', 'wb') as fp:
            pickle.dump(self.val_losses, fp)
        with open(self.config.current_dir + '/losses/dist_losses', 'wb') as fp:
            pickle.dump(self.dist_losses, fp)
        with open(self.config.current_dir + '/losses/tv_losses', 'wb') as fp:
            pickle.dump(self.tv_losses, fp)


def main():
    mode = 'private'
    # mode = 'cluster'
    config = patch_config_types[mode]()
    print('Starting train...', flush=True)
    adv_mask = AdversarialMask(config)
    adv_mask.train()
    print('Finished train...', flush=True)
    print('Starting test...', flush=True)
    evaluator = Evaluator(adv_mask)
    evaluator.test()
    print('Finished test...', flush=True)


if __name__ == '__main__':
    main()
