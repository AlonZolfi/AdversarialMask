import sys
import os

if sys.base_prefix.__contains__('home/zolfi'):
    sys.path.append('/home/zolfi/AdversarialMask')
    sys.path.append('/home/zolfi/AdversarialMask/patch')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
import random
from pathlib import Path
import pickle

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

import utils
import losses
from config import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation
from utils import load_embedder, EarlyStopping, get_patch


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
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size).to(device)
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
        adv_patch_cpu = utils.get_patch(self.config)
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
            for i_batch, (img_batch, _, cls_id) in progress_bar:
                (b_loss, sep_loss), vars = self.forward_step(img_batch, adv_patch_cpu, cls_id)

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
                    self.save_losses(epoch_length, train_loss, dist_loss, tv_loss)
                    progress_bar.set_postfix_str(prog_bar_desc.format(self.train_losses_epoch[-1],
                                                                      self.dist_losses[-1],
                                                                      self.tv_losses[-1],
                                                                      optimizer.param_groups[0]["lr"], ))
                del b_loss
                torch.cuda.empty_cache()
            if early_stop(self.train_losses_epoch[-1], adv_patch_cpu, epoch):
                self.best_patch = adv_patch_cpu
                break

            scheduler.step(self.train_losses_epoch[-1])
        self.best_patch = early_stop.best_patch
        self.save_final_objects()
        utils.plot_train_val_loss(self.config, self.train_losses_epoch, 'Epoch')
        utils.plot_train_val_loss(self.config, self.train_losses_iter, 'Iterations')
        utils.plot_separate_loss(self.config, self.train_losses_epoch, self.dist_losses, self.tv_losses)

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

    def forward_step(self, img_batch, adv_patch_cpu, cls_id):
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

    def save_losses(self, epoch_length, train_loss, dist_loss, tv_loss):
        train_loss /= epoch_length
        dist_loss /= epoch_length
        tv_loss /= epoch_length
        self.train_losses_epoch.append(train_loss)
        self.dist_losses.append(dist_loss)
        self.tv_losses.append(tv_loss)

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
    mode = 'universal'
    config = patch_config_types[mode]()
    print('Starting train...', flush=True)
    adv_mask = AdversarialMask(config)
    adv_mask.train()
    print('Finished train...', flush=True)


if __name__ == '__main__':
    main()
