import warnings

import torch
from torch.nn import CosineSimilarity
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import seaborn as sns
import pandas as pd
from brambox.stat import pr
matplotlib.use('Agg')

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, adv_mask_class) -> None:
        super().__init__()
        self.config = adv_mask_class.config
        self.adv_mask_class = adv_mask_class
        self.transform = transforms.Compose([transforms.Resize(self.config.patch_size), transforms.ToTensor()])
        self.blue_mask_t = self.load_mask(self.config.blue_mask_path)
        self.black_mask_t = self.load_mask(self.config.black_mask_path)
        self.white_mask_t = self.load_mask(self.config.white_mask_path)
        self.mask_names = ['Clean', 'Adv', 'Blue', 'Black', 'White']

    @torch.no_grad()
    def load_mask(self, mask_path):
        img = Image.open(mask_path)
        img_t = self.transform(img).unsqueeze(0).to(device)
        return img_t

    @torch.no_grad()
    def apply_mask(self, img_batch, img_names, patch_rgb, patch_alpha=None):
        preds = self.adv_mask_class.get_batch_landmarks(img_names)
        img_batch_applied = self.adv_mask_class.fxz_projector(img_batch, preds, patch_rgb, patch_alpha)
        return img_batch_applied

    @torch.no_grad()
    def get_similarity(self):
        all_test_image_clean = np.empty((0, 512))
        all_test_image_adv = np.empty((0, 512))
        all_test_image_blue = np.empty((0, 512))
        all_test_image_black = np.empty((0, 512))
        all_test_image_white = np.empty((0, 512))
        adv_patch = self.adv_mask_class.best_patch.to(device)
        for img_batch, img_names in self.adv_mask_class.test_loader:
            img_batch = img_batch.to(device)
            # Apply different types of masks
            img_batch_applied_adv = self.apply_mask(img_batch, img_names, adv_patch)
            img_batch_applied_blue = self.apply_mask(img_batch, img_names, self.blue_mask_t[:, :3],
                                                     self.blue_mask_t[:, 3])
            img_batch_applied_black = self.apply_mask(img_batch, img_names, self.black_mask_t[:, :3],
                                                      self.black_mask_t[:, 3])
            img_batch_applied_white = self.apply_mask(img_batch, img_names, self.white_mask_t[:, :3],
                                                      self.white_mask_t[:, 3])

            # Get embedding
            batch_emb_clean = self.adv_mask_class.embedder(img_batch.to(device)).cpu().numpy()
            batch_emb_adv = self.adv_mask_class.embedder(img_batch_applied_adv.to(device)).cpu().numpy()
            batch_emb_blue = self.adv_mask_class.embedder(img_batch_applied_blue.to(device)).cpu().numpy()
            batch_emb_black = self.adv_mask_class.embedder(img_batch_applied_black.to(device)).cpu().numpy()
            batch_emb_white = self.adv_mask_class.embedder(img_batch_applied_white.to(device)).cpu().numpy()

            # Save all embeddings
            all_test_image_clean = np.concatenate([all_test_image_clean, batch_emb_clean])
            all_test_image_adv = np.concatenate([all_test_image_adv, batch_emb_adv])
            all_test_image_blue = np.concatenate([all_test_image_blue, batch_emb_blue])
            all_test_image_black = np.concatenate([all_test_image_black, batch_emb_black])
            all_test_image_white = np.concatenate([all_test_image_white, batch_emb_white])

        target_embedding = self.adv_mask_class.target_embedding.cpu().numpy()
        clean_similarity = np.clip(cosine_similarity(all_test_image_clean, target_embedding), a_min=0, a_max=1)
        adv_similarity = np.clip(cosine_similarity(all_test_image_adv, target_embedding), a_min=0, a_max=1)
        blue_similarity = np.clip(cosine_similarity(all_test_image_blue, target_embedding), a_min=0, a_max=1)
        black_similarity = np.clip(cosine_similarity(all_test_image_black, target_embedding), a_min=0, a_max=1)
        white_similarity = np.clip(cosine_similarity(all_test_image_white, target_embedding), a_min=0, a_max=1)

        return clean_similarity, adv_similarity, blue_similarity, black_similarity, white_similarity

    def test(self):
        similarities = self.get_similarity()
        self.plot_sim_box(similarities)
        # precisions, recalls, thresholds, aps = self.get_pr(similarities)
        # self.plot_pr_curve(precisions, recalls, thresholds, aps)

    def plot_sim_box(self, similarities):
        sim_df = pd.DataFrame(columns=['Similarity', 'Mask Type'])
        for i in range(len(similarities)):
            tmp_df = pd.DataFrame(columns=['Similarity', 'Mask Type'])
            tmp_df['Similarity'] = similarities[i].squeeze()
            tmp_df['Mask Type'] = self.mask_names[i]
            sim_df = sim_df.append(tmp_df)
        ax = sns.boxplot(x="Mask Type", y="Similarity", data=sim_df)
        plt.savefig(self.adv_mask_class.current_dir + '/final_results/sim-boxes.png')
    '''def get_pr(self, similarities):
        y_true = np.ones(similarities[0].shape[0])

        precisions, recalls, thresholds, aps = [], [], [], []
        for similarity in similarities:
            precision, recall, _ = pr(pd.DataFrame(y_true), pd.DataFrame(similarity.cpu().numpy()))
            ap = average_precision_score(y_true, similarity.cpu().numpy())
            precisions.append(precision)
            recalls.append(recall)
            aps.append(ap)

        return precisions, recalls, thresholds, aps'''

    '''def plot_pr_curve(self, precisions, recalls, thresholds, aps):
        plt.plot([0, 1.05], [0, 1.05], '--', color='gray')
        title = 'Precision-Recall Curve'
        plt.title(title)
        for i in range(len(precisions)):
            plt.plot(recalls[i], precisions[i], label='{}: AP: {}%'.format(self.mask_names[i], round(aps[i] * 100, 2)))

        plt.gca().set_ylabel('Precision')
        plt.gca().set_xlabel('Recall')
        plt.gca().set_xlim([0, 1.05])
        plt.gca().set_ylim([0, 1.05])

        handles, labels = plt.gca().get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles),
                                      key=lambda t: float(t[0].split('AP: ')[1].replace('%', '')),
                                      reverse=True))
        plt.gca().legend(handles, labels, loc=4)
        plt.imshow()
        plt.savefig(self.adv_mask_class.current_dir + '/final_results/pr-curve.png')'''
