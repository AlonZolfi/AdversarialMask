import warnings
import utils
import torch
import os
from torch.nn import CosineSimilarity
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import seaborn as sns
import pandas as pd
matplotlib.use('Agg')

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, adv_mask_class) -> None:
        super().__init__()
        self.config = adv_mask_class.config
        self.adv_mask_class = adv_mask_class
        self.transform = transforms.Compose([transforms.Resize(self.config.patch_size), transforms.ToTensor()])
        self.random_mask_t = utils.load_mask(self.config, self.config.random_mask_path, device)
        self.blue_mask_t = utils.load_mask(self.config, self.config.blue_mask_path, device)
        self.black_mask_t = utils.load_mask(self.config, self.config.black_mask_path, device)
        self.white_mask_t = utils.load_mask(self.config, self.config.white_mask_path, device)
        self.mask_names = ['Clean', 'Adv', 'Random', 'Blue', 'Black', 'White']

    @torch.no_grad()
    def get_similarity(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            all_test_image_clean, all_test_image_adv, all_test_image_random, all_test_image_blue, all_test_image_black, all_test_image_white = self.init_embeddings()

            adv_patch = self.adv_mask_class.best_patch.to(device)
            for img_batch, img_names in tqdm(self.adv_mask_class.test_loader):
                img_batch = img_batch.to(device)
                # Apply different types of masks
                img_batch_applied_adv, img_batch_applied_random, img_batch_applied_blue, img_batch_applied_black, img_batch_applied_white = \
                    self.apply_all_masks(img_batch, adv_patch)

                # Get embedding
                batch_emb_clean, batch_emb_adv, batch_emb_random, batch_emb_blue, batch_emb_black, batch_emb_white = \
                    self.get_all_embeddings(img_batch, img_batch_applied_adv, img_batch_applied_random, img_batch_applied_blue, img_batch_applied_black, img_batch_applied_white)

                # Save all embeddings
                all_test_image_clean, all_test_image_adv, all_test_image_random, all_test_image_blue, all_test_image_black, all_test_image_white = \
                    self.save_all_embeddings(all_test_image_clean, all_test_image_adv, all_test_image_random, all_test_image_blue, all_test_image_black, all_test_image_white,
                        batch_emb_clean, batch_emb_adv, batch_emb_random, batch_emb_blue, batch_emb_black, batch_emb_white)

        clean_similarity, adv_similarity, random_similarity, blue_similarity, black_similarity, white_similarity = \
            self.calc_all_similarity(all_test_image_clean, all_test_image_adv, all_test_image_random, all_test_image_blue, all_test_image_black, all_test_image_white)

        return clean_similarity, adv_similarity, random_similarity, blue_similarity, black_similarity, white_similarity

    def test(self):
        if self.config.is_real_person:
            return
        similarities = self.get_similarity()
        self.plot_sim_box(similarities)
        # precisions, recalls, thresholds, aps = self.get_pr(similarities)
        # self.plot_pr_curve(precisions, recalls, thresholds, aps)

    def plot_sim_box(self, similarities):
        sim_df = pd.DataFrame()
        for i in range(len(similarities)):
            sim_df[self.mask_names[i]] = similarities[i].squeeze()
        sorted_index = sim_df.mean().sort_values(ascending=False).index
        sim_df_sorted = sim_df[sorted_index]
        sns.boxplot(data=sim_df_sorted).set_title('Similarities Difference')
        plt.xlabel('Mask Type')
        plt.ylabel('Similarity')
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

    def write_similarities_to_disk(self, clean_similarity, adv_similarity, random_similarity, blue_similarity,
                                   black_similarity, white_similarity):
        for similarity, mask_name in zip([clean_similarity, adv_similarity, random_similarity, blue_similarity,
                                          black_similarity, white_similarity], self.mask_names):
            with open(os.path.join(self.adv_mask_class.current_dir, 'saved_similarities', mask_name + '.npy'), 'wb') as f:
                np.save(f, similarity)

    def apply_all_masks(self, img_batch, adv_patch):
        img_batch_applied_adv = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                 self.adv_mask_class.fxz_projector, img_batch, adv_patch)
        img_batch_applied_random = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                    self.adv_mask_class.fxz_projector, img_batch, self.random_mask_t)
        img_batch_applied_blue = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                  self.adv_mask_class.fxz_projector, img_batch, self.blue_mask_t[:, :3],
                                                  self.blue_mask_t[:, 3])
        img_batch_applied_black = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                   self.adv_mask_class.fxz_projector, img_batch,
                                                   self.black_mask_t[:, :3],
                                                   self.black_mask_t[:, 3])
        img_batch_applied_white = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                   self.adv_mask_class.fxz_projector, img_batch,
                                                   self.white_mask_t[:, :3],
                                                   self.white_mask_t[:, 3])

        return img_batch_applied_adv, img_batch_applied_random, img_batch_applied_blue, img_batch_applied_black, img_batch_applied_white

    def get_all_embeddings(self, img_batch, img_batch_applied_adv, img_batch_applied_random, img_batch_applied_blue,
                           img_batch_applied_black, img_batch_applied_white):
        batch_emb_clean = self.adv_mask_class.embedder(img_batch.to(device)).cpu().numpy()
        batch_emb_adv = self.adv_mask_class.embedder(img_batch_applied_adv.to(device)).cpu().numpy()
        batch_emb_random = self.adv_mask_class.embedder(img_batch_applied_random.to(device)).cpu().numpy()
        batch_emb_blue = self.adv_mask_class.embedder(img_batch_applied_blue.to(device)).cpu().numpy()
        batch_emb_black = self.adv_mask_class.embedder(img_batch_applied_black.to(device)).cpu().numpy()
        batch_emb_white = self.adv_mask_class.embedder(img_batch_applied_white.to(device)).cpu().numpy()
        return batch_emb_clean, batch_emb_adv, batch_emb_random, batch_emb_blue, batch_emb_black, batch_emb_white

    def save_all_embeddings(self, all_test_image_clean, all_test_image_adv, all_test_image_random, all_test_image_blue,
                            all_test_image_black, all_test_image_white,
                            batch_emb_clean, batch_emb_adv, batch_emb_random, batch_emb_blue, batch_emb_black,
                            batch_emb_white):
        all_test_image_clean = np.concatenate([all_test_image_clean, batch_emb_clean])
        all_test_image_adv = np.concatenate([all_test_image_adv, batch_emb_adv])
        all_test_image_random = np.concatenate([all_test_image_random, batch_emb_random])
        all_test_image_blue = np.concatenate([all_test_image_blue, batch_emb_blue])
        all_test_image_black = np.concatenate([all_test_image_black, batch_emb_black])
        all_test_image_white = np.concatenate([all_test_image_white, batch_emb_white])
        return all_test_image_clean, all_test_image_adv, all_test_image_random, all_test_image_blue, all_test_image_black, all_test_image_white

    def calc_all_similarity(self, all_test_image_clean, all_test_image_adv, all_test_image_random, all_test_image_blue,
                            all_test_image_black, all_test_image_white):
        target_embedding = self.adv_mask_class.test_target_embedding.cpu().numpy()
        clean_similarity = cosine_similarity(all_test_image_clean, target_embedding)
        adv_similarity = cosine_similarity(all_test_image_adv, target_embedding)
        random_similarity = cosine_similarity(all_test_image_random, target_embedding)
        blue_similarity = cosine_similarity(all_test_image_blue, target_embedding)
        black_similarity = cosine_similarity(all_test_image_black, target_embedding)
        white_similarity = cosine_similarity(all_test_image_white, target_embedding)
        self.write_similarities_to_disk(clean_similarity, adv_similarity, random_similarity, blue_similarity,
                                        black_similarity, white_similarity)
        return clean_similarity, adv_similarity, random_similarity, blue_similarity, black_similarity, white_similarity

    def init_embeddings(self):
        all_test_image_clean = np.empty((0, 512))
        all_test_image_adv = np.empty((0, 512))
        all_test_image_random = np.empty((0, 512))
        all_test_image_blue = np.empty((0, 512))
        all_test_image_black = np.empty((0, 512))
        all_test_image_white = np.empty((0, 512))
        return all_test_image_clean, all_test_image_adv, all_test_image_random, all_test_image_blue, all_test_image_black, all_test_image_white

