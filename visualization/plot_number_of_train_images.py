import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt


def load_all_mask_similarities(raw_data_path, mask_name):
    sims = np.empty(0)
    for file_name in glob.glob(os.path.join(raw_data_path, '**', mask_name + '.npy'), recursive=True):
        with open(file_name, 'rb') as f:
            sim = np.load(f).squeeze()
            sims = np.concatenate([sims, sim])
    return sims


def plot_dependence(raw_data_path):
    number_of_images = [1, 2, 3, 4, 5, 6, 7, 10, 15]
    sims_avg = []
    list_dir = os.listdir(raw_data_path)
    for i in range(int(len(list_dir)/5)):
        current_avg = []
        for j in range(5):
            folder_name = list_dir[(i * 5) + j]
            current_avg.append(load_all_mask_similarities(os.path.join(raw_data_path, folder_name), 'Adv').mean())
        sims_avg.append(sum(current_avg)/5)

    plt.plot(number_of_images, sims_avg, marker='o')
    plt.ylabel('Similarity')
    plt.xlabel('Number of training Images')
    plt.xticks(number_of_images)
    plt.title('Effect of training images amount on the similarity')
    plt.savefig('amount_of_training_data.png')


plot_dependence('raw_data/3')
