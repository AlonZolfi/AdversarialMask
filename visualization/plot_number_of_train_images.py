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
    number_of_images = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
    sims_avg = []
    list_dir = ['experiments/July/13-07-2021_19-36-12', 'experiments/July/13-07-2021_19-36-53',
                'experiments/July/13-07-2021_19-37-29',
                'experiments/July/13-07-2021_19-38-03', 'experiments/July/13-07-2021_19-38-45',
                'experiments/July/13-07-2021_19-39-25', 'experiments/July/13-07-2021_19-40-07',
                'experiments/July/13-07-2021_19-41-02', 'experiments/July/13-07-2021_19-42-03',
                'experiments/July/13-07-2021_19-42-46', 'experiments/July/13-07-2021_19-43-48',
                'experiments/July/13-07-2021_19-44-52', 'experiments/July/13-07-2021_19-45-59',
                'experiments/July/13-07-2021_19-47-31', 'experiments/July/13-07-2021_19-48-45',
                'experiments/July/13-07-2021_19-49-57', 'experiments/July/13-07-2021_19-51-06',
                'experiments/July/13-07-2021_19-52-25', 'experiments/July/13-07-2021_19-53-30',
                'experiments/July/13-07-2021_19-54-54', 'experiments/July/13-07-2021_19-56-41',
                'experiments/July/13-07-2021_19-58-22', 'experiments/July/13-07-2021_19-59-59',
                'experiments/July/13-07-2021_20-01-50', 'experiments/July/13-07-2021_20-03-44',
                'experiments/July/13-07-2021_20-05-35', 'experiments/July/13-07-2021_20-07-49',
                'experiments/July/13-07-2021_20-09-33', 'experiments/July/13-07-2021_20-11-15',
                'experiments/July/13-07-2021_20-12-52', 'experiments/July/13-07-2021_20-15-39',
                'experiments/July/13-07-2021_20-17-53', 'experiments/July/13-07-2021_20-20-25',
                'experiments/July/13-07-2021_20-23-01', 'experiments/July/13-07-2021_20-26-37',
                'experiments/July/13-07-2021_20-30-01', 'experiments/July/13-07-2021_20-34-51',
                'experiments/July/13-07-2021_20-40-37', 'experiments/July/13-07-2021_20-43-39']
    list_dir = ['../patch/' + folder for folder in list_dir]
    # list_dir = os.listdir(raw_data_path)
    for i in range(int(len(list_dir)/3)):
        current_avg = []
        for j in range(3):
            folder_name = list_dir[(i * 3) + j]
            current_avg.append(load_all_mask_similarities(folder_name, 'Adv').mean())
            # current_avg.append(load_all_mask_similarities(os.path.join(raw_data_path, folder_name), 'Adv').mean())
        sims_avg.append(sum(current_avg)/3)

    plt.plot(number_of_images, sims_avg, marker='o')
    plt.ylabel('Similarity')
    plt.xlabel('Number of training Images')
    plt.xticks(number_of_images)
    plt.title('Effect of training images amount on the similarity')
    plt.savefig('amount_of_training_data.png')


plot_dependence('raw_data/3')
