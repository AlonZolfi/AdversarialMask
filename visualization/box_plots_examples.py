import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

mask_names = ['Clean', 'Adv', 'Random', 'Blue', 'Black', 'White']


def load_all_mask_similarities(raw_data_path, mask_name):
    sims = np.empty(0)
    for file_name in glob.glob(os.path.join(raw_data_path, '**', mask_name + '.npy'), recursive=True):
        with open(file_name, 'rb') as f:
            sim = np.load(f).squeeze()
            sims = np.concatenate([sims, sim])
    return sims


def plot_sim_box(similarities):
    sim_df = pd.DataFrame()
    for i in range(len(similarities)):
        sim_df[mask_names[i]] = similarities[i]
    sorted_index = sim_df.mean().sort_values(ascending=False).index
    sim_df_sorted = sim_df[sorted_index]
    sns.boxplot(data=sim_df_sorted).set_title('Similarities Difference')
    plt.xlabel('Mask Type')
    plt.ylabel('Similarity')
    plt.savefig('combined_box_plot.png')


def gather_sim_and_plot():
    sims = []
    for mask_name in mask_names:
        sims.append(load_all_mask_similarities('raw_data/1', mask_name))
    plot_sim_box(sims)


gather_sim_and_plot()
