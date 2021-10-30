import os
import pandas as pd
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


def append_mask_data(folder_name, folder, train_dataset_name, res_type, train_emb_name, col_name):
    single_df = pd.read_csv(
        os.path.join(folder_name, folder, 'final_results', 'stats', 'similarity', train_dataset_name, res_type,
                     'mean_df_person.csv'), header=0, index_col=0)
    if not isinstance(train_emb_name, str):
        train_emb_name = '+'.join(train_emb_name)
    new_row = pd.Series(single_df[col_name], name=train_emb_name)
    return new_row


def extract_stat_results():
    folder_name = 'copied'
    folders = os.listdir(folder_name)
    df_with = pd.DataFrame()
    df_without = pd.DataFrame()

    for folder in folders:
        with open(os.path.join(folder_name, folder, 'config.json')) as cfg:
            conf_file_data = json.load(cfg)
        train_emb_name = conf_file_data['train_embedder_names']
        train_dataset_name = conf_file_data['train_dataset_name']
        new_row = append_mask_data(folder_name, folder, train_dataset_name, 'with', train_emb_name, 'Adv')
        df_with = df_with.append(new_row)
        new_row = append_mask_data(folder_name, folder, train_dataset_name, 'without', train_emb_name, 'Adv')
        df_without = df_with.append(new_row)

    mask_names = ['Clean', 'Random', 'Blue', 'Black', 'White', 'Face1', "Face2", 'Face3']
    for mask in mask_names:
        new_row = append_mask_data(folder_name, folders[0], train_dataset_name, 'with', mask, mask)
        df_with = df_with.append(new_row)
        new_row = append_mask_data(folder_name, folders[0], train_dataset_name, 'without', mask, mask)
        df_without = df_without.append(new_row)

    df_with.to_csv('with_results.csv')
    df_without.to_csv('without_results.csv')


def load_sims(root_folder, lab, mask_name):
    with open(os.path.join(root_folder, 'saved_similarities', 'CASIA', 'resnet100_magface', lab, 'with_mask_' + mask_name + '.pickle'), 'rb') as f:
        person_sims = []
        while True:
            try:
                data = pickle.load(f).values()
                person_sims.extend(list(data))
            except EOFError:
                break
        person_avg_sim = sum(person_sims) / len(person_sims)
        return person_avg_sim


def create_sub_box_plots(job_id_universal, job_id_targeted):
    root_folder = 'October'
    for folder in os.listdir(root_folder):
        if job_id_universal in folder:
            folder_name = folder
            break
    full_path = os.path.join(root_folder, folder_name, 'saved_similarities', 'CASIA', 'resnet100_magface')
    persons_folder = [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]
    sims = {'Clean': [], 'Random': [], 'Blue': [], 'Face1': [], 'Face3': [], 'Adv': [], 'Targeted': []}
    for p_folder in persons_folder:
        for mask_name in sims.keys():
            if mask_name == 'Targeted':
                break
            sim = load_sims(os.path.join(root_folder, folder_name), p_folder, mask_name)
            sims[mask_name].append(sim)

    for folder in os.listdir(root_folder):
        if job_id_targeted in folder:
            folder_name = folder
            break

    full_path = os.path.join(root_folder, folder_name)
    for id_folder in os.listdir(full_path):
        if job_id_targeted in id_folder:
            stats_df = pd.read_csv(os.path.join(full_path, id_folder, 'final_results', 'stats', 'similarity', 'CASIA', 'with', 'mean_df_person.csv'), header=0, index_col=0)
            sim = stats_df.at['resnet100_magface', 'Adv']
            sims['Targeted'].append(sim)

    sim_df = pd.DataFrame()
    for i, mask_name in enumerate(sims.keys()):
        if mask_name == 'Face1':
            mask_name1 = 'Male\nFace'
        elif mask_name == 'Face3':
            mask_name1 = 'Female\nFace'
        else:
            mask_name1 = mask_name
        sim_df[mask_name1] = sims[mask_name]
    sorted_index = sim_df.mean().sort_values(ascending=False).index
    sim_df_sorted = sim_df[sorted_index]
    f, ax = plt.subplots()
    ax.tick_params(labeltop=False, labelright=True, labelleft=True)
    ax.yaxis.set_ticks_position('both')
    ax.set_ylim(bottom=-0.25, top=0.95)
    ax.locator_params(nbins=13, axis='y')
    sns.boxplot(data=sim_df_sorted, ax=ax)
    plt.xlabel('Mask Type', labelpad=7)
    plt.ylabel('Cosine Similarity')
    plt.savefig(os.path.join('simbox.png'))
    plt.close()


if __name__ == '__main__':
    # extract_stat_results()
    create_sub_box_plots('165132', '164142')
