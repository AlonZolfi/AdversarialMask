import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_heatmap(file_path):
    df = pd.read_csv(file_path, header=0, index_col=0)
    plt.yticks(rotation=0)
    f, ax = plt.subplots(1)
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    hm = sns.heatmap(df, cbar_kws={'label': 'Cosine Similarity'}, ax=ax, annot=True, cmap='coolwarm_r', fmt='.3g')
    hm.set_yticklabels(hm.get_yticklabels(), rotation=20)
    ax.hlines([4, len(df)-1], *ax.get_xlim())
    plt.xlabel('Victim Models', labelpad=7)
    plt.ylabel('Mask / Training Model')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.sep.join(file_path.split(os.path.sep)[:-1]),  'heads_heatmap.png'), dpi=1000)


plot_heatmap('C:\\Users\\Administrator\\Desktop\\University\\Work\\Results\\Universal\\heads\\heads_200.csv')


def plot_scatter(file_path):
    df = pd.read_csv(file_path, header=0)
    f, ax = plt.subplots()
    sns.barplot(data=df, x="Victim Model", y='Cosine Similarity', hue="Mask / Training Model")
    plt.xlabel('Victim Models')
    plt.ylabel('Mask / Training Model')
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(os.path.sep.join(file_path.split(os.path.sep)[:-1]),  'heads_scatter.png'), dpi=1000)


# plot_scatter('C:\\Users\\Administrator\\Desktop\\University\\Work\\Results\\Universal\\heads\\heads.csv')