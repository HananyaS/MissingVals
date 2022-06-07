import os
import math
import tqdm
import torch
import dtale
import warnings
import argparse
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from model import Model
from utils import timer
from sklearn import metrics
from torch.optim import Adam
from itertools import product
from datasets import TrainDataset
from sklearn.cluster import KMeans
from collections import namedtuple
from matplotlib import pyplot as plt
from torch.nn import functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from sklearn.feature_selection import mutual_info_classif as MIC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

str2bool = lambda x: x.lower() == "true"

parser = argparse.ArgumentParser()
parser.add_argument('--using_vals', type=str2bool, default=False)
parser.add_argument('--corr_threshold', type=float, default=-1)
parser.add_argument('--alpha', type=float, default=1)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define Data namedtuple
Data = namedtuple('Data', ['X', 'Y'])


def generate_toy_model():
    n_original_cols = 10
    n_comb_cols = 30
    n_noisy_cols = 20

    n_samples = 1000
    threshold = .6

    w = np.array(range(n_original_cols))
    w = w / np.sum(w)

    core_df = pd.DataFrame(np.random.random((n_samples, n_original_cols)),
                           columns=[f'Original {i + 1}' for i in range(n_original_cols)])
    scores = 1 / (1 + np.exp(-np.matmul(core_df, w)))
    labels = np.array(scores > threshold).astype(int)

    new_df = core_df.copy(deep=True)

    for i in range(n_comb_cols):
        chosen_cols = np.floor(2 * np.random.uniform(size=n_original_cols)).astype(int)
        w_col = np.random.random(len(chosen_cols))
        w_col = w_col / np.sum(w_col)
        new_df[f'Comb {i + 1}'] = np.sum(w_col[i] * chosen_cols[i] * core_df[core_df.columns[i]].values for i in
                                         range(len(chosen_cols)))

    noisy_df = new_df.copy(deep=True)

    for i in range(n_noisy_cols):
        noisy_df[f'Noisy {i + 1}'] = np.random.random(n_samples) + 1

    p_remove = np.random.uniform(size=noisy_df.shape[1])
    print(p_remove)

    for i, j in product(range(noisy_df.shape[0]), range(noisy_df.shape[1])):
        if np.random.uniform() < p_remove[j]:
            noisy_df.iloc[i, j] = np.nan

    final_df = noisy_df.copy(deep=True)
    final_df['Labels'] = labels

    print(final_df)


def load_dataset(ds_path) -> pd.DataFrame:
    df = pd.read_csv(ds_path, index_col=0)

    translations = {
        'מוצא אם': 'Origin Mother',
        'מוצא אב': 'Origin Father',
        'ארץ לידה': 'Country of Birth',
        'עישון': 'Smoking',
        'גיל': 'Age',
        'גובה': 'Height',
        'לפני הריון': 'before Pregnancy',
        'לפני היריון': 'before Pregnancy',
        'משקל': 'Weight',
        'לחץ דם': 'Blood Pressure',
        'הריון בר סיכון': 'High Risk Pregnancy',
    }

    new_cols = {}

    for i, c in enumerate(df.columns):
        c_new = c
        for t_heb, t_eng in translations.items():
            if t_heb in c:
                c_new = c_new.replace(t_heb, t_eng)

        new_cols[c] = c_new

    df.rename(columns=new_cols, inplace=True)
    return df


def calc_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def calc_precision(y_true, y_pred):
    try:
        return len(np.where((y_pred == y_true) & y_true)[0]) / len(np.where(y_pred)[0])
    except ZeroDivisionError:
        return "NaN"


def calc_recall(y_true, y_pred):
    return len(np.where((y_pred == y_true) & y_true)[0]) / len(np.where(y_true)[0])


def calc_auc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def run_clf(clf, X, Y):
    acc_scores, precision_scores, recall_scores, auc_scores = [], [], [], []

    for _ in range(30):
        pred = clf.predict(X)
        acc_scores.append(calc_accuracy(Y.values.reshape(-1), pred))
        # precision_scores.append(calc_precision(Y.values.reshape(-1), pred))
        # recall_scores.append(calc_recall(Y.values.reshape(-1), pred))
        auc_scores.append(calc_auc(Y.values.reshape(-1), pred))

    print(f'Part of bigger class:\t{Y.value_counts(normalize=True).max()}')

    print(f'Accuracy:\t{np.mean(acc_scores)}')
    print(f'Precision:\t{np.mean(precision_scores)}')
    print(f'Recall:\t{np.mean(recall_scores)}')
    print(f'AUC:\t{np.mean(auc_scores)}')


def run_all_clf(clf_lst, train_X, train_Y, test_X, test_Y):
    for clf in clf_lst:
        print(f'~~~~~~~~~~~~~~~~\t{str(clf)[:-2]}\t~~~~~~~~~~~~~~~~')

        print('Train:')
        run_clf(clf, train_X, train_Y)

        print('Test:')
        run_clf(clf, test_X, test_Y)


def find_corr(vals_df, existence_df, alpha=args.alpha, corr_threshold=args.corr_threshold):
    corr_vals_df = vals_df.corr().fillna(1.0)
    corr_existence_df = existence_df.corr().fillna(1.0)
    corr_combined_df = alpha * corr_vals_df + (1 - alpha) * corr_existence_df

    corr_cols = {}

    for i, c in enumerate(corr_combined_df.columns):
        for j in range(i):
            if abs(corr_combined_df.iat[i, j]) > corr_threshold:
                corr_cols[i, j] = corr_combined_df.iat[i, j]

    return corr_cols, corr_combined_df
    #     G.add_weighted_edges_from(i, j, weight=w)


def graph_from_sample(sample, corr_cols):
    feats = np.nan_to_num(sample)

    G = nx.Graph()

    G.add_nodes_from(range(len(feats)))
    nx.set_node_attributes(G, {i: {"value": v} for i, v in enumerate(feats)})
    G.add_weighted_edges_from([i, j, w] for (i, j), w in corr_cols.items() if not math.isnan(sample[i]) and
                              not math.isnan(sample[j]))

    return G


def calc_feat_imputation(df):
    vals_to_replace = {}
    cols = df.columns
    occurences = np.zeros((len(cols), len(cols))).astype(int)
    values = np.zeros((len(cols), len(cols)))

    for _, r in tqdm.tqdm(df.iterrows(), desc='Iterating over rows'):
        r = r.values
        existing_idx = list(filter(lambda x: not math.isnan(x[1]), enumerate(r)))
        for (i, x), (j, y) in product(existing_idx, repeat=2):
            if i == j:
                continue

            occurences[i, j] += 1
            occurences[j, i] += 1
            values[i, j] = x
            values[j, i] = y

    print(occurences)

    for i in range(len(cols)):
        vals_to_replace[i] = sum([values[i, j] / occurences[i, j] for j in range(len(cols)) if occurences[i, j] > 0])

    print(vals_to_replace)


def unsupervised_manner(df):
    labels = df['Labels'].values
    original_df_wo_labels = df.copy(deep=True).drop(columns=['Labels'])

    pca_values = PCA(n_components=2).fit_transform(original_df_wo_labels)

    pc1, pc2 = pca_values[:, 0], pca_values[:, 1]

    pca_df = pd.DataFrame(pca_values, columns=['PC1', 'PC2'])

    kmeans_pred = KMeans(n_clusters=2).fit_predict(pca_df)

    to_colors = lambda x: 'y' if x == 0 else 'c'
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(pc1, pc2, c=list(map(to_colors, kmeans_pred)))
    ax2.scatter(pc1, pc2, c=list(map(to_colors, labels)))

    ax1.set_title('KMeans Result')
    ax2.set_title('True Labels')

    plt.show()


def eval_model(model, loader):
    model.eval()
    true_labels = torch.Tensor().to(device)
    pred_labels = torch.Tensor().to(device)

    for batch, labels in loader:
        batch = batch.to(device)
        labels = labels.to(device)

        true_labels = torch.cat((true_labels, labels))
        pred = model(batch)
        pred = pred.max(1)[1]
        pred_labels = torch.cat((pred_labels, pred))

    auc = calc_auc(true_labels.cpu().numpy(), pred_labels.cpu().numpy())
    return auc


def train_model(model, train_loader, val_loader, optimizer, n_epochs=100):
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []

    for i in range(n_epochs):
        train_loss = 0
        model.train()

        for batch, labels in train_loader:
            optimizer.zero_grad()

            batch = batch.to(device)
            labels = labels.to(device)
            out = model(batch)

            # labels = torch.from_numpy(np.array([[1, 0] if x == 0 else [0, 1] for x in labels.cpu().numpy()])).float().to(device)
            #
            loss = F.nll_loss(out, target=labels, reduction='sum')
            # loss = F.binary_cross_entropy(out, target=labels, reduction='sum')
            # print(loss)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        train_auc = eval_model(model, train_loader)
        print(f'EPOCH {i + 1}')
        print(f'[TRAIN Loss]\t{train_loss}')
        print(f'[TRAIN AUC]\t{train_auc}')

        model.eval()
        val_loss = 0

        for batch, labels in val_loader:
            batch = batch.to(device)
            labels = labels.to(device)

            out = model(batch)

            # labels = torch.from_numpy(np.array([[1, 0] if x == 0 else [0, 1] for x in labels.cpu().numpy()])).float().to(device)
            #
            # val_loss += F.binary_cross_entropy(out, target=labels, reduction='sum').item()
            val_loss += F.nll_loss(out, target=labels, reduction='sum').item()

        val_loss /= len(val_loader.dataset)
        val_auc = eval_model(model, val_loader)

        print(f'[VALIDATION LOSS]\t{val_loss}')
        print(f'[VALIDATION AUC]\t{val_auc}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        print(f'\n~~~~~~~~~~~~~~~')

    plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss')

    plt.title('Loss')

    plt.legend()
    plt.show()

    plt.plot(range(1, n_epochs + 1), train_aucs, label='Train AUC')
    plt.plot(range(1, n_epochs + 1), val_aucs, label='Validation AUC')

    plt.title('AUC')

    plt.legend()
    plt.show()

    return model


@timer
def fill_df(df: pd.DataFrame):
    # get the mean and variance of each column
    means = df.mean()
    stds = df.std()

    df.fillna('NAN', inplace=True)
    # fill the missing values with the mean and variance of the column
    for i, j in itertools.product(range(df.shape[0]), range(df.shape[1])):
        if df.iloc[i, j] == 'NAN':
            df.iloc[i, j] = np.random.normal(means[j], stds[j])

    return df


def load_data(*paths, labels_col='tags'):
    data_list = []

    for p in paths:
        X = pd.read_csv(p)
        Y = X[labels_col]
        X = X.drop(columns=[labels_col])
        data_list.append(Data(X, Y))

    return data_list


def dtale_view(df, title):
    d = dtale.show(df, title=title)
    print(f'Link:\t{d.main_url()}')
    input('Press Enter to continue...')
    return


if __name__ == '__main__':
    TRAIN_PATH = 'Data/Old/splitted_data/train_val_preprocessed.csv'
    TEST_PATH = 'Data/Old/splitted_data/test_preprocessed.csv'

    train_data, test_data = load_data(TRAIN_PATH, TEST_PATH)

    train_X, train_Y = train_data.X, train_data.Y
    test_X, test_Y = test_data.X, test_data.Y

    # train_ratio = .8
    # num_train_labels = int(train_ratio * len(train_X))
    # full_train_X = fill_df(train_X.copy(deep=True))
    # full_test_X = fill_df(test_X.copy(deep=True))

    # train_X = full_train_X
    # test_X = full_test_X

    existence_df = train_X.isnull().astype(int).apply(lambda x: 1 - x)
    score_per_feature = existence_df.sum() / existence_df.shape[0]
    score_per_feature = score_per_feature.values

    sample_scores = []

    for i, r in existence_df.iterrows():
        sample_scores.append(np.dot(r.values, score_per_feature))

    # print('Num of empty samples: ', sample_scores.count(0) / existence_df.shape[0])
    train_X['Sample Score'] = sample_scores
    train_X['Sample Score'] = train_X['Sample Score'] / train_X['Sample Score'].max()
    train_w_labels = pd.concat((train_X, train_Y), axis=1)

    train_w_labels.sort_values(by='Sample Score', inplace=True, ascending=True)
    train_w_labels = train_w_labels.drop(train_w_labels[train_w_labels['Sample Score'] == 0].index)

    # # plot the sample scores distribution
    sample_scores = train_w_labels['Sample Score'].values
    plt.plot(range(1, len(sample_scores) + 1), sample_scores)
    plt.show()

    """
    # # find empty samples
    empty_samples = train_w_labels[train_w_labels['Sample Score'] == 0]
    print(empty_samples['tags'].value_counts() / len(empty_samples))

    # # plot the sample scores distribution among the labels
    mu_neg, sigma_neg = train_w_labels[train_w_labels['tags'] == 0]['Sample Score'].mean(), train_w_labels[
        train_w_labels['tags'] == 0]['Sample Score'].std()
    mu_pos, sigma_pos = train_w_labels[train_w_labels['tags'] == 1]['Sample Score'].mean(), train_w_labels[
        train_w_labels['tags'] == 1]['Sample Score'].std()

    print('mu_neg: ', mu_neg)
    print('sigma_neg: ', sigma_neg)
    print('mu_pos: ', mu_pos)
    print('sigma_pos: ', sigma_pos)

    norm_samples_pos = np.random.normal(mu_pos, sigma_pos, 1000)
    norm_samples_neg = np.random.normal(mu_neg, sigma_neg, 1000)

    # # plot the distribution of the normalized samples
    plt.hist(norm_samples_pos, bins=100, alpha=0.5, label='Positive')
    plt.hist(norm_samples_neg, bins=100, alpha=0.5, label='Negative')
    plt.legend(loc='upper right')
    plt.show()
    """

    # # calculate the mutual information between the samples and the labels
    mutual_info = MIC(train_X.fillna(0).values.reshape(-1, train_X.shape[1]), train_Y.values)
    # print('Mutual information:\n ', mutual_info)

    nunique = train_w_labels.nunique()
    print('Unique values per column:\n', nunique)

    dtypes = train_w_labels.dtypes
    print('Data types:\n', dtypes)

    dtale_view(train_w_labels, 'train_w_labels')

