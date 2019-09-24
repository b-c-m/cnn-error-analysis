from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


### PCA ###
def standardize(df):
    x = df.iloc[:, 1:1025].values                   #still hardcoded for 1024 feature vectors
    x_standardize = StandardScaler().fit_transform(x)

    return x_standardize


def pca(df, pcnr):
    x = standardize(df)

    pca = PCA(n_components=pcnr)
    x_pcs= pca.fit_transform(x)
    pca_columns = np.arange(pcnr)
    pca_df = pd.DataFrame(data=x_pcs, columns=pca_columns)

    # Explained Variance
    print("Explained Variance:")
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))

    # Plot the explained variance
    fig, ax = plt.subplots()
    ax.plot(list(range(1, len(pca.explained_variance_ratio_)+1)), np.cumsum(pca.explained_variance_ratio_))
    ax.set(xlabel='PCAs', ylabel=' cumulated explained variance')
    ax.grid()

    return pca_df, fig


def pca_with_target_variance(df, variance):
    x = standardize(df)

    pca = PCA(variance)
    x_pcs = pca.fit_transform(x)
    pca_df = pd.DataFrame(data=x_pcs)

    # Explained Variance
    print("Explained Variance:")
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))

    # Plot the explained variance
    fig, ax = plt.subplots()
    ax.plot(list(range(1, len(pca.explained_variance_ratio_)+1)), np.cumsum(pca.explained_variance_ratio_))
    ax.set(xlabel='PCAs', ylabel=' cumulated explained variance')
    ax.grid()

    return pca_df, fig


### TSNE ###
#RS = 123
def tsne(x, n_components):
    time_start = time.time()
    x_tsne = TSNE(random_state=RS, n_components=n_components).fit_transform(x)

    tsne_columns = np.arange(n_components)
    x_tsne_df = pd.DataFrame(data=x_tsne, columns=tsne_columns)

    print('t-sne done. time elapsed: {} seconds'.format(time.time() - time_start))
    return x_tsne_df


### K-Means ###
def kmeans(x, n_clusters=10):
    """
    docu https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    :param x: dataframe
    :param n_clusters:
    :return:
    """
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters)
    kmeans.fit(np.array(x))
    return kmeans


def kmeans_eval(x, upper_k = 10):

    """
    run kmeans() k-1 times and evaluate the partitions for different K
    :param x: input as dataframe
    :param upper_k: max k for kmeans partitions to compare
    :return: the graph to plot the inertias, partitions
    """
    time_start = time.time()

    partitions = []
    inertia = []
    for k in range(2, upper_k+1):

        a = kmeans(x, n_clusters=k)
        partitions.append(a)
        inertia.append(a.inertia_)

    k_values = list(range(2, upper_k+1))

    fig, ax = plt.subplots()
    ax.plot(k_values, inertia)
    ax.set(xlabel='k-value', ylabel='inertia')
    ax.grid()

    print('kmeans done ' + str(len(k_values)) +' times . time elapsed: {} seconds'.format(time.time() - time_start))

    return fig, partitions

