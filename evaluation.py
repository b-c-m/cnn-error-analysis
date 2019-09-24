import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import NearestNeighbors
from .cifar10 import *

from PIL import Image


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    # added anfigure as returnable object for better function handling
    # changed size parameters for figure and colorbar

    :param y_true: true label per class; np.array([1,2,3,4,5,6,7,8,9,0,0,1,2])
    :param y_pred: predicted label per class; np.array([1,2,3,4,5,6,4,3,4,5,6,0,9])
    :param classes: list containing the class names
    :param normalize: boolean
    :param title:
    :param cmap:
    :return:
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(6,6))                     # change size
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, shrink = 0.6)               # fraction to match graph size better
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax




def plot_images(images_indices, x_logits_df, images, visualizer):
    """

    :param images_indices: indices of relevant images
    :param x_logits_df: the prediction and true_label columns are needed for confusion analysis
    :param images: np.array containing the image values
    :param visualizer: needed for colors
    :return: pillow image showing the relevant images with border in prediction-color
    """

    width = 360
    height = int((len(images_indices) / 10) + 1) * 36
    out_image = Image.new('RGB', (width, height), "black")

    x = 0
    y = 0
    for image_index in images_indices:

        color = visualizer.get_color_for_label(x_logits_df.loc[image_index]["prediction"])

        border = Image.new('RGB', (36, 36), color)
        img = Image.fromarray(np.uint8(images[image_index, :, :, :] * 255))
        border.paste(img, (2, 2))

        out_image.paste(border, (x, y))

        x += 36
        if x == 360:
            x = 0
            y += 36

    return out_image

################################################
#                     KPIs                     #


def nearest_neighbors(fv_df, neighbors=20):
    x_pca_nnb = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(fv_df.iloc[:, 1:1025])
    distances, indices = x_pca_nnb.kneighbors(fv_df.iloc[:, 1:1025])

    return distances, indices

def train_test_coverage(x_fv_df, nn_indices):
    """
    calcultes the average train-points in the 20 nearest neighbors for the test-set

    :param x_fv_df: containing the n-dimensional data to search for nearest neighbors
                    index < 9999 is test data
    :return:
    """


    # Choose only testset
    test_fv_df = x_fv_df.loc[x_fv_df.index < 10000]

    class_images_indices = x_fv_df.index.tolist()
    class_testimages_indices = test_fv_df.index.tolist()

    print("class_testimages_indices len " + str(len(class_testimages_indices)))

    train_neighbors = 0
    for test_image_nr in class_testimages_indices:

        nearest_neighbors = nn_indices[test_image_nr]

        for neighbor in nearest_neighbors:

            if neighbor in class_images_indices:
                if neighbor > 9999:  # is train?
                    train_neighbors += 1

    ttc = train_neighbors / len(class_testimages_indices)

    return ttc


## helper
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def confidence(logits_df):
    test_logits_df = logits_df.loc[logits_df.index < 10000]
    min_sm = []
    avg_confs = []
    softmaxes = []
    for j_image in range(0, len(test_logits_df)):

        if test_logits_df.iloc[j_image, 11] == test_logits_df.iloc[j_image, 12]:
            softmaxes.append(softmax(test_logits_df.iloc[j_image, 1:11].tolist()))

    df = pd.DataFrame(softmaxes)

    for i_class in range(0, 10):
        class_softmax_values = df.iloc[:, i_class]
        avg_confs.append(np.average(class_softmax_values))
        min_sm.append(min(class_softmax_values))

    return avg_confs, min_sm


def calculate_all_kpis(nn_matrix, fv_df, logits_df):
    """
    
    :param nn_matrix: externally computed nearest neighbor indices
    :param fv_df: whole dataset
    :param logits_df: whole dataset
    :return:
    """
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    condfidence_per_class = []
    confidence_own_class = []
    ttc = []
    min_sm_all = []

    for current_class in class_names:
        print("current class - >", current_class)

        # slice
        x_fv_df = slice_df_by_classes(fv_df, [current_class])
        x_logits_df = slice_df_by_classes(logits_df, [current_class])

        # confidence
        confs, min_sm = confidence(x_logits_df)
        condfidence_per_class.append(confs)
        confidence_own_class.append(confs[cifar_labelnr_from_labelname(current_class)])
        min_sm_all.append(min_sm)

        ttc.append(train_test_coverage(x_fv_df, nn_matrix))

    print("condfidence_per_class= ", condfidence_per_class)
    print("confidence_own_class= ", confidence_own_class)
    print("ttc= ", ttc)

    return condfidence_per_class, confidence_own_class, ttc, min_sm_all


##########################################################################
def plot_single_class_confusion_table(y_pred_cluster):
    """
    plot a confusion matrix for cluster off a single class
    #atm not used
    :param y_pred_cluster:
    :return:
    """

    cm = []
    for i in range(0, len(y_pred_cluster)):
        for j in range(0, len(y_pred_cluster[i])):
            if sum(y_pred_cluster[i][j]) > 0:
                cm.append(y_pred_cluster[i][j])

    cm = np.asarray(cm)
    print("cm =")
    print(cm)

    fig, ax = plt.subplots(figsize=(6, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.6)  # fraction to match graph size better

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
           yticklabels=range(0, len(y_pred_cluster)),
           title="Per Cluster Confusion",
           ylabel='Cluster',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

