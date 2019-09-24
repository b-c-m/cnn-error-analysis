import os
import numpy as np
import _pickle
import matplotlib.pyplot as plt

import pandas as pd

#TODO: change folder location for cifar10 files (python version)
DATA_PATH = "D:/tmp/cifar10_pythonversion/cifar-10-batches-py/"


"""
CifarLoader: from Learning TensorFlow by Tom Hope, Yehezkel S. Resheff, and Itay Lieder - 2018 -  P.62f
"""

class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x,y = self.images[self._i:self._i+batch_size],self.labels[self._i:self._i+batch_size]
        self._i = (self._i+batch_size) % len(self.images)                                # % len ... ?????
        return x,y


# needed in CifarLoader.load()
def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        print(fo.name)
        #encoding='binary') #encoding='latin1'
        dict = _pickle.load(fo, encoding='latin1')
    return dict


# needed in CifarLoader.load()
def one_hot(vec, vals = 10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]).load()    #bin
        self.test = CifarLoader(["test_batch"]).load()                                       #bin



#########################################
# Helper Stuff

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# CSV + DataFrame Manipulation
def load_csv_into_df(csv_url):
    """
    loads data from the csv-file into pandas dataframe
    :param csv_url:
    :return:
    """

    df = pd.read_csv(csv_url, sep=',', low_memory=False, error_bad_lines=False)  # dtype=dict, #nrows=10000

    return df


def slice_df_by_classes(df, classes):
    """
    select all rows that contain the corresponding classes
    :param df:
    :param classes: as a list
    :return:
    """
    df_temp = pd.DataFrame()

    for classtype in classes:
        # Slice only Rows with Classnames
        df_sliced = df.loc[df['image'].str.contains(classtype)]

        df_temp = pd.concat([df_temp, df_sliced], axis=0)

    print("df shape " + str(df_sliced.shape))

    return df_temp


#############################################################
#            Managing CIFAR10 Classes/Labels                #

def cifar_labelname_from_one_hot(b):
    """
    transforms one-hot-encoded vector to classname
    :param b:
    :return:
    """

    b = b.reshape(10)

    if b[0] == 1:
        return "airplane"
    if b[1] == 1:
        return "automobile"
    if b[2] == 1:
        return "bird"
    if b[3] == 1:
        return "cat"
    if b[4] == 1:
        return "deer"
    if b[5] == 1:
        return "dog"
    if b[6] == 1:
        return "frog"
    if b[7] == 1:
        return "horse"
    if b[8] == 1:
        return "ship"
    if b[9] == 1:
        return "truck"

def cifar_labelnr_from_labelname(b):

    if "airplane" in b:
        return 0
    if "automobile" in b:
        return 1
    if "bird" in b:
        return 2
    if "cat" in b:
        return 3
    if "deer" in b:
        return 4
    if "dog" in b:
        return 5
    if "frog" in b:
        return 6
    if "horse" in b:
        return 7
    if "ship" in b:
        return 8
    if "truck" in b:
        return 9


def convert_label_to_one_hot(labelstring):
    one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if "airplane" in labelstring:
        one_hot[0] = 1
    if "automobile" in labelstring:
        one_hot[1] = 1
    if "bird" in labelstring:
        one_hot[2] = 1
    if "cat" in labelstring:
        one_hot[3] = 1
    if "deer" in labelstring:
        one_hot[4] = 1
    if "dog" in labelstring:
        one_hot[5] = 1
    if "frog" in labelstring:
        one_hot[6] = 1
    if "horse" in labelstring:
        one_hot[7] = 1
    if "ship" in labelstring:
        one_hot[8] = 1
    if "truck" in labelstring:
        one_hot[9] = 1

    return one_hot


#####################################
# Dataframe manipulation for logits #
def add_predicted_classnr_column(df):
    """
    add a column with predicted class, only for logits_df
    :param df:
    :return:
    """

    predicted_classes_indexnr = []
    for i in range(len(df)):
        activations = df.iloc[i, 1:11]
        activations_as_float = [float(i) for i in activations]
        predicted_classes_indexnr.append(np.argmax(activations_as_float))

    df['prediction'] = predicted_classes_indexnr

    return df


def add_true_classnr_column(df):
    """
    add a column with the true class label, only for logits_df
    :param df:
    :return:
    """
    correct_classes_indexnr = []
    for i in range(len(df)):
        correct_label_nr = cifar_labelnr_from_labelname(df.iloc[i, 0])
        correct_classes_indexnr.append(correct_label_nr)

    df['true_label'] = correct_classes_indexnr

    return df

