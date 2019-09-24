import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import math

from .cifar10 import slice_df_by_classes

from PIL import Image, ImageDraw, ImageFont


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Visualizer:
    """
    Object matches color to class
    """

    def __init__(self, classnames, colors=None):

        self.classnames = classnames
        self.classes = list(range(0, len(self.classnames)))

        if colors is None:
            self.colors = ['#e6194B', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#911eb4', '#D3D3D3', '#f032e6',           #TODO: len(classnames) different colors + ensure len(classnames) = len(colors)
                 '#ffffff']
        else:
            self.colors = colors

    def get_color_legend(self):

        img = Image.new('RGB', (300, 400), color="black")
        d = ImageDraw.Draw(img)

        imgx = 10
        imgy = 10
        for classname, color in zip(self.classnames, self.colors):
            d.text((imgx, imgy), classname, fill=color, font=ImageFont.truetype("arial.ttf", 30))
            imgy += 30

        return img

    def get_color_for_label(self, labelnr):

        if labelnr <= len(self.colors) - 1:
            color = self.colors[labelnr]
            return color

        else:
            print("no matching color for label " + str(labelnr))

    def plot_pixel_2d(self, position_df, logits_df, mode="error", clusterlabel=None, class_nr = None, width = 2000, height = 2000, dotsize=4):
        """



        :param position_df: dim1 and dim2 values as pandas dataframe
        :param logits_df:
        :param mode:
        :return:
        """
        # bring positions in range [0,1]
        tx, ty = position_df.loc[:, 0], position_df.loc[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        border_gap = dotsize  # prevent new_img from clipping outside the main_image
        main_image = Image.new('RGB', (width, height))

        if mode =="clusterlabel":
            colormap = plt.get_cmap("tab20",len(np.unique(clusterlabel)))

        if mode == "confidence":
            colormap = plt.get_cmap('RdYlBu')

        for idx, x in enumerate(logits_df.values, 0):

            if mode == "clusterlabel":
                pixelcolor = matplotlib.colors.to_hex(colormap(clusterlabel[idx]))


            if mode =="output":

                sm_argsort = np.argsort(logits_df.iloc[idx, 1:11].tolist())
                sm_argsort = sm_argsort.tolist()
                sm_argsort.reverse()
                pixelcolor = self.get_color_for_label(sm_argsort[class_nr])

            if mode == "confidence":
                softmax_result = softmax(logits_df.iloc[idx, 1:11].tolist())
                pixelcolor = matplotlib.colors.to_hex(colormap(softmax_result[class_nr])) #(colormap(max(softmax(logits_df.iloc[idx, 1:11].tolist()))))

            if mode == "datasplit":

                if logits_df.index[idx] <= 9999:
                    pixelcolor = "#00bcff" #blue
                elif logits_df.index[idx] > 9999:
                    pixelcolor = "#ff6433" #orange

            if mode == "label":
                pixelcolor = self.get_color_for_label(logits_df.iloc[idx, 12])                       # 12 true label #TODO: grab the "true_label" column by name

            if mode == "prediction":
                pixelcolor = self.get_color_for_label((logits_df.iloc[idx, 11]))                     # 11 prediction #TODO same

            if mode == "error":
                if (x[11] == x[12]):
                    pixelcolor = "green"
                elif (x[11] != x[12]):
                    pixelcolor = "red"

            new_im = Image.new('RGB', (dotsize, dotsize), pixelcolor)

            main_image.paste(new_im, (int((width - border_gap) * tx[idx]), int((height - border_gap) * ty[idx])))

        # plt.imshow(full_image)
        # full_image.save('iv1_testset_test_correct_pixel.png')

        return main_image


    def plot_images_2d(self, position_df, logits_df, x_images, mode="prediction", width=3000, height=3000):
        """
        places the pictures of the chosen subset according to the 2d coordinates

        # modified from https://medium.com/@pslinge144/representation-learning-cifar-10-23b0d9833c40

        :param position_df: dataframe with x- and y-position in columns 0 and 1.
        :param logits_df: dataframe with prediction and true label in columns 11 and 12
        :param x_images: np.array with shape (?,32,32,3) containing the image data
        :return:
        """

        # bring positions in range [0,1]
        tx, ty = position_df.loc[:, 0], position_df.loc[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        max_dim = 50  # prevent new_img from clipping outside the main image
        full_image = Image.new('RGB', (width, height))

        for idx, x in enumerate(x_images, 0):
            tile = Image.fromarray(np.uint8(x * 255))

            border_color="lightgray"

            if mode == "prediction":
                border_color = self.get_color_for_label(logits_df.iloc[idx, 11])


            # draw border 1px
            old_size = tile.size
            new_size = (old_size[0] + 2, old_size[1] + 2)

            new_im = Image.new('RGB', new_size, border_color)
            new_im.paste(tile, (int((new_size[0] - old_size[0]) / 2), int((new_size[1] - old_size[1]) / 2)))
            #   plt.imshow(new_im)

            full_image.paste(tile, (int((width - max_dim) * tx[idx]), int((height - max_dim) * ty[idx])))

        # plt.imshow(full_image)
        # full_image.save('iv1_testset_truecolor.png')

        return full_image

    def plot_images_2d_borderless(self, tx, ty, x_images, width=2000, height=2000):
        """

        :param
        :param x_images:  only relevant images
        :param width:
        :param height:
        :return: """

        # tx = positions_df.loc[:,0]
        # ty = positions_df.loc[:,1]

        max_dim = 50  # prevent pasted images from clipping outside the main image
        main_image = Image.new('RGB', (width, height))

        for idx, x in enumerate(x_images, 0):
            tile = Image.fromarray(np.uint8(x * 255))
            main_image.paste(tile, (int((width - max_dim) * tx[idx]), int((height - max_dim) * ty[idx])))

        return main_image

    def plot_images_per_cluster(self, position_df,label,x_logits_df, images, savepath):

        cluster_list = []
        for cluster in range(len(np.unique(label))):
            newlist = []
            cluster_list.append(newlist)

        for a in range(0, len(label)):
            cluster_list[label[a]].append(a)
        # cluster_list = [[9, 10, 16, 21, 24][1,2,22,25,26]] values are 0-999 (or 0-5999) sorted

        relevant_indices = x_logits_df.index.tolist()

        # bring positions in range [0,1]
        tx, ty = position_df.loc[:, 0], position_df.loc[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        for cluster_nr in range(0, len(cluster_list)):

            cluster_indices = list(relevant_indices[i] for i in cluster_list[cluster_nr])
            x_images = images[cluster_indices, :, :, :]

            x_tx = list(tx[i] for i in cluster_list[cluster_nr])
            x_ty = list(ty[i] for i in cluster_list[cluster_nr])

            main_image = self.plot_images_2d_borderless(x_tx,x_ty,x_images)
            main_image.save(savepath + 'cluster'+str(cluster_nr)+'.png')

        return

####################################################################################
###  Plot the Nearest Images in for all False Negatives of a class

def plot_near_images(images_indices, x_logits_df, images, visualizer):
    """

    :param images_indices: indices of relevant images
    :param x_logits_df: the prediction and true_label columns are needed for confusion analysis
    :param images: np.array containing the image values
    :param visualizer: needed for colors
    :return: pillow image showing the relevant images with border in prediction-color
    """

    width = 396
    rows = 1
    if int(len(images_indices) / 10) > 1:
        rows = int(len(images_indices) / 10)
    height = rows * 46

    out_image = Image.new('RGB', (width, height), "black")

    x = 0
    y = 0
    for image_index in images_indices:

        # Border 1
        color = visualizer.get_color_for_label(x_logits_df.loc[image_index]["prediction"])
        border = Image.new('RGB', (36, 46), color)
        img = Image.fromarray(np.uint8(images[image_index, :, :, :] * 255))
        # draw conf?
        conf = np.max(softmax(x_logits_df.iloc[image_index, 1:11].tolist()))  #

        d = ImageDraw.Draw(border)
        d.text((10, 36), str(truncate(conf, 2)), fill="black", font=ImageFont.truetype("arial.ttf", 8))

        # Border 2
        color2 = visualizer.get_color_for_label(x_logits_df.loc[image_index]["true_label"])
        border2 = Image.new('RGB', (36, 34), color2)

        border.paste(border2, (0, 0))
        border.paste(img, (2, 2))
        out_image.paste(border, (x, y))

        x += 36  ##maybe space between images?!?! :)
        if x == 396:
            x = 0
            y += 46

    return out_image


def print_nn_one_image(image_nr, nn_indices, logits_df, images, visualizer):
    nearest_neighbors = nn_indices[image_nr]
    image_nn = plot_near_images(nearest_neighbors[0:11], logits_df, images, visualizer)

    return image_nn


def get_false_negatives_for_class(logits_df, current_class):
    class_logits_df = slice_df_by_classes(logits_df, [current_class])

    a = []
    for i in range(10):
        a.append([])

    df_indices = class_logits_df.index.values
    for i in range(len(class_logits_df)):
        a[class_logits_df.loc[df_indices[i]]["prediction"]].append(df_indices[i])

    false_negatives = a
    return false_negatives


def plot_nn_for_class(images_to_print, nn_indices, logits_df, images, visualizer):
    ten_image_rows = []
    for image_nr in images_to_print:
        img = print_nn_one_image(image_nr, nn_indices, logits_df, images, visualizer)
        ten_image_rows.append(img)

    final_class_img = Image.new('RGB', (396, (len(ten_image_rows) * 50)), "black")

    x = 0
    y = 0
    for i in range(len(ten_image_rows)):
        final_class_img.paste(ten_image_rows[i], (x, y))
        y += 50

    return final_class_img

#helper
def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n



######################################
##### visualize clustering results #####
def assign_cluster_manually(data, quantile_cutoff):
    """
    assign only quantile_cutoff for kmeans cluster
    """

    rows = len(data[:, 0])
    columns = len(data[0, :])
    print(rows)
    print(columns)

    data_label = []
    min_distance = []
    # data is 5000,12 nd array
    for row in range(rows):
        result = np.where(data[row, :] == np.amin(data[row, :]))
        data_label.append(result[0][0])
        min_distance.append(np.amin(data[row, :]))

    distances_per_cluster = []
    for i in range(columns):
        distances_per_cluster.append([])

    for cluster in range(columns):  # iterate over labels
        for index in range(rows):
            if data_label[index] == cluster:
                distances_per_cluster[cluster].append(min_distance[index])

    thresholds = []
    for i in range(columns):
        thresholds.append(np.quantile(distances_per_cluster[i], quantile_cutoff))

    labels = []
    for j in range(rows):
        if min_distance[j] < thresholds[data_label[j]]:
            labels.append(data_label[j])
        else:
            labels.append(columns)

    return labels


#not used in current analysis:
def show_cluster_with_color(x_tsne, x_pca, kmeans_object):
    # bring positions in range [0,1]
    tx, ty = x_tsne.loc[:, 0], x_tsne.loc[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    labels = kmeans_object.predict(np.array(x_pca))

    fig = plt.figure(figsize=(12, 12))
    plt.scatter(tx, 1 - ty, c=labels,
                s=5, cmap='tab20c');

    # print(labels)
    # print(labels.shape)
    return fig


def plot_decision_boundaries(kmeans_obj, df):
    """

    predicts a cluster for each pixel in the visualization and colors it accordingly
    resulting image shows pixel-precise decision boundaries
    only works for partitions in 2d space !

    copied from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
    small modification, flipped all y-values to match the Image coordinates
    """
    data = np.array(df)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Plot the decision boundary. For that, we will assign a color to each point
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    pointlabels = kmeans_obj.predict(np.c_[xx.ravel(), -yy.ravel()])

    # Put the result into a color plot
    Z = pointlabels.reshape(xx.shape)

    figure1 = plt.figure()
    #plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(data[:, 0], - data[:, 1], 'k.', markersize=2)

    # Plot the centroids as a white X
    centroids = kmeans_obj.cluster_centers_
    plt.scatter(centroids[:, 0], - centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    plt.title('K-means clustering \n'
              'Centroids are marked with white cross')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    return figure1


