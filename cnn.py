import numpy as np
import os

import tensorflow as tf
import csv
import cifar10 as c10 # my own cifar10 module

from datasets import cifar10 # from tensorflow slim.datasets

from nets import inception
from preprocessing import inception_preprocessing
from tensorflow.contrib import slim


def load_batch(dataset, batch_size=32, height=224, width=224, is_training=False):
    """Loads a single batch of data.
    from https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)

    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
        [image, image_raw, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)

    return images, images_raw, labels


def get_init_fn(checkpoints_dir, model_starting_ckpt):
    checkpoint_exclude_scopes = ["InceptionV1/Logits", "InceptionV1/AuxLogits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, model_starting_ckpt),
        variables_to_restore)


def train_model(cifar10_data_dir, train_dir, checkpoints_dir, model_starting_ckpt, lr=0.0001,steps=200000, lower_lr_every_x_steps=50000):
    """
    train an inception model on the cifar10 dataset

    :param cifar10_data_dir:
    :param train_dir:
    :param checkpoints_dir:
    :param model_starting_ckpt:
    :param lr:
    :param steps:
    :param lower_lr_every_x_steps:
    :return:
    """

    image_size = inception.inception_v1.default_image_size #taken from tfslim.inception-def, but is 224

    loops = int(steps / lower_lr_every_x_steps)

    for i in range(loops):

        step_target = (i+1) * lower_lr_every_x_steps

        print("learning rate is " + str(lr) + " and step_target is " + str(step_target))

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)

            dataset = cifar10.get_split('train', cifar10_data_dir)
            images, _, labels = load_batch(dataset, height=image_size, width=image_size)

            val_dataset = cifar10.get_split('test', cifar10_data_dir)
            val_images, _, val_labels = load_batch(val_dataset, height=image_size, width=image_size)

            # Create the model, use the default arg scope to configure the batch norm parameters.
            with slim.arg_scope(inception.inception_v1_arg_scope()):
                logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)
                val_logits, _ = inception.inception_v1(val_images, num_classes=dataset.num_classes, is_training=False,
                                                       reuse=True)

            # Specify the loss function:
            one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
            slim.losses.softmax_cross_entropy(logits, one_hot_labels)
            total_loss = slim.losses.get_total_loss()

            # Specify the `validation` loss function:
            val_one_hot_labels = slim.one_hot_encoding(val_labels, dataset.num_classes)
            val_loss = tf.losses.softmax_cross_entropy(val_one_hot_labels, val_logits,
                                                       loss_collection="validation")

            # Create some summaries to visualize the training process:
            tf.summary.scalar('losses/Total Loss', total_loss)
            tf.summary.scalar('validation/Validation_Loss', val_loss)

            # Specify the optimizer and create the train op:
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_op = slim.learning.create_train_op(total_loss, optimizer)

            my_saver = tf.train.Saver(max_to_keep=50)

            # Run the training:
            final_loss = slim.learning.train(
                train_op,
                logdir=train_dir,
                init_fn=get_init_fn(checkpoints_dir, model_starting_ckpt),
                number_of_steps=step_target,
                saver=my_saver,
                save_summaries_secs=60,
                save_interval_secs=1200)

        print('Finished training. Last batch loss %f' % final_loss)

        lr = lr / 10
        step_target += lower_lr_every_x_steps


def extract_features(checkpoints_dir, model_ckpt, slim_model_name, cifar_data_manager):
    """
    extract features of defined tensors for each cifar10 image
    appends features to corresponding csv-file

    :param checkpoints_dir:
    :param model_ckpt:
    :param slim_model_name:
    :param cifar_data_manager:
    :return:
    """

    image_size = inception.inception_v1.default_image_size
    print(image_size)

    with tf.Graph().as_default():

        image = tf.placeholder(tf.float32, shape=[32, 32, 3], name="x")

        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(processed_images, num_classes=10, is_training=False)

        probabilities = tf.nn.softmax(logits)

        # Load Images
        x_test = cifar_data_manager.test.images.reshape(10, 1000, 32, 32, 3)
        y_test = cifar_data_manager.test.labels.reshape(10, 1000, 10)

        x_train = cifar_data_manager.train.images.reshape(50, 1000, 32, 32, 3)
        y_train = cifar_data_manager.train.labels.reshape(50, 1000, 10)

        X = np.concatenate((x_test, x_train), axis=0)
        Y = np.concatenate((y_test, y_train), axis=0)

        ''' <Tensors to Analyse> '''
        feature_vector = tf.get_default_graph().get_tensor_by_name('InceptionV1/Logits/AvgPool_0a_7x7/AvgPool:0')
        logits_vector = tf.get_default_graph().get_tensor_by_name('InceptionV1/Logits/Predictions/Reshape:0')
        '''</ Tensors to Analyse>'''

        # define function to load model weights
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, model_ckpt),
            slim.get_model_variables(slim_model_name))

        with tf.Session() as sess:
            init_fn(sess)

            '''< Prepare csv header >'''
            # prepare csv header for extracted feature vector
            fieldnames_fv = ["image"]
            for i in range(0, 1024):
                fieldnames_fv.append("n" + str(i))

            fieldnames_logits = ["image"]
            for i in range(0, 10):
                fieldnames_logits.append("out" + str(i))
            '''</ Prepare csv header >'''

            ''' <Run Inference and save to file.csv> '''
            # create+open csv-files
            with open('InceptionV1_Cifar_fv_.csv', mode='a', newline="", encoding="utf-8") as fv_csv, open(
                    'InceptionV1_Cifar_logits_.csv', mode='a', newline="", encoding="utf-8") as logits_csv:

                # feature vector Header
                writer_1 = csv.DictWriter(fv_csv, delimiter=',', fieldnames=fieldnames_fv)
                writer_1.writeheader()

                # logits vector header
                writer_2 = csv.DictWriter(logits_csv, delimiter=',', fieldnames=fieldnames_logits)
                writer_2.writeheader()

                """ iterate over batches"""
                for batch in range(60):
                    """iterate over images"""
                    for image_nr in range(1000):

                        image_x = (X[batch, image_nr]).reshape(1, 32, 32, 3)
                        label_y = (Y[batch, image_nr]).reshape(1, 10)

                        # print("image input shape: " + str(image_x.shape))
                        feed = {image: image_x[0, :, :, :]}

                        probabilities_np, feature_vec, logits_vec = sess.run(
                            [probabilities, feature_vector, logits_vector], feed_dict=feed)

                        # convert one-hot encoded Y to name:
                        image_label = c10.cifar_labelname_from_one_hot(label_y)
                        image_name = image_label + str(batch * 1000 + image_nr)

                        # save extracted features for image
                        fv_dict = {'image': image_name}
                        for i in range(0, 1024):
                            fv_dict[fieldnames_fv[i + 1]] = feature_vec.reshape([1024])[i]
                        writer_1.writerow(fv_dict)

                        # save logits for image
                        logits_dict = {'image': image_name}
                        for i in range(0, 10):
                            logits_dict[fieldnames_logits[i + 1]] = logits_vec.reshape([10])[i]
                        writer_2.writerow(logits_dict)

