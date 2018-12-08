import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import cv2
from sklearn.cluster import KMeans
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.preprocessing import StandardScaler
import time
from sklearn.decomposition import PCA

NUM_CLUSTERS = 30
LABELS = ['airplanes', 'Faces_easy', 'Motorbikes', 'Leopards']


def cluster(feature_vectors):
    """
    Runs K-Means clustering
    :param feature_vectors:
    :return: kmeans fitted object
    """
    print('Running KMeans...')
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(feature_vectors)
    return kmeans


def split_train_test(path):
    dataset = np.empty((0, 2))

    for idx, label in enumerate(LABELS):
        label_path = join(path, label)
        files = [join(label_path, f) for f in listdir(label_path) if isfile(join(label_path, f))]

        dataset_files = np.array(files)
        dataset_classes = np.repeat(idx, dataset_files.shape[0])
        dataset_files = np.stack((dataset_classes, dataset_files), axis=-1)

        # concatenate to dataset all samples of this label
        dataset = np.concatenate((dataset, dataset_files))

    X = dataset[:, 1]
    y = dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    return X_train, X_test, y_train, y_test


def extract_feature(img):
    """
    Extract features with SIFT.
    SIFT detects several keypoints per image.
    For each keypoint detected, a 128-D feature vector is generated.
    Therefore, this function returns a list of 128-D feature vectors.
    :param img: numpy array for image loaded via cv2.imread()
    :return: list of 128-D feature vectors.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)

    return descs


def generate_bag_of_words(img, kmeans):
    """
    Generates the bag of visual words.
    First, we extract a number of feature vectors for this image. (one for each keypoint in the image)
    Then, we find (predict) to which cluster in the fitted kmeans each feature vector will belong.
    Lastly, we count how many feature vectors have fallen into each cluster (histogram).
    We return this histogram.

    :param img: numpy array for image loaded via cv2.imread()
    :param kmeans: kmeans object already fitted
    :return: histogram with the count of how many feature vectors fell into each cluster
    """
    feature_vectors = extract_feature(img)
    labels = kmeans.predict(feature_vectors)
    bag = np.bincount(labels, minlength=NUM_CLUSTERS)

    return bag


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Source for this method: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_kmeans_pca(kmeans, X):
    """
    Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html
    Plot each feature vector in a 2D space and color codes it by the cluster to which it was assigned.
    :return:
    """
    y_pred = kmeans.predict(X)
    X_reduced = PCA(n_components=2).fit_transform(X)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred)
    plt.title("K-means")


def plot_svm(X, y_pred):
    """
    Source: https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
    Plots each data sample and color codes it by class prediction in a 2D space.
    We use PCA to reduce dimensionality.
    :param X:
    :return:
    """
    X_reduced = PCA(n_components=2).fit_transform(X)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred)
    plt.title('SVM')


def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-a',
                        '--assets_folder',
                        required=True,
                        help='The absolute path to assets folder')

    parser.add_argument('-o',
                        '--output_folder',
                        required=True,
                        help='The absolute path to outputs folder')

    args = parser.parse_args()

    # set input and output paths
    input_path = "{}/{}".format(args.assets_folder, "imgs/caltech101")
    output_path = args.output_folder

    # split into train and test datasets
    X_train, X_test, y_train, y_test = split_train_test(input_path)

    # extract SIFT feature vectors for all images and concatenate them all into all_feature_vectors
    all_feature_vectors = np.empty((0, 128))
    for f in X_train:
        img = cv2.imread(f)
        feature_vectors = extract_feature(img)
        all_feature_vectors = np.concatenate((all_feature_vectors, feature_vectors), axis=0)

    # clusterize using kmeans
    kmeans = cluster(all_feature_vectors)

    # generate the bag of visual words for each image and concatenate them all it into train_bags
    train_bags = np.empty((0, NUM_CLUSTERS))
    for f in X_train:
        img = cv2.imread(f)
        bag = generate_bag_of_words(img, kmeans)
        train_bags = np.vstack((train_bags, bag))

    # normalize all bags
    train_bags = StandardScaler().fit_transform(train_bags)

    # train SVM
    clf = SVC(gamma='auto')
    clf.fit(train_bags, y_train)

    # generate bag of visual words for images in the test dataset
    test_bags = np.empty((0, NUM_CLUSTERS))
    for f in X_test:
        img = cv2.imread(f)
        bag = generate_bag_of_words(img, kmeans)
        test_bags = np.vstack((test_bags, bag))

    # normalize all bags
    test_bags = StandardScaler().fit_transform(test_bags)

    # test accuracy
    score = clf.score(test_bags, y_test)
    print('Overall accuracy (test dataset):', score)
    y_pred = clf.predict(test_bags)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=LABELS,
                          title='Confusion matrix, without normalization')
    plt.savefig(os.path.join(output_path, 'non_normalized_confusion_matrix.png'))

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=LABELS, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(os.path.join(output_path, 'normalized_confusion_matrix.png'))

    # Plot K-means in training dataset
    plt.figure()
    plot_kmeans_pca(kmeans, all_feature_vectors)
    plt.savefig(os.path.join(output_path, 'kmeans.png'))

    # Plot each image and its prediction
    plt.figure()
    plot_svm(test_bags, y_pred)
    plt.savefig(os.path.join(output_path, 'svm_predicted.png'))

    # Plot each image and its ground truth
    plt.figure()
    plot_svm(test_bags, y_test)
    plt.savefig(os.path.join(output_path, 'svm_ground_truth.png'))

    # Calculate elapsed time
    end = time.time()
    elapsed = end - start
    print('Elapsed time (seconds):', elapsed)


if __name__ == '__main__':
    main()
