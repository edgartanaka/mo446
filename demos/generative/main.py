import matplotlib
matplotlib.use('agg')
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os
import gzip
import argparse
import numpy as np


def load_mnist(path, kind='t10k'):
    """
    Loads MNIST fashion dataset

    Reference https://github.com/zalandoresearch/fashion-mnist
    :param path:
    :param kind:
    :return:
    """
    """Load MNIST data from `path`"""
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16)
        images_count = int(images.shape[0]/784)
        images = images.reshape(images_count, 784)

    return images


def plot_generated_data(data, width, output_file):
    """
    Plots 100 square images in a matplotlib figure

    This code was based on https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.htmls
    :param data: numpy array with dimensions (100, N) where N is the number of pixels in the image
    :param width: width of each square image
    :param output_file: file path of where the matplotlib figure will be saved
    :return: None
    """
    fig, ax = plt.subplots(10, 10, figsize=(width, width), subplot_kw=dict(xticks=[], yticks=[]))

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(width, width), cmap='binary')
        im.set_clim(0, 16)

    fig.savefig(output_file)
    plt.close()
    print('Saved plot to ' + output_file)


def run_pipeline(data, sub_sample_size, image_width, n_components, var, prefix_file, output_path):
    """
    Runs the pipeline:
    - plot sample of original data
    - reduce dimensionality with PCA
    - subsample data
    - fit GMM
    - generate data
    - plot generated data

    This code was based on https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
    :param data: the data itself
    :param sub_sample_size: size of subsampling, use -1 for not subsampling
    :param image_width: width of the images
    :param n_components: number of GMM components
    :param var: variance to keep after PCA dimensionality reduction
    :param prefix_file: string that indicates the prefix of files to be outputted
    :return:
    """
    # original data: subsample 100 images and plot them
    data = np.asarray(data)
    original_sample = data[:100, :]
    plot_generated_data(original_sample, image_width, os.path.join(output_path, prefix_file + '_original_data.png'))
    print('Original space - number of dimensions:' + str(original_sample.shape[1]))

    # reduce dimensionality
    pca = PCA(var, whiten=True)
    data = pca.fit_transform(data)
    print('Reduced space - number of dimensions:' + str(data.shape[1]))

    # sub sample data
    np.random.shuffle(data)
    if sub_sample_size > 0:
        data = data[0:sub_sample_size, :]

    # fit data
    # this is similar to learning the gaussian distributions or learning how data is generated
    gmm = GaussianMixture(n_components, covariance_type='full', random_state=0)
    gmm.fit(data)

    # generate a sample of 100 synthetic data points
    X_sample, _ = gmm.sample(100)

    # transform from reduced dimension space back to original space
    generated_data = pca.inverse_transform(X_sample)

    # plot synthetic data
    plot_generated_data(generated_data, image_width, os.path.join(output_path, prefix_file + '_generated_data.png'))


def run_fashion_example(input_path, output_path):
    print('---------------------------------------------------------')
    print('Running pipeline for Fashion MNIST dataset...')
    X_test = load_mnist(input_path, kind='t10k')
    data = X_test.data
    run_pipeline(data=data,
                 sub_sample_size=5000,
                 n_components=100,
                 image_width=28,
                 var=0.96,
                 prefix_file='fashion',
                 output_path=output_path)
    print('FINISHED')
    print('---------------------------------------------------------')


def run_digits_example(output_path):
    print('---------------------------------------------------------')
    print('Running pipeline for MNIST dataset...')
    data = load_digits().data
    run_pipeline(data=data,
                 sub_sample_size=-1,
                 n_components=110,
                 image_width=8,
                 var=0.99,
                 prefix_file='digits',
                 output_path=output_path)
    print('FINISHED')
    print('---------------------------------------------------------')


def main():
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

    input_path = "{}/{}".format(args.assets_folder, "imgs")
    output_path = args.output_folder

    run_fashion_example(input_path, output_path)
    run_digits_example(output_path)


if __name__ == '__main__':
    main()
