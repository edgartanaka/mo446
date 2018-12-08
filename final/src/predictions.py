import os
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import normalize
from fit_predict_sk import fit_predict_sk, compute_kernel, fit_predict_reg

base_path = 'input/Data/'

# Load data
print('Loading data...')

target_features = scipy.io.loadmat(os.path.join(base_path, 'Image data/target_features.mat'))

print('Data LOADED!')

def get_object_counts_features():
    counts = target_features['Counts'].todense().T
    imgs_num = counts.shape[0]
    max_count = np.max(counts)
    steps = 20

    # marginalize

    # object_count_features = np.empty((imgs_num, steps - 1))
    # for i in range(imgs_num):
    #     object_counts_img = counts[i, :]
    #     hist, _ = np.histogram(object_counts_img, bins=bins)
    #     object_count_features[i, :] = hist

    # hist function in matlab defines the bins as bin centers
    # https://www.mathworks.com/help/matlab/ref/hist.html
    # therefore, here we need to try to mimic this behavior
    step_size = round(max_count / steps, 4)
    bin_centers = np.arange(0, max_count + 2 * step_size, step_size)
    bins_edges = bin_centers - step_size / 2

    # marginalize
    object_count_features = np.empty((imgs_num, len(bins_edges) - 1))
    for i in range(imgs_num):
        hist, _ = np.histogram(counts[i, :], bins=bins_edges)
        object_count_features[i, :] = hist

    # normalize
    return normalize(object_count_features, norm='l1')


def get_object_areas_features():
    areas = target_features['Areas'].todense()
    areas = areas.T
    imgs_num = areas.shape[0]
    max_count = np.max(areas)
    steps = 600

    # marginalize
    # X = np.empty((imgs_num, steps - 1))
    # for i in range(imgs_num):
    #     X[i, :], _ = np.histogram(areas[i, :], bins=bins)

    # hist function in matlab defines the bins as bin centers
    # https://www.mathworks.com/help/matlab/ref/hist.html
    # therefore, here we need to try to mimic this behavior
    step_size = round(max_count / steps, 4)
    bin_centers = np.arange(0, max_count + 2 * step_size, step_size)
    bins_edges = bin_centers - step_size / 2

    # marginalize
    object_areas_features = np.empty((imgs_num, len(bins_edges) - 1))
    for i in range(imgs_num):
        hist, _ = np.histogram(areas[i, :], bins=bins_edges)
        object_areas_features[i, :] = hist

    # normalize
    return normalize(object_areas_features, norm='l1')


def get_multiscale_object_areas():
    sptHistObjects = target_features['sptHistObjects']
    areas = target_features['Areas'].todense()
    num_objects = areas.shape[0];
    num_spt_hists = int(sptHistObjects.shape[1] / num_objects);
    imgs_num = sptHistObjects.shape[0]
    num_steps = 20;

    m_xs = None

    # num_spt_hists is equal to 5 (6800 / 1360 = 5)
    # each loop here is a quadrant (why 5 and not 4??)
    for j in range(0, num_spt_hists):
        x = sptHistObjects[:, (num_objects * j):(num_objects * (j + 1))]
        max_area = np.max(x)

        # hist function in matlab defines the bins as bin centers
        # https://www.mathworks.com/help/matlab/ref/hist.html
        # therefore, here we need to try to mimic this behavior
        step_size = round(max_area / num_steps, 4)

        bin_centers = np.arange(0, max_area + 2 * step_size, step_size)
        bins_edges = bin_centers - step_size / 2

        # marginalize
        m_x = np.empty((imgs_num, len(bins_edges) - 1))
        for i in range(imgs_num):
            m_x[i, :], _ = np.histogram(x[i, :], bins=bins_edges)

        if m_xs is None:
            m_xs = m_x
        else:
            m_xs = np.hstack((m_xs, m_x))

    # normalize
    X = m_xs  # this is (2400, 105)
    X = normalize(X, norm='l1')

    return X


def get_object_presences_features():
    X = target_features['Counts'].todense().T
    X = X > 0

    # normalize
    X = normalize(X, norm='l1')
    return X


def get_labeled_object_counts():
    X = target_features['Counts'].todense().T

    # normalize
    X = normalize(X, norm='l1')
    return X


def get_labeled_object_areas():
    X = target_features['Areas'].todense().T

    # normalize
    X = normalize(X, norm='l1')
    return X


def get_labeled_multiscale_object_areas_features():
    return target_features['sptHistObjects']


def get_scene_category_features():
    X = target_features['sceneCatFeatures']

    # normalize
    X = normalize(X, norm='l1')
    return X


def get_objects_scenes_features():
    X1 = target_features['sptHistObjects']
    X2 = target_features['sceneCatFeatures']
    X2 = normalize(X2, norm='l1')
    X = np.hstack((X1, X2))
    return X


def main():
    df = pd.DataFrame(columns=['description', 'top20', 'top100', 'bottom100', 'bottom20', 'spearman'])
    features_functions = [
                          get_object_counts_features,
                          get_object_areas_features,
                          get_multiscale_object_areas,
                          get_object_presences_features,
                          get_labeled_object_counts,
                          get_labeled_object_areas,
                          get_labeled_multiscale_object_areas_features,
                          get_scene_category_features,
                          get_objects_scenes_features
                          ]

    # features_functions = [get_objects_scenes_features]
    # regressors = ['svr', 'rf', 'gb', 'knn']
    # regressors = ['rf', 'gb', 'knn', 'ada', 'cat']
    regressors = ['svr']

    for f in features_functions:
        for r in regressors:
            print('-----------------------------------------')
            print('Running', r,  f.__name__)

            # get features
            X = f()

            if r == 'svr':
                _, top20, top100, bottom100, bottom20, spearman = fit_predict_sk(compute_kernel(X, X))
            else:
                _, top20, top100, bottom100, bottom20, spearman = fit_predict_reg(X, r)

            df = df.append({'description': r + ' - ' + f.__name__.replace('get_',''),
                            'top20': top20,
                            'top100': top100,
                            'bottom100': bottom100,
                            'bottom20': bottom20,
                            'spearman': spearman},
                           ignore_index=True)
            print('-----------------------------------------\n\n')

    df.to_csv('output/table1.csv')
    print('Results outputted to output/table1.csv')


if __name__ == "__main__":
    main()