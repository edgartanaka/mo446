""""
University of Campinas

MO446 - Computer Vision (Prof. Adin)
Final Project - Image Memorability

Authors:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Computation of image memorability values using global features.
"""

from fit_predict_sk import compute_kernel, run_grid_sk, try_params_sk
from scipy.stats.stats import spearmanr
from scipy.io import loadmat
from sklearn.svm import SVR
from tqdm import tqdm
import numpy as np
import os

data_path = "input/Data/"

target_features = loadmat(os.path.join("output", "target_features.mat"))

# Features of each descriptor
pixel_features = target_features["pixel_histograms"]
gist_features  = target_features["gist"]
sift_features  = target_features["sptHistsift"]
surf_features  = target_features["sptHistsurf"]
orb_features   = target_features["sptHistorb"]
brisk_features = target_features["sptHistbrisk"]
#ssim_features  = target_features["sptHistssim"]
#hog_features   = target_features["sptHisthog"]

# Splits
mat = loadmat(os.path.join(data_path, "Random splits/image_train_indices.mat"))
image_train_indices = mat["image_train_indices"]

mat = loadmat(os.path.join(data_path, "Random splits/image_test_indices.mat"))
image_test_indices = mat["image_test_indices"]

mat = loadmat(os.path.join(data_path, "Random splits/subject_hrs1.mat"))
subject_hrs1 = mat["subject_hrs1"]

mat = loadmat(os.path.join(data_path, "Random splits/subject_hrs2.mat"))
subject_hrs2 = mat["subject_hrs2"]

# Indices for grid search
mat = loadmat(os.path.join(data_path, "Random splits/T1_train_indices.mat"))
T1_train_indices = mat["T1_train_indices"]

mat = loadmat(os.path.join(data_path, "Random splits/T2_train_indices.mat"))
T2_train_indices = mat["T2_train_indices"]

mat = loadmat(os.path.join(data_path, "Random splits/T1_train_subject_hrs.mat"))
T1_train_subject_hrs = mat["T1_train_subject_hrs"]

mat = loadmat(os.path.join(data_path, "Random splits/T2_train_subject_hrs.mat"))
T2_train_subject_hrs = mat["T2_train_subject_hrs"]


def main():    
    with open(os.path.join("output", "target_features_results.txt"), "w") \
                                                                    as output:
        print("\n=== Pixels histogram ===")
        output.write("=== Pixels histogram ===\n")
        pixels = pixel_features.reshape(pixel_features.shape[0],
                 pixel_features.shape[1] * pixel_features.shape[2])
        X_p = pixels / np.sum(pixels[0])
        params = {"type": "histintersection"}
        X_kernel_p = compute_kernel(X_p, X_p, params)
        rtr, top20, top100, bottom100, bottom20, spearman = \
                                            fit_predict_sk(X_kernel_p, params)
        output.write("top 20: {0}\n".format(top20))
        output.write("top 100: {0}\n".format(top100))
        output.write("bottom 100: {0}\n".format(bottom100))
        output.write("bottom 20: {0}\n".format(bottom20))
        output.write("corr avg: {0}\n\n".format(spearman))

        print("\n=== GIST ===")
        output.write("=== GIST ===\n")
        X_g = gist_features
        params = {"type": "rbf", "sigma": 0.5}
        X_kernel_g = compute_kernel(X_g, X_g, params)
        rtr, top20, top100, bottom100, bottom20, spearman = \
                                            fit_predict_sk(X_kernel_g, params)
        output.write("top 20: {0}\n".format(top20))
        output.write("top 100: {0}\n".format(top100))
        output.write("bottom 100: {0}\n".format(bottom100))
        output.write("bottom 20: {0}\n".format(bottom20))
        output.write("corr avg: {0}\n\n".format(spearman))

        print("\n=== SIFT ===")
        output.write("=== SIFT ===\n")
        X_s = sift_features
        params = {"type": "histintersection"}
        X_kernel_s = compute_kernel(X_s, X_s, params)
        rtr, top20, top100, bottom100, bottom20, spearman = \
                                            fit_predict_sk(X_kernel_s, params)
        output.write("top 20: {0}\n".format(top20))
        output.write("top 100: {0}\n".format(top100))
        output.write("bottom 100: {0}\n".format(bottom100))
        output.write("bottom 20: {0}\n".format(bottom20))
        output.write("corr avg: {0}\n\n".format(spearman))

        print("\n=== SURF ===")
        output.write("=== SURF ===\n")
        X_u = surf_features
        params = {"type": "histintersection"}
        X_kernel_u = compute_kernel(X_u, X_u, params)
        rtr, top20, top100, bottom100, bottom20, spearman = \
                                            fit_predict_sk(X_kernel_u, params)
        output.write("top 20: {0}\n".format(top20))
        output.write("top 100: {0}\n".format(top100))
        output.write("bottom 100: {0}\n".format(bottom100))
        output.write("bottom 20: {0}\n".format(bottom20))
        output.write("corr avg: {0}\n\n".format(spearman))

        print("\n=== ORB ===")
        output.write("=== ORB ===\n")
        X_o = orb_features
        params = {"type": "histintersection"}
        X_kernel_o = compute_kernel(X_o, X_o, params)
        rtr, top20, top100, bottom100, bottom20, spearman = \
                                            fit_predict_sk(X_kernel_o, params)
        output.write("top 20: {0}\n".format(top20))
        output.write("top 100: {0}\n".format(top100))
        output.write("bottom 100: {0}\n".format(bottom100))
        output.write("bottom 20: {0}\n".format(bottom20))
        output.write("corr avg: {0}\n\n".format(spearman))

        print("\n=== BRISK ===")
        output.write("=== BRISK ===\n")
        X_b = brisk_features
        params = {"type": "histintersection"}
        X_kernel_b = compute_kernel(X_b, X_b, params)
        rtr, top20, top100, bottom100, bottom20, spearman = \
                                            fit_predict_sk(X_kernel_b, params)
        output.write("top 20: {0}\n".format(top20))
        output.write("top 100: {0}\n".format(top100))
        output.write("bottom 100: {0}\n".format(bottom100))
        output.write("bottom 20: {0}\n".format(bottom20))
        output.write("corr avg: {0}\n\n".format(spearman))

#        print("\n=== SSIM ===")
#        output.write("=== SSIM ===\n")
#        X_i = ssim_features
#        params = {"type": "histintersection"}
#        X_kernel_i = compute_kernel(X_i, X_i, params)
#        rtr, top20, top100, bottom100, bottom20, spearman = \
#                                            fit_predict_sk(X_kernel_i, params)
#        output.write("top 20: {0}\n".format(top20))
#        output.write("top 100: {0}\n".format(top100))
#        output.write("bottom 100: {0}\n".format(bottom100))
#        output.write("bottom 20: {0}\n".format(bottom20))
#        output.write("corr avg: {0}\n\n".format(spearman))
#
#        print("\n=== HOG ===")
#        output.write("=== HOG ===\n")
#        X_h = hog_features
#        params = {"type": "histintersection"}
#        X_kernel_h = compute_kernel(X_h, X_h, params)
#        rtr, top20, top100, bottom100, bottom20, spearman = \
#                                            fit_predict_sk(X_kernel_h, params)
#        output.write("top 20: {0}\n".format(top20))
#        output.write("top 100: {0}\n".format(top100))
#        output.write("bottom 100: {0}\n".format(bottom100))
#        output.write("bottom 20: {0}\n".format(bottom20))
#        output.write("corr avg: {0}\n\n".format(spearman))

        print("\n=== All global features ===")
        output.write("=== All global features ===\n")
        params = {"type": "histintersection"}
        X_kernel_all = np.ones(X_kernel_p.shape)
        X_kernel_all *= X_kernel_p
        X_kernel_all *= X_kernel_g
        X_kernel_all *= X_kernel_s
        X_kernel_all *= X_kernel_u
        X_kernel_all *= X_kernel_o
        X_kernel_all *= X_kernel_b
#        X_kernel_all *= X_kernel_i
#        X_kernel_all *= X_kernel_h
        rtr, top20, top100, bottom100, bottom20, spearman = \
                                                fit_predict_sk(X_kernel_all, params)
        output.write("top 20: {0}\n".format(top20))
        output.write("top 100: {0}\n".format(top100))
        output.write("bottom 100: {0}\n".format(bottom100))
        output.write("bottom 20: {0}\n".format(bottom20))
        output.write("corr avg: {0}\n\n".format(spearman))


def fit_predict_sk(features_kernel: np.ndarray, params: dict):
    # features_kernel must be (2400, 2400)
    splits = 25

    # train and test for real now
    top_20_scores = []
    top_100_scores = []
    bottom_100_scores = []
    bottom_20 = []
    corrs = []
    
    rtr = {}
    max_rho = -float("inf")
    
    for i in tqdm(range(splits)):
        best_r, best_c, best_p = run_grid_sk(features_kernel, i)
        
        train_idx = image_train_indices[i, :] - 1
        test_idx = image_test_indices[i, :] - 1
        X_train = features_kernel[train_idx[:, None], train_idx] 
        X_test = features_kernel[test_idx[:, None], train_idx]  
        y_train = subject_hrs1[i, train_idx]
        y_test = subject_hrs2[i, test_idx]

        # epsilon SVR with histogram intersection kernel 
        svr = SVR(kernel='precomputed', C=best_c, epsilon=best_p)
        predicted_memo_scores = svr.fit(X_train, y_train).predict(X_test)
        
        # get indices of sorted memo scores
        idx = np.argsort(predicted_memo_scores)

        empirical_memo_scores = y_test[idx]

        rho, _ = spearmanr(predicted_memo_scores, y_test)
        
        if rho > max_rho:
            rtr['param'] = params
            rtr['model'] = svr
            rtr['test_features'] = X_test
            rtr['predicted'] = predicted_memo_scores
            rtr['test_label'] = y_test
            rtr['train_indices'] = train_idx
            rtr['test_indices'] = test_idx
            rtr['C'] = best_c
            rtr['e'] = best_p
        
        
        corrs.append(rho)
        top_20_scores.append(empirical_memo_scores[-20:])
        top_100_scores.append(empirical_memo_scores[-100:])
        bottom_100_scores.append(empirical_memo_scores[0:100])
        bottom_20.append(empirical_memo_scores[0:20])
        
    top20 = np.sum(top_20_scores) / (25 * 20)
    top100 = np.sum(top_100_scores) / (25 * 100)
    bottom100 = np.sum(bottom_100_scores) / (25 * 100)
    bottom20 = np.sum(bottom_20) / (25 * 20)
    spearman = np.mean(corrs)

    print('top 20:', top20)
    print('top 100:', top100)
    print('bottom 100:', bottom100)
    print('bottom 20:', bottom20)
    print('corr avg:', spearman)

    return rtr, top20, top100, bottom100, bottom20, spearman


if __name__ == "__main__":
    main()
