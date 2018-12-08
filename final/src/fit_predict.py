# import numpy as np
# from scipy.stats.stats import spearmanr
# from sklearn.svm import SVR
# from tqdm import tqdm
# import scipy, os
# import cv2
#
# base_path = '../matlab/data/'
# mat = scipy.io.loadmat(os.path.join(base_path, 'Random splits/image_test_indices.mat'))
# image_test_indices = mat['image_test_indices']
#
# mat = scipy.io.loadmat(os.path.join(base_path, 'Random splits/image_train_indices.mat'))
# image_train_indices = mat['image_train_indices']
#
# mat = scipy.io.loadmat(os.path.join(base_path, 'Random splits/subject_hrs1.mat'))
# subject_hrs1 = mat['subject_hrs1']
#
# mat = scipy.io.loadmat(os.path.join(base_path, 'Random splits/subject_hrs2.mat'))
# subject_hrs2 = mat['subject_hrs2']
#
# # for grid search
# mat = scipy.io.loadmat(os.path.join(base_path, 'Random splits/T1_train_indices.mat'))
# T1_train_indices = mat['T1_train_indices']
#
# mat = scipy.io.loadmat(os.path.join(base_path, 'Random splits/T2_train_indices.mat'))
# T2_train_indices = mat['T2_train_indices']
#
# mat = scipy.io.loadmat(os.path.join(base_path, 'Random splits/T1_train_subject_hrs.mat'))
# T1_train_subject_hrs = mat['T1_train_subject_hrs']
#
# mat = scipy.io.loadmat(os.path.join(base_path, 'Random splits/T2_train_subject_hrs.mat'))
# T2_train_subject_hrs = mat['T2_train_subject_hrs']
#
#
# def compute_kernel(data_1, data_2):
#     """
#     Implementation of Histogram Intersection Kernel
#     :param data_1:
#     :param data_2:
#     :return:
#     """
#
#     print('Computing kernel.... This may take a while')
#
#     if np.any(data_1 < 0) or np.any(data_2 < 0):
#         warnings.warn('Min kernel requires data to be strictly positive!')
#
#     kernel = np.zeros((data_1.shape[0], data_2.shape[0]))
#
#     for d in range(data_1.shape[1]):
#         column_1 = data_1[:, d].reshape(-1, 1)
#         column_2 = data_2[:, d].reshape(-1, 1)
#         kernel += np.minimum(column_1, column_2.T)
#
#     print('Kernel DONE!')
#
#     return kernel
#
#
# def try_params(features, split_idx, c, p, fw):
#     '''
#     For SKLeanr, features must be (2400x2400) as a kernel
#     For OpenCV, features must be the feature vectors
#
#     split_idx: splits from 1 to 25
#     '''
#     T1_train_labels_all = T1_train_subject_hrs[split_idx, :]
#     T2_train_labels_all = T2_train_subject_hrs[split_idx, :]
#
#     # build the grid search 4 different splits
#     all_rs = []
#     for i in range(4):
#         train_idx = np.ravel(T1_train_indices[split_idx, i, :]) - 1  # 1-based index
#         test_idx = np.ravel(T2_train_indices[split_idx, i, :]) - 1
#
#         if fw == 'sk':
#             X_train = features[train_idx[:, None], train_idx]
#             X_test = features[test_idx[:, None], train_idx]
#             y_train = T1_train_labels_all[train_idx]
#             y_test = T2_train_labels_all[test_idx]
#
#             # epsilon SVR with histogram intersection kernel
#             svr = SVR(kernel='precomputed', C=c, epsilon=p)
#             predicted_memo_scores = svr.fit(X_train, y_train).predict(X_test)
#         elif fw == 'cv':
#             X_train = features[train_idx, :]
#             X_test = features[test_idx, :]
#             y_train = T1_train_labels_all[train_idx]
#             y_test = T2_train_labels_all[test_idx]
#
#             # opencv demands type float32 for train/test
#             X_train = X_train.astype(np.float32)
#             y_train = y_train.astype(np.float32)
#             X_test = X_test.astype(np.float32)
#             y_test = y_test.astype(np.float32)
#
#             # epsilon SVR with histogram intersection kernel
#             svm = cv2.ml.SVM_create()
#             svm.setKernel(cv2.ml.SVM_INTER)
#             svm.setType(cv2.ml.SVM_EPS_SVR)
#             svm.setC(c)
#             svm.setP(p)
#             svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
#             predicted_memo_scores = np.ravel(svm.predict(X_test)[1])
#         else:
#             raise Exception('invalid option')
#
#         # measure squared correlation coefficient between predicted and ground truth
#         r = np.corrcoef(predicted_memo_scores, y_test)
#         all_rs.append(r)
#
#     return np.mean(all_rs)
#
#
# def run_grid(features, split_idx, fw):
#     '''
#     Grid search for hyperparams c and p of SVR.
#     Finds the best c and p based on squared correlation coefficient.
#     For each pair of c and p, tries 4 splits, averages the squared correlation coefficient on all.
#     Returns the best pair c and p with highest mean squared correlation coefficient.
#     '''
#     # return the best c and p
#     best_r = -1
#     best_c = 0.01
#     best_p = 0.01
#     for i in range(1, 7):
#         for j in range(1, 7):
#             p = 10 ** (i - 5)
#             c = 10 ** (j - 5)
#
#             try:
#                 mean_r = try_params(features, split_idx, c, p, fw)
#                 if mean_r > best_r:
#                     best_r = mean_r
#                     best_c = c
#                     best_p = p
#
#             except Exception as e:
#                 print('exception for hyperparams:', c, p)
#                 print(e)
#                 pass
#
#     return best_r, best_c, best_p
#
#
# def fit_predict(features, fw):
#     # features_kernel must be (2400, 2400)
#     splits = 25
#
#     # train and test for real now
#     top_20_scores = []
#     top_100_scores = []
#     bottom_100_scores = []
#     bottom_20 = []
#     corrs = []
#
#     features_kernel = None
#
#     if fw == 'sk':
#         features_kernel = compute_kernel(features, features)
#
#     for i in tqdm(range(splits)):
#         train_idx = image_train_indices[i, :] - 1
#         test_idx = image_test_indices[i, :] - 1
#
#         if fw == 'sk':
#             best_r, best_c, best_p = run_grid(features_kernel, i, fw)
#             X_train = features_kernel[train_idx[:, None], train_idx]
#             X_test = features_kernel[test_idx[:, None], train_idx]
#             y_train = subject_hrs1[i, train_idx]
#             y_test = subject_hrs2[i, test_idx]
#
#             # epsilon SVR with histogram intersection kernel
#             svr = SVR(kernel='precomputed', C=best_c, epsilon=best_p)
#             predicted_memo_scores = svr.fit(X_train, y_train).predict(X_test)
#         elif fw == 'cv':
#             best_r, best_c, best_p = run_grid(features, i, fw)
#             X_train = features[train_idx, :]
#             X_test = features[test_idx, :]
#             y_train = subject_hrs1[i, train_idx]
#             y_test = subject_hrs2[i, test_idx]
#
#             # opencv demands type float32 for train/test
#             X_train = X_train.astype(np.float32)
#             y_train = y_train.astype(np.float32)
#             X_test = X_test.astype(np.float32)
#             y_test = y_test.astype(np.float32)
#
#             # epsilon SVR with histogram intersection kernel
#             svm = cv2.ml.SVM_create()
#             svm.setKernel(cv2.ml.SVM_INTER)
#             svm.setType(cv2.ml.SVM_EPS_SVR)
#             svm.setC(best_c)
#             svm.setP(best_p)
#             svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
#             _, predicted_memo_scores = svm.predict(X_test)
#         else:
#             raise Exception('invalid option')
#
#         # get indices of sorted memo scores
#         idx = np.argsort(predicted_memo_scores)
#
#         # rank the test measurements according to predictions
#         empirical_memo_scores = y_test[idx]
#
#         # calculate the measured memorability scores according to the ranking
#         top_20_scores.append(empirical_memo_scores[-20:])
#         top_100_scores.append(empirical_memo_scores[-100:])
#         bottom_100_scores.append(empirical_memo_scores[0:100])
#         bottom_20.append(empirical_memo_scores[0:20])
#
#         # calculate spearman rank correlation
#         rho, _ = spearmanr(predicted_memo_scores, y_test)
#         corrs.append(rho)
#
#     top20 = np.sum(top_20_scores) / (25 * 20)
#     top100 = np.sum(top_100_scores) / (25 * 100)
#     bottom100 = np.sum(bottom_100_scores) / (25 * 100)
#     bottom20 = np.sum(bottom_20) / (25 * 20)
#     spearman = np.mean(corrs)
#
#     print('top 20:', top20)
#     print('top 100:', top100)
#     print('bottom 100:', bottom100)
#     print('bottom 20:', bottom20)
#     print('corr avg:', spearman)
#
#     return top20, top100, bottom100, bottom20, spearman
#
#
