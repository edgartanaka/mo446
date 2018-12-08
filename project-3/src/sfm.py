""""
University of Campinas

MO446 - Computer Vision (Prof. Adin)
Project 3 - Reconstruction of 3D Shape from Video

Authors:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Implementation of Structure from Motion (SfM) using our own implementation of
Lucas-Kanade flow algorithm.
"""

import os
import cv2
import klt
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

# Variables to help reading the image sequence
image_prefix = "hotel.seq"
input_frames = os.path.join("input", image_prefix)
n_frames = len(os.listdir(input_frames))

# Parameters for Shi-Tomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Take first frame and find corners in it
img_path = os.path.join(input_frames, image_prefix + "1.png")
old_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

U = np.zeros((n_frames, p0.shape[0]))
V = np.zeros((n_frames, p0.shape[0]))

S = np.ones((p0.shape[0], 1))

U[0, :] = p0[:, 0, 0]
V[0, :] = p0[:, 0, 1]

loops = 0
to_remove = np.array((), np.uint8)

for i in range(2, n_frames + 1):
    img_path = os.path.join(input_frames, image_prefix + str(i) + ".png")
    frame_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Calculate optical flow
    p1, st, err = klt.calcLK(old_gray, frame_gray, p0, 15)

    # Handling lost points
    to_remove = np.append(to_remove, np.where(st == 0)[0])

    # Select good points
    good_new = p1.reshape(-1, p1.shape[-1])

    U[loops, :] = good_new[:, 0]
    V[loops, :] = good_new[:, 1]

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    loops += 1

W = np.concatenate((U, V), axis = 0)

# Remove lost tracks
to_remove = list(set(to_remove))
W = np.delete(W, to_remove, 1)

# Compute the centroid of the image points
a_f = np.mean(W, axis = 1).reshape(-1, 1)

# Subtract this value to center the data
W = W - a_f

# Factorise the measurement matrix using Singular Value Decomposition
U, S, V_t = np.linalg.svd(W)

# As W has rank 3, there are only 3 non-zero singular values
U_ = U[:, :3]
S_ = np.diag(S[:3])
V_ = V_t[:3, :]

M_h = U_      # R*
S_h = S_ @ V_ # S*

# Computing the metric constraints to solve affine ambiguity

Is = M_h[:n_frames, :]
Js = M_h[n_frames:, :]

func = lambda x, y: (x[0] * y[0],
                     x[0] * y[1] + x[1] * y[0],
                     x[0] * y[2] + x[2] * y[0],
                     x[1] * y[1],
                     x[1] * y[2] + x[2] * y[1],
                     x[2] * y[2])

G = np.zeros((3 * n_frames, 6))

for f in range(3 * n_frames):
    if f < n_frames:
        G[f, :] = func(Is[f, :], Is[f, :])
    elif f < 2 * n_frames:
        G[f, :] = func(Js[(f % (n_frames)), :],
                       Js[(f % (n_frames)), :])
    else:
        G[f, :] = func(Is[(f % (2 * n_frames)), :],
                       Js[(f % (2 * n_frames)), :])

c = np.concatenate((np.ones((2 * n_frames, 1)), np.zeros((n_frames, 1))))

l = np.linalg.lstsq(G, c, None)[0]

def squarer(array):
    n = int(np.sqrt(2 * array.size))
    R, C = np.triu_indices(n)
    out = np.zeros((n, n))
    out[R, C] = array
    out[C, R] = array
    return out

L = squarer(l.reshape(-1))

Q = np.linalg.cholesky(L)

M = M_h @ Q
S = np.linalg.inv(Q) @ S_h

# Generate the cloud of points
cloud = PyntCloud(pd.DataFrame(
    data = np.hstack((S.T, 255 * np.ones((S.T.shape[0], 3)))),
    columns = ["x", "y", "z", "red", "green", "blue"]))

cloud.to_file("output/sfm/hotel.ply")
