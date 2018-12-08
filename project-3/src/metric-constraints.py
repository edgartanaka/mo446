import numpy as np
import scipy.linalg as sp

def calc(M_h, S_h, n_frames):
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
            G[f, :] = func(Js[(f % (n_frames + 1)) + 1, :],
                           Js[(f % (n_frames + 1)) + 1, :])
        else:
            G[f, :] = func(Is[(f % (2 * n_frames)), :],
                           Js[(f % (2 * n_frames)), :])

    c = np.concatenate((np.ones((2 * n_frames, 1)), np.zeros((n_frames, 1))))

    U, S, V_t = sp.svd(G)

    hatl = U.conjugate().T @ c
    hatl = hatl.ravel()

    y = np.array((hatl[0] / S[0],
                  hatl[1] / S[1],
                  hatl[2] / S[2],
                  hatl[3] / S[3],
                  hatl[4] / S[4],
                  hatl[5] / S[5]))

    l = V_t @ y
    print("Residue with SVD = Gl - c: {0}".format(sp.norm(G @ l - c)))

    l2 = sp.lstsq(G, c)[0]
    print("Residue with lstsq = Gl - c: {0}".format(sp.norm(G @ l2 - c)))

    L = np.array(((l[0], l[1], l[2]),
                  (l[1], l[3], l[4]),
                  (l[2], l[4], l[5])))

    Q = sp.cholesky(L)

    M = M_h @ Q
    S = sp.inv(Q) @ S_h
