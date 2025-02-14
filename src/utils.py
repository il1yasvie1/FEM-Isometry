from firedrake import *
import matplotlib.pyplot as plt

LeviCivita = Constant([[[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                       [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                       [[0, -1, 0], [1, 0, 0], [0, 0, 0]]])

ProjMat = Constant([[1, 0], [0, 1], [0, 0]])


def skewsym(w):
    return dot(LeviCivita, w)


def expm(w, reduced=True):
    v = sqrt(dot(w, w))
    X = dot(LeviCivita, w)
    expw = Identity(3) + conditional(
        eq(v, 0),
        X + 0.5 * dot(X, X),
        sin(v)/v * X + (1 - cos(v))/v**2 * dot(X, X)
        )
    if reduced:
        return dot(expw, ProjMat)
    else:
        return expw


def dexpm(w, dw, expw=None, reduced=True):
    if not expw:
        expw = expm(w, reduced=False)
    v = dot(w, w)
    dexpw = conditional(eq(v, 0),
                       skewsym(dw),
                       -dot(dot(expw, skewsym(dw)), outer(w, w) + dot(expw.T - Identity(3), skewsym(w)))/v
                       )
    if reduced:
        return dot(dexpw, ProjMat)
    else:
        return dexpw


def plot_deformation(y, fname, equal_aspect=False):
    y1, y2, y3 = y.dat.data.T
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d', azim=-60, elev=30)
    trisurf = ax.plot_trisurf(
        y1, y2, y3,
        cmap='viridis',
        facecolors=plt.cm.viridis(y3),
        edgecolor='k',
        linewidth=0.2,
        antialiased=True,
        alpha=0.9,
        shade=True
    )
    ax.grid(False)
    ax.set_xlabel('y1')
    ax.set_ylabel('y2')
    ax.set_zlabel('y3')
    if equal_aspect:
        ax.set_aspect('equal')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
