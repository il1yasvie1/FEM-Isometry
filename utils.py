from firedrake import *

# Levi-Civita tensor as a constant, used for generating skew-symmetric matrix in 3D.
LeviCivita = Constant([[[0, 0, 0], [0, 0, 1], [0, -1, 0]],
                       [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                       [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]])

# Projection matrix to reduce the matrix exponent
ProjMat = Constant([[1, 0], [0, 1], [0, 0]])


def ExponentialMap(X):
    """
    Computes the exponential map of a skew-symmetric matrix `X` to generate rotation matrices.

    :param X: a 3x3 skew-symmetric UFL matrix.
    :return: a 3x3 UFL matrix that represents the exponential map of `X`.
    """
    v = sqrt(X[0, 1]**2+X[0, 2]**2+X[1, 2]**2)
    return Identity(3) + conditional(eq(v, 0),
            zero((3, 3)),
            sin(v)/v * X + (1 - cos(v))/v**2 * dot(X, X))


def RExpm(x):
    """
    Reduced exponential map. Computes the projection of the exponential map of `x` onto a 2D subspace.

    :param x: a 3D UFL vector in 3D.
    :return: a 3x2 UFL matrix representing the reduced rotation matrix.
    """
    return dot(ExponentialMap(dot(LeviCivita, x)), ProjMat)


def dRExpm(x, v):
    expm = ExponentialMap(dot(LeviCivita, x))
    return dot(dot(expm, dot(LeviCivita, v)), ProjMat)


def plot_solution(z):
    import matplotlib.pyplot as plt
    y1, y2, y3 = z.sub(0).dat.data.T

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
    ax.set_aspect('equal')

    plt.savefig('./outputs/solution.png', dpi=300, bbox_inches='tight', transparent=False)
    output_file = VTKFile("./outputs/solution.pvd", project_output=True)
    output_file.write(z.sub(0), z.sub(1))
