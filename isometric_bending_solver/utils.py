from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

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


def compute_isometry_defect(y):
    Dy = grad(y)
    Id = dot(Dy.T, Dy)
    return assemble(inner(Id - Identity(2), Id - Identity(2))*dx)**0.5


def compute_surface_area(y):
    return assemble(sqrt(dot(cross(grad(y)[:, 0], grad(y)[:, 1]), cross(grad(y)[:, 0], grad(y)[:, 1])))*dx)


def manufactured_solution(problem, theta=np.pi/4):
    Z = problem.function_space
    x = SpatialCoordinate(problem.mesh)
    y_exact = Function(Z.sub(0)).interpolate(as_vector([sin(theta*x[0])/theta, x[1], 1 - cos(theta*x[0])/theta]))
    f = Function(Z.sub(0)).interpolate(as_vector([sin(theta*x[0])*theta**3, 0,  -1*cos(theta*x[0])*theta**3]))
    phi = Function(Z.sub(2)).interpolate(as_matrix([[cos(theta*x[0]), 0], [0, 1], [sin(theta*x[0]), 0]]))
    problem.f = f
    problem.g = y_exact
    problem.phi = phi
    return y_exact


def compute_critical_beta(problem,
                          maximal_beta=1e2, minimal_beta=1e-12,
                          rtol=1e-4, max_it=50, verbose=True):
    n_iters = 0

    while (maximal_beta - minimal_beta)/maximal_beta > rtol and n_iters < max_it:
        if verbose:
            print(f'\r n_iters:{n_iters} | maximal_beta:{maximal_beta} | min_beta:{minimal_beta}', end='')
        n_iters += 1
        beta = 10**((np.log10(maximal_beta) + np.log10(minimal_beta))/2)
        # beta = (maximal_beta + minimal_beta)/2

        problem.beta.assign(beta)
        try:
            z = problem.solve()
            maximal_beta = beta
        except ConvergenceError:
            minimal_beta = beta

    if verbose:
        print(f'\ntol: {maximal_beta - minimal_beta}')

    return maximal_beta


def plot_with_linear_fit(x, y, fname,
                         xlabel, ylabel, title,
                         inver_xaxis=True, show_linear_fit=True):

    fig = plt.figure(figsize=(12, 10))
    plt.plot(x, y)
    coeffs = np.polyfit(x, y, 1)
    y_fit = np.polyval(coeffs, x)
    if show_linear_fit==True:
        plt.plot(x, y_fit, linestyle="--", linewidth=2, color="red", label=f"y={coeffs[0]:.4e}x")
        plt.legend()
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    if inver_xaxis:
        plt.gca().invert_xaxis()
    plt.savefig(fname)

    return coeffs
