from src.continuation import IsometricBendingProblem
from firedrake import *
from src.utils import plot_deformation
import hydra
import matplotlib.pyplot as plt
import numpy as np


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run(cfg):

    beta_list = []
    h_list = []
    error_list = []

    beta_start = 3e-6
    beta_stop = 8e-7
    n_beta = 20

    beta_values = np.linspace(beta_start, beta_stop, n_beta)
    nx_values = np.linspace(10, 35, 20, dtype=int)
    print(nx_values)

    problem = IsometricBendingProblem(cfg)
    problem.r0.assign(1e3)

    for nx in nx_values:
        conv_beta = np.inf
        conv_error = 1e10

        mesh = SquareMesh(nx, nx, 1)
        problem.mesh = mesh
        problem.build_problem()
        h = assemble(CellSize(problem.mesh)*dx)
        x = SpatialCoordinate(problem.mesh)
        y_exact = Function(problem.function_space.sub(0)).interpolate(as_vector([4*sin(np.pi*x[0]/4)/np.pi , x[1], 1 - 4*cos(np.pi*x[0]/4)/np.pi]))

        for beta in beta_values:
            problem.beta.assign(beta)
            try:
                z = problem.solve()
                y = z.sub(0)
                conv_beta = beta
                conv_error = norm(assemble(y-y_exact))

            except ConvergenceError:
                print(f"\r[{100 * (nx-min(nx_values)) / max(nx_values):.2f}%] "
                      f"CellSize: [{h:.3e}] | "
                      f"Beta: [{conv_beta:.3e}] | "
                      f"L² Error: [{conv_error:.3e}]",
                      end='')
                beta_list += [conv_beta]
                error_list += [conv_error]
                h_list += [h]
                beta_values = np.linspace(conv_beta, beta_stop, n_beta)
                break

    fig = plt.figure(figsize=(12, 10))
    plt.plot(h_list, beta_list)
    coeffs = np.polyfit(h_list, beta_list, 1)
    beta_fit = np.polyval(coeffs, h_list)
    print(f'\nconeffs for beta: {coeffs}')
    plt.plot(h_list, beta_fit, linestyle="--", linewidth=2, color="red", label=f"y={coeffs[0]:.4e}x")
    plt.xlabel('Mesh Size h', fontsize=14)
    plt.ylabel('The Least Converged-Beta', fontsize=14)
    plt.title('Converged-Beta Value with Mesh Size', fontsize=16)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig('./outputs/figures/test_beta-1e3.png')

    fig = plt.figure(figsize=(12, 10))
    plt.plot(h_list, error_list)
    coeffs = np.polyfit(h_list, error_list, 1)
    error_fit = np.polyval(coeffs, h_list)
    plt.plot(h_list, error_fit, linestyle="--", linewidth=2, color="red", label="Linear Fit")
    plt.xlabel('Mesh Size h', fontsize=14)
    plt.ylabel('L^2 Error at the Least Converged-Beta', fontsize=14)
    plt.title('L^2 Error with Mesh Size', fontsize=16)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig('./outputs/figures/test_error-1e3.png')

    fig = plt.figure(figsize=(12, 10))
    plt.plot(np.log(h_list), np.log(beta_list))
    coeffs = np.polyfit(np.log(h_list), np.log(beta_list), 1)
    beta_fit = np.polyval(coeffs, np.log(h_list))
    plt.plot(np.log(h_list), beta_fit, linestyle="--", linewidth=2, color="red", label=f"y={coeffs[0]:.4e}x")
    plt.xlabel('log(h)', fontsize=14)
    plt.ylabel('log(beta)', fontsize=14)
    plt.title('Check Linearity of Converged-Beta', fontsize=16)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig('./outputs/figures/test_linear_beta-1e3.png')

    fig = plt.figure(figsize=(12, 10))
    plt.plot(np.log(h_list), np.log(error_list))
    coeffs = np.polyfit(np.log(h_list), np.log(error_list), 1)
    error_fit = np.polyval(coeffs, np.log(h_list))
    plt.plot(np.log(h_list), error_fit, linestyle="--", linewidth=2, color="red", label=f"y={coeffs[0]:.4e}x")
    plt.xlabel('log(h)', fontsize=14)
    plt.ylabel('log(error)', fontsize=14)
    plt.title('Check Linearity of L^2 Error', fontsize=16)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig('./outputs/figures/test_linear_error-1e3.png')


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run0(cfg):
    beta_values = np.logspace(10, -10, 40)
    nx = 10
    problem = IsometricBendingProblem(cfg)
    mesh = SquareMesh(nx, nx, 1)
    problem.mesh = mesh
    problem.r0.assign(1e9)
    problem.build_problem()

    x = SpatialCoordinate(problem.mesh)
    y_exact = Function(problem.function_space.sub(0)).interpolate(as_vector([4*sin(np.pi*x[0]/4)/np.pi , x[1], 1 - 4*cos(np.pi*x[0]/4)/np.pi]))

    conv_error = []
    beta_list = []

    for beta in beta_values:
        problem.beta.assign(beta)
        try:
            z = problem.solve()
            y = z.sub(0)
            conv_error += [norm(assemble(y-y_exact))]
            beta_list += [beta]
        except ConvergenceError:
            # conv_error += [1e-1]
            pass

    fig = plt.figure(figsize=(12, 10))
    # plt.loglog(beta_values, conv_error)
    plt.loglog(beta_list, conv_error)
    plt.xlabel('Beta', fontsize=14)
    plt.ylabel('L^2 Error', fontsize=14)
    plt.title('L^2 Error with Beta', fontsize=16)
    plt.gca().invert_xaxis()
    plt.savefig('./outputs/figures/error_beta-c-1e9.png')

if __name__ == "__main__":
    run()
