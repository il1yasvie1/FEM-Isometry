from src.problem import IsometricBendingProblem
from firedrake import *
from src.utils import plot_with_linear_fit, compute_critical_beta, plot_deformation, manufactured_solution, compute_isometry_defect, compute_surface_area
import hydra
import matplotlib.pyplot as plt
import numpy as np


@hydra.main(config_path="conf", config_name="config", version_base=None)
def test_Beta(cfg):
    problem = IsometricBendingProblem(cfg)
    # mesh = SquareMesh(20, 20, 1)
    # problem.mesh = mesh
    # problem.build_problem()
    Z = problem.function_space
    y_exact = manufactured_solution(problem)

    beta_values = np.logspace(1, -6, 10)
    errors = []
    isometry_defects = []
    surface_areas = []
    beta_list = []

    for beta in beta_values:
        problem.beta.assign(beta)
        try:
            z = problem.solve()
            errors += [norm(assemble(z.sub(0)-y_exact))]
            isometry_defects += [compute_isometry_defect(z.sub(0))]
            surface_areas += [compute_surface_area(z.sub(0))]
            beta_list += [beta]
        except ConvergenceError:
            pass

    plot_with_linear_fit(np.log10(beta_list), errors, './outputs/figures/Beta-Error.png', 'log(Beta)', 'L² Error', 'L² Error with Beta', show_linear_fit=False)
    plot_with_linear_fit(np.log10(beta_list), isometry_defects, './outputs/figures/Beta-Iso.png', 'log(Beta)', 'Isometry defect', 'Isometry defect with Beta', show_linear_fit=False)
    plot_with_linear_fit(np.log10(beta_list), surface_areas, './outputs/figures/Beta-Area.png', 'log(Beta)', 'Surface Area', 'Surface Area with Beta', show_linear_fit=False)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def criticalBeta_meshSize(cfg):
    maximal_beta = 6e-7
    minimal_beta = 2e-7
    r0 = 50
    r1 = 1000

    # 50, 50
    # maximal_beta = 3e-6
    # minimal_beta = 8e-7

    # 1000, 50
    # maximal_beta = 3e-6
    # minimal_beta = 8e-7

    beta_list = []
    h_list = []
    errors_list = []
    isometry_defects_list = []
    nx_values = np.linspace(10, 25, 12, dtype=int)
    # nx_values = [10, 25]

    problem = IsometricBendingProblem(cfg)
    problem.r0.assign(r0)
    problem.r1.assign(r1)
    subtitle = f'-r0:{r0}-r1:{r1}'

    for nx in nx_values:
        mesh = SquareMesh(nx, nx, 1)
        problem.mesh = mesh
        problem.build_problem()
        y_exact = manufactured_solution(problem)
        h = assemble(CellSize(problem.mesh)*dx)

        criticalBeta, beta_values, errors, isometry_defects = compute_critical_beta(problem, maximal_beta, minimal_beta,
                                                                          verbose=True,
                                                                          y_exact=y_exact,)
        beta_list.append(criticalBeta)
        errors_list.append(errors[-1])
        isometry_defects_list.append(isometry_defects[-1])
        h_list.append(h)
        maximal_beta = criticalBeta
        print(f"[{nx}/{nx_values[-1]}] "
                      f"CellSize: [{h:.3e}] | "
                      f"Beta: [{criticalBeta:.3e}] | "
                      f"L² Error: [{errors[-1]:.3e}]")

    print('')

    plot_with_linear_fit(h_list, beta_list,
                         './outputs/figures/MeshSize-CritBeta/Beta'+subtitle+'.png',
                         'h', 'Beta', 'Beta with MeshSize')
    plot_with_linear_fit(h_list, errors_list,
                         './outputs/figures/MeshSize-CritBeta/Error'+subtitle+'.png',
                         'h', 'L² Error', 'L² Error with MeshSize at Critical Beta')
    plot_with_linear_fit(h_list, isometry_defects_list,
                         './outputs/figures/MeshSize-CritBeta/Iso'+subtitle+'.png',
                         'h', 'Isometry Defect', 'Isometry Defect with MeshSize at Critical Beta')

    plot_with_linear_fit(np.log10(h_list), np.log10(beta_list),
                         './outputs/figures/MeshSize-CritBeta/Beta-ll'+subtitle+'.png',
                         'log(h)', 'log(Beta)', 'Beta with MeshSize')
    plot_with_linear_fit(np.log10(h_list), np.log10(errors_list),
                         './outputs/figures/MeshSize-CritBeta/Error-ll'+subtitle+'.png',
                         'log(h)', 'log(L² Error)', 'L² Error with MeshSize at Critical Beta')
    plot_with_linear_fit(np.log10(h_list), np.log10(isometry_defects_list),
                         './outputs/figures/MeshSize-CritBeta/Iso-ll'+subtitle+'.png',
                         'log(h)', 'log(Isometry Defect)', 'Isometry Defect with MeshSize at Critical Beta')


@hydra.main(config_path="conf", config_name="config", version_base=None)
def criticalBeta_r0(cfg):
    maximal_beta = 3e-6
    minimal_beta = 2e-6
    r1 = 1000
    r0_values = np.logspace(0, 9, 10)

    subtitle = f'-r1:{r1}'

    beta_list = []
    r0_list = []
    errors_list = []
    isometry_defects_list = []

    problem = IsometricBendingProblem(cfg)
    problem.r1.assign(r1)

    for r0 in r0_values:
        problem.r0.assign(r0)
        y_exact = manufactured_solution(problem)

        criticalBeta, beta_values, errors, isometry_defects = compute_critical_beta(problem,
                                                                          maximal_beta, minimal_beta,
                                                                          verbose=True,
                                                                          y_exact=y_exact)
        beta_list.append(criticalBeta)
        errors_list.append(errors[-1])
        isometry_defects_list.append(isometry_defects[-1])
        r0_list.append(r0)

        print(f"\rr0: [{r0:.3e}] | "
              f"Beta: [{criticalBeta:.3e}] | "
              f"L² Error: [{errors[-1]:.3e}]",
              end='')

    print('')

    # assert(False)
    plot_with_linear_fit(np.log10(r0_list), beta_list,
                         './outputs/figures/r0-CritBeta/Beta'+subtitle+'.png', 'log(r0)', 'Beta', 'Critical Beta with r0', False, False)
    plot_with_linear_fit(np.log10(r0_list), errors_list,
                         './outputs/figures/r0-CritBeta/Error'+subtitle+'.png', 'log(r0)', 'L² Error', 'L² Error with r0 at Critical Beta', False, False)
    plot_with_linear_fit(np.log10(r0_list), isometry_defects_list,
                         './outputs/figures/r0-CritBeta/Iso'+subtitle+'.png', 'log(r0)', 'Isometry Defect', 'Isometry Defect with r0 at Critical Beta', False, False)

    plot_with_linear_fit(np.log10(r0_list), np.log10(beta_list),
                         './outputs/figures/r0-CritBeta/Beta-ll'+subtitle+'.png', 'log(r0)', 'log(Beta)', 'Beta with r0', False, False)
    plot_with_linear_fit(np.log10(r0_list), np.log10(errors_list),
                         './outputs/figures/r0-CritBeta/Error-ll'+subtitle+'.png', 'log(r0)', 'log(L² Error)', 'L² Error with r0 at Critical Beta', False, False)
    plot_with_linear_fit(np.log10(r0_list), np.log10(isometry_defects_list),
                         './outputs/figures/r0-CritBeta/Iso-ll'+subtitle+'.png', 'log(r0)', 'log(Isometry Defect)', 'Isometry Defect with r0 at Critical Beta', False, False)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def criticalBeta_r1(cfg):
    maximal_beta = 1e-3
    minimal_beta = 1e-12
    r0 = 1000
    subtitle = f'-r0:{r0}'

    beta_list = []
    r1_list = []
    errors_list = []
    isometry_defects_list = []

    r1_values = np.logspace(0, 9, 20)

    problem = IsometricBendingProblem(cfg)
    problem.r0.assign(r0)

    for r1 in r1_values:
        problem.r1.assign(r1)
        y_exact = manufactured_solution(problem)

        criticalBeta, beta_values, errors, isometry_defects = compute_critical_beta(problem,
                                                                          maximal_beta, minimal_beta,
                                                                          verbose=True,
                                                                          y_exact=y_exact)
        beta_list.append(criticalBeta)
        errors_list.append(errors[-1])
        isometry_defects_list.append(isometry_defects[-1])
        r1_list.append(r1)

        print(f"\rr1: [{r1:.3e}] | "
              f"Beta: [{criticalBeta:.3e}] | "
              f"L² Error: [{errors[-1]:.3e}]",
              end='')

    print('')
    

    plot_with_linear_fit(np.log10(r1_list), beta_list,
                         './outputs/figures/r1-CritBeta/Beta'+subtitle+'.png', 'log(r1)', 'Beta', 'Critical Beta with r1', False, False)
    plot_with_linear_fit(np.log10(r1_list), errors_list,
                         './outputs/figures/r1-CritBeta/Error'+subtitle+'.png', 'log(r1)', 'L² Error', 'L² Error with r1 at Critical Beta', False, False)
    plot_with_linear_fit(np.log10(r1_list), isometry_defects_list,
                         './outputs/figures/r1-CritBeta/Iso'+subtitle+'.png', 'log(r1)', 'Isometry Defect', 'Isometry Defect with r1 at Critical Beta', False, False)

    plot_with_linear_fit(np.log10(r1_list), np.log10(beta_list),
                         './outputs/figures/r1-CritBeta/Beta-ll'+subtitle+'.png', 'log(r1)', 'log(Beta)', 'Beta with r1', False, False)
    plot_with_linear_fit(np.log10(r1_list), np.log10(errors_list),
                         './outputs/figures/r1-CritBeta/Error-ll'+subtitle+'.png', 'log(r1)', 'log(L² Error)', 'L² Error with r1 at Critical Beta', False, False)
    plot_with_linear_fit(np.log10(r1_list), np.log10(isometry_defects_list),
                         './outputs/figures/r1-CritBeta/Iso-ll'+subtitle+'.png', 'log(r1)', 'log(Isometry Defect)', 'Isometry Defect with r1 at Critical Beta', False, False)


if __name__ == "__main__":
    test_Beta()
    # criticalBeta_meshSize()
    # criticalBeta_r0()
    # criticalBeta_r1()