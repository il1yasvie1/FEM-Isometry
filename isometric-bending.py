from isometric_bending_solver.problem import IsometricBendingProblem
from firedrake import *
from isometric_bending_solver.utils import plot_deformation, compute_isometry_defect, compute_surface_area
import hydra


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run(cfg):
    problem = IsometricBendingProblem(cfg, False)
    z = problem.solve(fname='./outputs/solution.pvd', verbose=True)
    y, w, p = z.subfunctions
    plot_deformation(Function(problem.V_cg).project(y), './outputs/figures/solution.png')
    print(f'Surface Area: {compute_surface_area(y)}')
    print(f'Isometry Defect: {compute_isometry_defect(y)}')


if __name__ == "__main__":
    run()
