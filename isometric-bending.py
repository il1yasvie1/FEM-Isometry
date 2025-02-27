from src.problem import IsometricBendingProblem
from firedrake import *
from src.utils import plot_deformation, expm, compute_isometry_defect, compute_surface_area, compute_critical_beta
import hydra


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run(cfg):
    problem = IsometricBendingProblem(cfg)
    print(f'MeshSize: {assemble(CellSize(problem.mesh)*dx)}')

    problem.r.assign(50)
    z = problem.solve(fname='./outputs/solution.pvd', verbose=True)
    y, w, p = z.subfunctions
    plot_deformation(y, './outputs/figures/solution.png')
    print(f'Surface Area: {compute_surface_area(y)}')
    print(f'Isometry Defect: {compute_isometry_defect(y)}')
    Z = problem.function_space
    x = SpatialCoordinate(problem.mesh)
    import numpy as np
    theta = np.pi/4
    y_exact = Function(Z.sub(0)).interpolate(as_vector([sin(theta*x[0])/theta, x[1], 1 - cos(theta*x[0])/theta]))
    print(f'L² Error: {norm(assemble(z.sub(0)-y_exact))}')


if __name__ == "__main__":
    run()
