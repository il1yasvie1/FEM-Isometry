from src.continuation import IsometricBendingProblem
from firedrake import *
from src.utils import plot_deformation, expm
import hydra


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run(cfg):
    problem = IsometricBendingProblem(cfg)
    z = problem.solve(fname='./outputs/solution.pvd')
    # z = problem.solve()
    y, w, p = z.subfunctions
    plot_deformation(y, './outputs/figures/solution.png')
    # x = SpatialCoordinate(problem.mesh)
    # y_exact = Function(problem.function_space.sub(0)).interpolate(as_vector([4*sin(np.pi*x[0]/4)/np.pi , x[1], 1 - 4*cos(np.pi*x[0]/4)/np.pi]))
    # print(norm(assemble(y-y_exact)))
    # print(assemble(inner(p, grad(y)-expm(w))*dx))
    # print(assemble(inner(grad(y)-expm(w), grad(y)-expm(w))*dx)**0.5)



if __name__ == "__main__":
    run()
