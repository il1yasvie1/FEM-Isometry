from src.problem import IsometricBendingProblem
from firedrake import *
from src.utils import plot_deformation
import hydra


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run(cfg):
    problem = IsometricBendingProblem(cfg)
    # z = problem.solve(fname='./outputs/solution.pvd')
    z = problem.solve()
    y = z.sub(0)
    plot_deformation(y, './outputs/figures/solution.png')


if __name__ == "__main__":
    run()
