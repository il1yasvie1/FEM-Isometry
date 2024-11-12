from firedrake import *
from utils import RExpm, plot_solution


class IsometricBendingProblem:
    def __init__(self, mesh, boundary_conditions) -> None:
        self.mesh = mesh
        self.boundary_conditions = boundary_conditions
        self.function_space = self._set_function_space()
        self._problem = self._set_problem()


    def _set_function_space(self):
        self._V = VectorFunctionSpace(self.mesh, 'CG', degree=2, dim=3)
        self._W = VectorFunctionSpace(self.mesh, 'CG', degree=1, dim=3)
        return MixedFunctionSpace([self._V, self._W])


    def residual(self, z):
        f = Function(self._V).interpolate(self.boundary_conditions['f'])
        g = Function(self._V).interpolate(self.boundary_conditions['g'])
        Phi = Function(self._V).interpolate(self.boundary_conditions['Phi'])
        sub_domain = self.boundary_conditions['sub_domain']

        y, sigma = split(z)

        E = inner(grad(RExpm(sigma)), grad(RExpm(sigma)))/2*dx - dot(f, y)*dx
        E += inner(grad(y)-RExpm(sigma), grad(y)-RExpm(sigma))/2*dx
        E += dot(y-g, y-g)*10/2*ds(sub_domain)
        E += dot(sigma-Phi, sigma-Phi)/2*ds(sub_domain)

        return derivative(E, z)


    def _set_problem(self):
        Z = self.function_space
        x = SpatialCoordinate(self.mesh)

        z0 = Function(Z)
        y0, sigma0 = z0.subfunctions
        y0.interpolate(as_vector([x[0], x[1], 0]))
        sigma0.interpolate(as_vector([0, 0, 0]))
        z = Function(Z)
        z.assign(z0)
        F = self.residual(z)
        problem = NonlinearVariationalProblem(F, z)
        return z, problem


    def solution(self, solver_parameters={}):
        z, problem = self._problem
        solver = NonlinearVariationalSolver(problem,
                                            solver_parameters = solver_parameters)
        solver.solve()
        return z


if __name__  == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    q_degree = 4
    dx = dx(metadata={'quadrature_degree': q_degree})
    dS = dS(metadata={'quadrature_degree': q_degree-1})
    ds = ds(metadata={'quadrature_degree': q_degree-1})

    mesh = SquareMesh(nx=10, ny=10, L=1)
    boundary_conditions = {
        'sub_domain': (1, 2),
        'g': as_vector([SpatialCoordinate(mesh)[0], SpatialCoordinate(mesh)[1], 0]),
        # 'Phi': as_matrix([[1, 0], [0, 1], [0, 0]]),
        'Phi': as_vector([0, 0, 0]),
        'f': as_vector([0, 0, 1]),
    }
    solver_parameters = {
                    'snes_type': 'ngmres',
                    'snes_linesearch_type': 'cp',
                    'snes_linesearch_alpha': 1e-2,
                    'snes_monitor': None,
                    'snes_view': None,
                    'snes_converged_reason': None,
                    'ksp_monitor': None,
                    'snes_atol': 1e-13,
                    'snes_rtol': 1e-16,
    }
    prob = IsometricBendingProblem(mesh, boundary_conditions)
    z = prob.solution(solver_parameters=solver_parameters)
    plot_solution(z)