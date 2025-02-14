import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from .utils import *
import numpy as np

q_degree = 2
dx = dx(metadata={'quadrature_degree': q_degree})
dS = dS(metadata={'quadrature_degree': q_degree})
ds = ds(metadata={'quadrature_degree': q_degree})


class IsometricBendingProblem:
    def __init__(self, cfg):
        self.config = cfg
        self.mesh = self.create_mesh()
        self.sub_domain = tuple(cfg['sub_domain'])

        self.function_space = None

        self.f = None
        self.g = None
        self.phi = None

        self.f0 = None
        self.g0 = None
        self.phi0 = None

        self.values = eval(cfg['alpha'])
        self.alpha = Constant(self.values[0])

        self.r0 = Constant(50)
        self.r1 = Constant(50)
        self.beta = Constant(1e-4)

        self.build_problem()

    def create_mesh(self):
        supported_mesh_types = {
            'RectangleMesh': RectangleMesh,
            'SquareMesh': SquareMesh,
        }
        mesh_cfg = self.config['mesh']
        mesh_type = mesh_cfg['type']
        if mesh_type in supported_mesh_types:
            mesh = supported_mesh_types[mesh_type](**mesh_cfg['parameters'])
            return mesh
        else:
            raise NotImplementedError

    def create_function_space(self):
        V = VectorFunctionSpace(self.mesh, 'CG', degree=2, dim=3, name='Deformation')
        W = VectorFunctionSpace(self.mesh, 'DG', degree=1, dim=3)
        P = TensorFunctionSpace(self.mesh, 'DG', degree=1, shape=(3, 2))
        self.f = Function(V)
        self.g = Function(V)
        self.phi = Function(P)

        Z = V*W*P
        return Z

    def build_problem(self):
        x = SpatialCoordinate(self.mesh)
        self.function_space = self.create_function_space()
        f_expr = as_vector([eval(expr) for expr in self.config['f']])
        g_expr = as_vector([eval(expr) for expr in self.config['g']])
        phi_expr = as_matrix([[eval(expr) for expr in row] for row in self.config['phi']])
        self.f.interpolate(f_expr)
        self.g.interpolate(g_expr)
        self.phi.interpolate(phi_expr)

        if self.config['f0']:
            f0_expr = as_vector([eval(expr) for expr in self.config['f0']])
            self.f0 = Function(self.function_space.sub(0)).interpolate(f0_expr)

        if self.config['g0']:
            g0_expr = as_vector([eval(expr) for expr in self.config['g0']])
            self.g0 = Function(self.function_space.sub(0)).interpolate(g0_expr)

        if self.config['phi0']:
            phi0_expr = as_matrix([[eval(expr) for expr in row] for row in self.config['phi0']])
            self.phi0 = Function(self.function_space.sub(2)).interpolate(phi0_expr)

    def residual(self, z):
        y, w, p = split(z)
        n = FacetNormal(self.mesh)
        h = CellSize(self.mesh)
        h_avg = (h('+') + h('-'))/2.0

        r0, r1, beta = self.r0, self.r1, self.beta

        E = inner(grad(grad(y)), grad(grad(y)))/2*dx
        E -= inner(dot(avg(grad(grad(y))), n('+')), jump(grad(y)))/2*dS
        E += r1*inner(jump(grad(y)), jump(grad(y)))/h_avg*dS

        if self.f0:
            E -= dot(self.alpha*self.f + (1-self.alpha)*self.f0, y)*dx
        else:
            E -= dot(self.f, y)*dx

        if self.g0:
            E += r0/2*dot(y - self.alpha*self.g - (1-self.alpha)*self.g0, y - self.alpha*self.g - (1-self.alpha)*self.g0)*ds(self.sub_domain)
        else:
            E += r0/2*dot(y - self.g, y - self.g)*ds(self.sub_domain)

        if self.phi0:
            E -= inner(dot(grad(grad(y)), n), grad(y) - self.alpha*self.phi - (1-self.alpha)*self.phi0)*ds(self.sub_domain)
            E += r1/2*inner(grad(y)-self.alpha*self.phi - (1-self.alpha)*self.phi0, grad(y) - self.alpha*self.phi - (1-self.alpha)*self.phi0)/h*ds(self.sub_domain)
        else:
            E -= inner(dot(grad(grad(y)), n), grad(y) - self.phi)*ds(self.sub_domain)
            E += r1/2*inner(grad(y)-self.phi, grad(y)-self.phi)/h*ds(self.sub_domain)

        E += inner(p, grad(y)-expm(w))*dx
        E -= beta*inner(p, p)*dx
        F = derivative(E, z)

        return F

    def inital_guess(self):
        x = SpatialCoordinate(self.mesh)
        z0 = Function(self.function_space)
        y0, w0, p0 = z0.subfunctions
        y0.interpolate(as_vector([x[0], x[1], 0]))
        w0.interpolate(as_vector([0, 0, 0]))
        p0.interpolate(as_matrix([[0, 0], [0, 0], [0, 0]]))
        return z0

    def solver_parameters(self):
        return {'snes_max_it': 25,
                'ksp_max_it': 2000,
                'snes_converged_reason': None,
                'ksp_converged_reason': None,
                'snes_monitor': None,
                }

    def solve(self, fname=None, verbose=False):
        z = Function(self.function_space).assign(self.inital_guess())
        if verbose:
            print(f'{self.values[0]}/{self.values[-1]}')
        solve(self.residual(z)==0, z, solver_parameters=self.solver_parameters())

        if fname:
            output_file = VTKFile(fname, project_output=True)
            output_file.write(z.sub(0), time=self.alpha)

        R = FunctionSpace(self.mesh, 'R', 0)
        dalpha = Function(R)
        F = self.residual(z)
        dFda = action(derivative(F, self.alpha), dalpha)
        dFdz = derivative(F, z)
        dz = Function(self.function_space)
        problem = LinearVariationalProblem(dFdz, -dFda, dz)
        solver = LinearVariationalSolver(problem, solver_parameters=self.solver_parameters())

        nproblem = NonlinearVariationalProblem(self.residual(z), z)
        nsolver = NonlinearVariationalSolver(nproblem, solver_parameters=self.solver_parameters())

        for alpha in self.values[1:]:
            if verbose:
                print(f'{alpha}/{self.values[-1]}')
            dalpha.assign(alpha-self.alpha)
            self.alpha.assign(alpha)
            solver.solve()
            z += dz

            nsolver.solve()

            if fname:
                output_file.write(z.sub(0), time=alpha)
        return z
