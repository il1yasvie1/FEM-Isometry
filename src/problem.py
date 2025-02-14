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
        self.function_space = self.create_function_space()

        self.sub_domain = tuple(cfg['sub_domain'])

        self.f = Function(self.function_space.sub(0))
        self.g = Function(self.function_space.sub(0))
        self.phi = Function(self.function_space.sub(2))
        self.bcs = None

        if 'alpha' in cfg:
            self.values = eval(cfg['alpha'])
            self.alpha = Constant(self.values[0])
            self.history = []
        else:
            self.alpha = False

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
        Z = V*W*P
        return Z

    def build_problem(self):
        x = SpatialCoordinate(self.mesh)
        if self.alpha:
            alpha = self.alpha
        f_expr = as_vector([eval(expr) for expr in self.config['f']])
        g_expr = as_vector([eval(expr) for expr in self.config['g']])
        phi_expr = as_matrix([[eval(expr) for expr in row] for row in self.config['phi']])

        self.f.interpolate(f_expr)
        self.g.interpolate(g_expr)
        self.phi.interpolate(phi_expr)
        self.bcs = DirichletBC(self.function_space.sub(0), self.g, self.sub_domain)

    def residual(self, z):
        y, w, p = split(z)
        n = FacetNormal(self.mesh)
        h = CellSize(self.mesh)
        h_avg = (h('+') + h('-'))/2.0
        r = Constant(50)
        beta = Constant(1e-4)

        E = inner(grad(grad(y)), grad(grad(y)))/2*dx - dot(self.f, y)*dx
        E -= inner(dot(avg(grad(grad(y))), n('+')), jump(grad(y)))/2*dS
        E -= inner(dot(grad(grad(y)), n), grad(y) - self.phi)*ds(self.sub_domain)

        E += r*inner(jump(grad(y)), jump(grad(y)))/h_avg*dS
        E += r/2*inner(grad(y)-self.phi, grad(y)-self.phi)/h*ds(self.sub_domain)

        E += inner(p, grad(y)-expm(w))*dx
        E -= beta*h*inner(p, p)*dx

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
        return {
            "snes_max_it": 100,
            "snes_atol": 1.0e-7,
            "snes_rtol": 1.0e-10,
            "snes_max_linear_solve_fail": 100,
            # "snes_linesearch_type": "l2",
            # "snes_linesearch_maxstep": 1.0,
            # "snes_monitor": None,
            # "snes_linesearch_monitor": None,
            # "snes_converged_reason": None,
            # "mat_type": "aij",
            # "ksp_type": "gmres",
            # "ksp_monitor_cancel": None,
            # "ksp_converged_reason": None,
            # "ksp_max_it": 2000,
            # "pc_type": "lu",
            # "pc_factor_mat_solver_type": "mumps",
            # "eps_type": "krylovschur",
            # "eps_target": -1,
            # "eps_monitor_all": None,
            # "eps_converged_reason": None,
            # "eps_nev": 1,
            # "st_type": "sinvert",
            # "st_ksp_type": "preonly",
            # "st_pc_type": "lu",
            # "st_pc_factor_mat_solver_type": "mumps",
        }

    def solve(self, fname=None):
        z = Function(self.function_space).assign(self.inital_guess())

        if 'alpha' not in self.config:
            solve(self.residual(z)==0, z, self.bcs, solver_parameters=self.solver_parameters())
            if fname:
                output_file = VTKFile(fname, project_output=True)
                output_file.write(z.sub(0))
            return z

        history = []
        if fname:
            output_file = VTKFile(fname, project_output=True)

        for alpha in self.values:
            print(f'{alpha}/{self.values[-1]}')
            self.alpha.assign(alpha)
            self.build_problem()
            solve(self.residual(z)==0, z, self.bcs, solver_parameters=self.solver_parameters())
            history += [Function(self.function_space).assign(z)]

            if fname:
                output_file.write(z.sub(0), time=alpha)

        self.history = history
        return z
