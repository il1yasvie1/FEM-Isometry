# flake8: noqa: F403, F405
import numpy as np
from .utils import *
from firedrake import *
import os
os.environ["OMP_NUM_THREADS"] = "1"


q_degree = 2
dx = dx(metadata={'quadrature_degree': q_degree})
dS = dS(metadata={'quadrature_degree': q_degree})
ds = ds(metadata={'quadrature_degree': q_degree})


class IsometricBendingProblem:
    def __init__(self, cfg):
        self.config = cfg

        self.continuation = cfg.get('continuation', False)
        self.on_f, self.on_g, self.on_phi = [None] * 3
        if self.continuation:
            self.on_f = cfg['continuation'].get('f0', False)
            self.on_g = cfg['continuation'].get('g0', False)
            self.on_phi = cfg['continuation'].get('phi0', False)
            self.tangent = cfg['continuation'].get('tangent', False)

        self.nitsche = cfg.get('nitsche', False)
        if self.nitsche:
            self.r0 = Constant(cfg.get('r0', 1e6))

        self.isRegularised = cfg.get('isRegularised', False)
        if self.isRegularised:
            self.beta = cfg['isRegularised'].get('beta', Constant(1e-3))

        self.r = Constant(cfg.get('r', 50))

        self.mesh = self.create_mesh()
        self.function_space = self.create_function_space()

        self._get_ufl_expr()
        self._interpolate()

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
        self.family = self.config['function_space']['family']
        self.degree = int(self.config['function_space']['degree'])
        if self.family in {'CG', 'Lagrange',
                           'DG', 'Discontinuous Lagrange'}:
            V = VectorFunctionSpace(self.mesh, self.family, self.degree, 3)
            self.V_cg = V
            W = VectorFunctionSpace(self.mesh, 'DG', self.degree - 1, 3)
            P = TensorFunctionSpace(self.mesh, 'DG', self.degree - 1, (3, 2))
        elif self.family in {'BELL', 'Bell', 'ARG', 'Argyris',
                             'HER', 'Hermite',
                             'MOR', 'Morley'}:
            V = VectorFunctionSpace(self.mesh, self.family, self.degree, 3)
            self.V_cg = VectorFunctionSpace(self.mesh, 'CG', self.degree, 3)
            W = VectorFunctionSpace(self.mesh, 'CG', self.degree - 1, 3)
            P = TensorFunctionSpace(self.mesh, 'CG', self.degree - 1, (3, 2))
        self.R = FunctionSpace(self.mesh, 'R', 0)
        Z = V * W * P
        return Z

    def _get_ufl_expr(self):
        x = SpatialCoordinate(self.mesh)
        self.f_expr = as_vector([eval(expr) for expr in self.config['f']])
        self.g_expr = as_vector([eval(expr) for expr in self.config['g']])
        self.phi_expr = as_matrix([[eval(expr) for expr in row]
                                  for row in self.config['phi']])
        self.sub_domain = tuple(self.config['sub_domain'])

        if self.continuation:
            continuation_cfg = self.config['continuation']
            self.values = eval(continuation_cfg['alpha'])

            if self.on_f:
                self.f0_expr = as_vector([eval(expr)
                                         for expr in continuation_cfg['f0']])

            if self.on_g:
                self.g0_expr = as_vector([eval(expr)
                                         for expr in continuation_cfg['g0']])

            if self.on_phi:
                self.phi0_expr = as_matrix(
                    [[eval(expr) for expr in row] for row in continuation_cfg['phi0']])

    def _interpolate(self):
        f_cg = Function(self.V_cg).interpolate(self.f_expr)
        self.f = Function(self.function_space.sub(0)).project(f_cg)
        g_cg = Function(self.V_cg).interpolate(self.g_expr)
        self.g = Function(self.function_space.sub(0)).project(g_cg)
        self.phi = Function(
            self.function_space.sub(2)).interpolate(
            self.phi_expr)

        if self.nitsche:
            self.bcs = None
        else:
            self.bcs = [
                DirichletBC(
                    self.function_space.sub(0),
                    self.g,
                    self.sub_domain)]

        if self.continuation:
            self.alpha = Function(self.R).assign(self.values[0])
            if self.on_f:
                f0_cg = Function(self.V_cg).interpolate(self.f0_expr)
                self.f0 = Function(
                    self.function_space.sub(0)).project(
                    f0_cg)
            if self.on_g:
                g0_cg = Function(self.V_cg).interpolate(self.g0_expr)
                self.g0 = Function(
                    self.function_space.sub(0)).project(
                    g0_cg)

                if self.nitsche:
                    self.bcs = None
                else:
                    self.bcs = [DirichletBC(self.function_space.sub(0),
                                            self.alpha * self.g + (1 - self.alpha) * self.g0,
                                            self.sub_domain)]
            if self.on_phi:
                self.phi0 = Function(
                    self.function_space.sub(2)).interpolate(
                    self.phi0_expr)

    def residual(self, z):
        y, w, p = split(z)
        n = FacetNormal(self.mesh)
        h = CellSize(self.mesh)
        h_avg = (h('+') + h('-')) / 2.0

        r = self.r
        Dy = grad(y)
        DDy = grad(Dy)

        E = .5 * inner(DDy, DDy) * dx
        E -= inner(dot(avg(DDy), n('+')), jump(Dy)) * dS
        E += .5 * r / h_avg * inner(jump(Dy), jump(Dy)) * dS

        if self.continuation:
            E += Constant(0.) * self.alpha * inner(p, p) * dx

        if self.on_f:
            E -= dot(self.alpha * self.f + (1 - self.alpha) * self.f0, y) * dx

        else:
            E -= dot(self.f, y) * dx

        if self.on_phi:
            E -= inner(dot(DDy, n), Dy - self.alpha * self.phi -
                       (1 - self.alpha) * self.phi0) * ds(self.sub_domain)
            E += .5 * r / h * inner(Dy - self.alpha * self.phi - (1 - self.alpha) * self.phi0,
                                    Dy - self.alpha * self.phi - (1 - self.alpha) * self.phi0) * ds(self.sub_domain)

        else:
            E -= inner(dot(DDy, n), Dy - self.phi) * ds(self.sub_domain)
            E += .5 * r / h * inner(Dy - self.phi,
                                    Dy - self.phi) * ds(self.sub_domain)

        if self.nitsche:
            r0 = self.r0
            E += dot(dot(avg(div(DDy)), n('+')), jump(y)) * dS
            E += .5 * r0 / h_avg**(-1.5) * dot(jump(y), jump(y)) * dS
            if self.on_g:
                E += dot(dot(div(DDy), n), y - self.alpha * self.g -
                         (1 - self.alpha) * self.g0) * ds(self.sub_domain)
                E += .5 * r0 / h**(-1.5) * dot(y - self.alpha * self.g - (1 - self.alpha) * self.g0,
                                               y - self.alpha * self.g - (1 - self.alpha) * self.g0) * ds(self.sub_domain)
            else:
                E += dot(dot(div(DDy), n), y - self.g) * ds(self.sub_domain)
                E += .5 * r0 / h**(-1.5) * dot(y - self.g,
                                               y - self.g) * ds(self.sub_domain)

        E += inner(p, grad(y) - expm(w)) * dx

        if self.isRegularised:
            E -= self.beta * inner(p, p) * dx
            self.Jp = None
        else:
            self.Jp = derivative(derivative(E - inner(p, p) * dx, z), z)
        F = derivative(E, z)
        return F

    def initial_guess(self):
        x = SpatialCoordinate(self.mesh)
        z0 = Function(self.function_space)
        y0, w0, p0 = z0.subfunctions
        y0.project(as_vector([x[0], x[1], 0]))
        w0.interpolate(as_vector([0, 0, 0]))
        p0.interpolate(as_matrix([[0, 0], [0, 0], [0, 0]]))
        return z0

    def solver_parameters(self):
        return {
            # 'snes_converged_reason': None,
            # 'ksp_converged_reason': None,
            # 'ksp_monitor': None,
            # 'snes_monitor': None,
            # 'snes_type': 'newtonls',
            # 'ksp_type': 'gmres',
            # 'ksp_gmres_restart': 30,
            'snes_rtol': 1e-6,
            'snes_atol': 1e-8,
            # 'snes_stol': 1e-50,
            # 'snes_max_it': 50,
            # 'ksp_rtol': 1e-8,
            # 'ksp_atol': 1e-50,
            # 'ksp_divtol': 1e4,
            # 'ksp_max_it': 10000,
            # 'ksp_view': None,
            # 'pc_type': 'ilu',

            # "pc_type": "python",
            # "pc_python_type": "firedrake.ASMStarPC",
            # "pc_star_construct_dim": 0,
            # "pc_star_sub_sub_pc_type": 'lu',
            # "pc_star_sub_sub_pc_factor_mat_solver_type": 'umfpack',
            # vanka, star,
            # mat: mumps, superlu

            # "pc_type": "python",
            # "pc_python_type": "firedrake.PatchPC",
            # "patch_pc_patch_save_operators": True,
            # "patch_pc_patch_partition_of_unity": True,
            # "patch_pc_patch_sub_mat_type": "seqdense",
            # "patch_pc_patch_construct_dim": 0,
            # "patch_pc_patch_construct_type": "star",
            # "patch_pc_patch_local_type": "additive",
            # "patch_pc_patch_precompute_element_tensors": True,
            # "patch_pc_patch_symmetrise_sweep": False,
            # "patch_sub_ksp_type": "preonly",
            # "patch_sub_pc_type": "ilu",
            # 'patch_sub_pc_factor_mat_solver_type': 'superlu',
            # "patch_sub_pc_factor_shift_type": "nonzero",
        }

    def _solve_continuation(self, z, output_file=None, verbose=False):
        for alpha in self.values:
            if verbose:
                print(
                    f'\rProgress: {100 * alpha / self.values[-1]:.1f}%', end='')

            self.alpha.assign(alpha)
            self.nsolver.solve()

            if output_file:
                output_file.write(z.sub(0), time=alpha)
        print('')
        print("=" * 60)
        return z

    def _solve_continuation_tangent(self, z, output_file=None, verbose=False):
        dalpha = Function(self.R)
        F = self.residual(z)
        dFda = derivative(F, self.alpha, dalpha)
        dFdz = derivative(F, z)
        dz = Function(self.function_space)
        if self.on_g and not self.nitsche:
            dbcs = DirichletBC(
                self.function_space.sub(0),
                (self.g - self.g0) * dalpha,
                self.sub_domain)
        else:
            dbcs = None

        problem = LinearVariationalProblem(dFdz, -dFda, dz, bcs=dbcs)
        solver = LinearVariationalSolver(problem)

        for alpha in self.values[1:]:
            if verbose:
                print(
                    f'\rProgress: {100 * alpha / self.values[-1]:.1f}%', end='')
            dalpha.assign(alpha - self.alpha)
            self.alpha.assign(alpha)
            solver.solve()
            z += dz
            self.nsolver.solve()

            if output_file:
                output_file.write(z.sub(0), time=alpha)
        print('')
        print("=" * 60)
        return z

    def solve(self, fname=None, verbose=False):
        print("=" * 60)
        print('██╗███████╗ ██████╗ ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗')
        print('██║██╔════╝██╔═══██╗████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝')
        print('██║███████╗██║   ██║██╔████╔██║█████╗     ██║   ██████╔╝██║██║ ')
        print('██║╚════██║██║   ██║██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║ ')
        print('██║███████║╚██████╔╝██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗')
        print('╚═╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝')
        print("=" * 60)
        print(f"{'Continuation:':<30} {self.continuation}")
        print(f"{'Family:':<30} {self.family}_{self.degree}")
        print(f"{'Nitsche\'s Approach:':<30} {self.nitsche}")
        print(f"{'isRegularised:':<30} {self.isRegularised}")
        print(f"{'MeshSize:':<30} {assemble(CellSize(self.mesh) * dx):.6f}")
        print("=" * 60)

        z = Function(self.function_space).assign(self.initial_guess())
        F = self.residual(z)
        self.nproblem = NonlinearVariationalProblem(
            F, z, bcs=self.bcs, Jp=self.Jp)
        self.nsolver = NonlinearVariationalSolver(
            self.nproblem, solver_parameters=self.solver_parameters())
        self.nsolver.solve()

        if fname:
            output_file = VTKFile(fname, project_output=True)
            output_file.write(z.sub(0))

        if self.continuation:
            if self.tangent:
                return self._solve_continuation_tangent(
                    z, output_file, verbose)
            else:
                return self._solve_continuation(z, output_file, verbose)
        else:
            return z
