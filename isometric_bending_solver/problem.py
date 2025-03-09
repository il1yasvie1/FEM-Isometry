# flake8: noqa: F403, F405
import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np
from .utils import *

q_degree = 2
dx = dx(metadata={'quadrature_degree': q_degree})
dS = dS(metadata={'quadrature_degree': q_degree})
ds = ds(metadata={'quadrature_degree': q_degree})


class IsometricBendingProblem:
    def __init__(self, cfg, from_dict=True):
        self.config = cfg
        self.mesh = self.create_mesh(from_dict)
        self.function_space = self.create_function_space()

        self.continuation = cfg.get('continuation')
        self.r = Constant(cfg.get('r', 50))
        self.f, self.g, self.phi = [None] * 3
        self.sub_domain = None
        self._get_ufl_expr(from_dict)
        self.Jp = None
        self.bcs = None

        if self.continuation in ({}, False, None):
            self.continuation = False
        else:
            self.a = cfg['continuation'].get('a')
            self.range = cfg['continuation'].get('range')
            self.a.assign(self.range[0])
            self.step_size = cfg['continuation'].get('step_size')
            self.tangent = cfg['continuation'].get('tangent', False)
            self.saved = cfg['continuation'].get('saved', False)
            self.continuation = {'range': self.range,
                                 'step_size': self.step_size,
                                 'tangent': self.tangent,
                                 'saved': self.saved}
            self.values = list(np.arange(self.range[0], self.range[1], self.step_size))
            if self.values[-1] != self.range[-1]:
                self.values.append(self.range[-1])

        self.stabilized = cfg.get('stabilized')
        if self.stabilized in (False, None):
            self.stabilized = False
        else:
            if self.stabilized:
                self.beta = Constant(1e-3)
            elif isinstance(self.stabilized, dict):
                self.beta = Constant(cfg['stabilized'].get('beta', 1e-3))
            else:
                raise NotImplementedError
            self.stabilized = {'beta': float(self.beta)}

        self.nitsche = cfg.get('nitsche')
        if self.nitsche in (False, None):
            self.nitsche = False
            self.bcs = DirichletBC(
                self.function_space.sub(0),
                self.g,
                self.sub_domain)
        else:
            if self.nitsche:
                self.r0 = Constant(1e6)
            elif isinstance(self.nitsche, dict):
                self.r0 = Constant(cfg['nitsche'].get('r0', 1e6))
            else:
                raise NotImplementedError
            self.nitsche = {'r0': float(self.r0)}

        self.solver_parameters = dict(
            cfg.get(
                'solver_parameters',
                {'snes_rtol': 1e-6, 'snes_atol': 1e-8, }))

        self.initial_guess = cfg.get('initial_guess')
        if self.initial_guess in (False, None):
            self.initial_guess = False
        else:
            self.y0_expr = cfg['initial_guess']['y0']
            self.w0_expr = cfg['initial_guess']['w0']

    def create_mesh(self, from_dict):
        if from_dict:
            return self.config['mesh']
        else:
            supported_mesh_types = {
                'RectangleMesh': RectangleMesh,
                'SquareMesh': SquareMesh,
            }
            mesh_cfg = self.config['mesh']
            mesh_type = mesh_cfg['type']
            if mesh_type in supported_mesh_types:
                mesh = supported_mesh_types[mesh_type](
                    **mesh_cfg['parameters'])
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

    def _get_ufl_expr(self, from_dict):
        if from_dict:
            self.f = self.config['f']
            self.g = self.config['g']
            self.phi = self.config['phi']
            self.sub_domain = self.config['sub_domain']
        else:
            x = SpatialCoordinate(self.mesh)
            if self.continuation:
                a = self.a
            self.f_expr = as_vector([eval(expr) for expr in self.config['f']])
            self.g_expr = as_vector([eval(expr) for expr in self.config['g']])
            self.phi_expr = as_matrix([[eval(expr) for expr in row]
                                       for row in self.config['phi']])
            self.sub_domain = tuple(self.config['sub_domain'])

    def residual(self, z):
        y, w, p = split(z)
        n = FacetNormal(self.mesh)
        h = CellSize(self.mesh)
        h_avg = (h('+') + h('-')) / 2.0

        r = self.r
        f, g, phi = self.f, self.g, self.phi
        sub_domain = self.sub_domain

        Dy = grad(y)
        DDy = grad(Dy)

        E = .5 * inner(DDy, DDy) * dx
        E -= inner(dot(avg(DDy), n('+')), jump(Dy)) * dS
        E += .5 * r / h_avg * inner(jump(Dy), jump(Dy)) * dS

        if self.continuation:
            E += Constant(0.) * self.a * inner(p, p) * dx

        E -= dot(f, y) * dx
        E -= inner(dot(DDy, n), Dy - phi) * ds(sub_domain)
        E += .5 * r / h * inner(Dy - phi,
                                Dy - phi) * ds(sub_domain)

        if self.nitsche:
            r0 = self.r0
            E += dot(dot(avg(div(DDy)), n('+')), jump(y)) * dS
            E += .5 * r0 / h_avg**(-3) * dot(jump(y), jump(y)) * dS

            E += dot(dot(div(DDy), n), y - g) * ds(sub_domain)
            E += .5 * r0 / h**(-3) * dot(y - g,
                                         y - g) * ds(sub_domain)

        E += inner(p, grad(y) - expm(w)) * dx

        if self.stabilized:
            E -= self.beta * inner(p, p) * dx
        else:
            self.Jp = derivative(derivative(E - inner(p, p) * dx, z), z)
        F = derivative(E, z)
        return F

    def _set_initial_guess(self):
        z0 = Function(self.function_space)
        y0, w0, p0 = z0.subfunctions
        if self.initial_guess:
            y0.project(self.y0_expr)
            w0.interpolate(self.w0_expr)
            p0.interpolate(as_matrix([[0, 0], [0, 0], [0, 0]]))
        else:
            x = SpatialCoordinate(self.mesh)
            y0.project(as_vector([x[0], x[1], 0]))
            w0.interpolate(as_vector([0, 0, 0]))
            p0.interpolate(as_matrix([[0, 0], [0, 0], [0, 0]]))
        return z0

    def _solve_continuation(self, z, output_file=None, verbose=False):
        y_list = [Function(self.function_space.sub(0)).assign(z.sub(0))]
        for a in self.values:
            if verbose:
                print(
                    f'\rProgress: {100 * (a - self.values[0]) / self.values[-1]:.1f}%', end='')

            self.a.assign(a)
            self.nsolver.solve()

            if output_file:
                output_file.write(z.sub(0), time=a)

            if self.saved:
                y_list.append(
                    Function(
                        self.function_space.sub(0)).assign(
                        z.sub(0)))

        print('')
        print("Solver converged")
        print("=" * 60)
        if self.saved:
            return z, y_list
        else:
            return z

    def _solve_continuation_tangent(self, z, output_file=None, verbose=False):
        da = Constant(0.0)
        F = self.residual(z)
        dFda = derivative(F, self.a, da)
        dFdz = derivative(F, z)
        dz = Function(self.function_space)
        if not self.nitsche:
            dbcs = DirichletBC(
                self.function_space.sub(0),
                derivative(self.g, self.a, da),
                self.sub_domain)
        else:
            dbcs = None

        y_list = [Function(self.function_space.sub(0)).assign(z.sub(0))]
        problem = LinearVariationalProblem(dFdz, -dFda, dz, bcs=dbcs)
        solver = LinearVariationalSolver(problem)

        for a in self.values[1:]:
            if verbose:
                print(
                    f'\rProgress: {100 * (a - self.values[0]) / self.values[-1]:.1f}%', end='')
            da.assign(a - self.a)
            self.a.assign(a)
            solver.solve()
            z += dz
            self.nsolver.solve()

            if output_file:
                output_file.write(z.sub(0), time=float(a))

            if self.saved:
                y_list.append(
                    Function(
                        self.function_space.sub(0)).assign(
                        z.sub(0)))
        print('')
        print("Solver converged")
        print("=" * 60)
        if self.saved:
            return z, y_list
        else:
            return z
        
    def _optimal_continuation(self):
        min_step = 0.001

        z = Function(self.function_space).assign(self._set_initial_guess())
        F = self.residual(z)
        self.nproblem = NonlinearVariationalProblem(
            F, z, bcs=self.bcs, Jp=self.Jp)
        self.nsolver = NonlinearVariationalSolver(
            self.nproblem, solver_parameters=self.solver_parameters)
        
        self.nsolver.solve()
        a_values = list(np.arange(self.values[0], 1.0, min_step))
        a0, a1 = 0, len(a_values)-1

        continuation_list = [a0]
        step_list = []
        z0 = Function(self.function_space).assign(z)
        avg_step = 0
        minimal_step = 1

        while continuation_list[-1] != a_values[-1]:
            print(f'a0={a_values[a0]: .3f}, a1={a_values[a1]: .3f}, avg_step={avg_step*min_step: .3f}, minimal_step={minimal_step: .3f}')
            converged = False
            amid = a1
            while a1 - a0 > 1:
                try:
                    self.a.assign(a_values[amid])
                    self.nsolver.solve()
                    z.assign(z0)
                    a0 = amid
                    converged = True

                except ConvergenceError:
                    z.assign(z0)
                    a1 = amid
                amid = (a0 + a1)//2

            if converged:
                z.assign(z0)
                self.a.assign(a_values[amid])
                self.nsolver.solve()
                z0.assign(z)
                step_list.append(amid - continuation_list[-1])
                minimal_step = min(minimal_step, (amid-continuation_list[-1])*min_step)
                avg_step = int(np.mean(step_list))
                a0 = amid
                a1 = int(min(amid + 3*avg_step, len(a_values)-1))
                continuation_list.append(a0)
            else:
                print('min step fails!')
                break

    def solve(self, fname=None, verbose=False):
        print("=" * 60)
        print('██╗███████╗ ██████╗ ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗')
        print('██║██╔════╝██╔═══██╗████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝')
        print('██║███████╗██║   ██║██╔████╔██║█████╗     ██║   ██████╔╝██║██║ ')
        print('██║╚════██║██║   ██║██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║ ')
        print('██║███████║╚██████╔╝██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗')
        print('╚═╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝')
        print("=" * 60)
        print(f"{'Family:':<30} {self.family}_{self.degree}")
        print(f"{'MeshSize:':<30} {assemble(CellSize(self.mesh) * dx):.6f}")
        print(f"{'Continuation:':<30} {self.continuation}")
        print(f"{'Nitsches Approach:':<30} {self.nitsche}")
        print(f"{'stabilized:':<30} {self.stabilized}")
        print("=" * 60)

        z = Function(self.function_space).assign(self._set_initial_guess())
        F = self.residual(z)
        self.nproblem = NonlinearVariationalProblem(
            F, z, bcs=self.bcs, Jp=self.Jp)
        self.nsolver = NonlinearVariationalSolver(
            self.nproblem, solver_parameters=self.solver_parameters)

        try:
            self.nsolver.solve()

            if fname:
                output_file = VTKFile(fname, project_output=True)
                output_file.write(z.sub(0))
            else:
                output_file = None

            if self.continuation:
                if self.tangent:
                    return self._solve_continuation_tangent(
                        z, output_file, verbose)
                else:
                    return self._solve_continuation(z, output_file, verbose)
            else:
                print("Solver converged")
                return z

        except ConvergenceError:
            print("Convergence Error!")
