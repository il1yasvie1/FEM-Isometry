from isometric_bending_solver.problem import IsometricBendingProblem
from isometric_bending_solver.utils import compute_surface_area, compute_isometry_defect, plot_deformation, plot_deformation_anim
from firedrake import *
import numpy as np


nx = 8
family, degree = ('CG', 2)
mesh = RectangleMesh(nx, nx, 2 * np.pi, 1)
x = SpatialCoordinate(mesh)
a = Constant(0.0)

config = {'mesh': mesh,
          'function_space': {'family': family,
                             'degree': degree},
          'continuation': {'a': a,
                           'range': (0, np.pi),
                           'step_size': 0.01,
                           'tangent': True,
                           'saved': True},
          'stabilized': True,
          'f': as_vector([sin(x[0]), 0, -cos(x[0])]),
          'g': as_vector([sin(x[0]),
                          conditional(lt(x[0], np.pi), 0.5 + (x[1] - 0.5) * cos(a), x[1]),
                          conditional(lt(x[0], np.pi), (x[1] - 0.5) * sin(a), 0)
                          ]),
          'phi': as_matrix([[1, 0],
                            [0, conditional(lt(x[0], np.pi), cos(a), 1)],
                            [0, conditional(lt(x[0], np.pi), sin(a), 0)]]),
          'sub_domain': (1, 2),
          'initial_guess': {'y0': as_vector([sin(x[0]), x[1], 1 - cos(x[0])]),
                            'w0': as_vector([0, -x[0], 0])},
          'solver_parameters': {},
          }

problem = IsometricBendingProblem(config)
z, y_list = problem.solve(verbose=True)
y, w, p = z.subfunctions

print(abs(compute_surface_area(y) - 2 * np.pi))
print(compute_isometry_defect(y))

x0 = np.linspace(0, 2 * np.pi, 100)
x1 = np.linspace(0, 1, 20)
plot_deformation(x0, x1, y, './outputs/figures/mobius.png')
plot_deformation_anim(
    x0,
    x1,
    y_list,
    xlim=(-2, 2),
    ylim=(-2, 2),
    zlim=(-.5, 2),
    fname='./outputs/figures/mobius.gif')
