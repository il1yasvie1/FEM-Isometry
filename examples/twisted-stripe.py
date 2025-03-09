from isometric_bending_solver.problem import IsometricBendingProblem
from isometric_bending_solver.utils import compute_surface_area, compute_isometry_defect, plot_deformation, plot_deformation_anim
from firedrake import *
import numpy as np


ny = 16
family, degree = ('CG', 2)
mesh = RectangleMesh(5*ny, ny, 5, 1)
x = SpatialCoordinate(mesh)
a = Constant(0.0)

config = {'mesh': mesh,
          'function_space': {'family': family,
                             'degree': degree},
          'continuation': {'a': a,
                           'range': (0, np.pi),
                           'step_size': 0.1,
                           'tangent': True,
                           'saved': True},
          'stabilized': True,
          'f': as_vector([0, 0, -1e-16]),
          'g': as_vector([x[0],
                          0.5 + (x[1] - 0.5) * cos(a),
                          conditional(lt(x[0], 2.5), 1, -1)*(x[1] - 0.5) * sin(a)
                          ]),
          'phi': as_matrix([[1, 0],
                            [0, cos(a)],
                            [0, conditional(lt(x[0], 2.5), 1, -1)*sin(a)]]),
          'sub_domain': (1, 2),
          'solver_parameters': {},
          }

problem = IsometricBendingProblem(config)
z, y_list = problem.solve(verbose=True)
y, w, p = z.subfunctions

print(abs(compute_surface_area(y) - 5))
print(compute_isometry_defect(y))

x0 = np.linspace(0, 5, 5*ny)
x1 = np.linspace(0, 1, ny)
plot_deformation(x0, x1, y, './outputs/figures/twisted-stripe.png')
plot_deformation_anim(
    x0,
    x1,
    y_list,
    xlim=(0, 5),
    ylim=(0, 1),
    zlim=(-.5, .5),
    fname='./outputs/figures/twisted-stripe.gif')
