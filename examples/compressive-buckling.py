from isometric_bending_solver.problem1 import IsometricBendingProblem
from isometric_bending_solver.utils import compute_surface_area, compute_isometry_defect, plot_deformation, plot_deformation_anim
from firedrake import *
import numpy as np

ny = 16
r = 50
family, degree, nitsche = ('CG', 2, False)
stabilized = True
mesh = RectangleMesh(4 * ny, ny, 4, 1)

x = SpatialCoordinate(mesh)
a = Constant(0.0)
values = list(np.arange(0.0014, 1.4, 0.007)) + [1.4]
# values = list(np.arange(0.01, 1.4, 0.01)) + [1.4]
fname = './outputs/solution.pvd'

config = {'mesh': mesh,
          'function_space': {'family': family,
                             'degree': degree},
          'continuation': {'a': a,
                           'values': values,
                           'tangent': True,
                           'saved': True},
          'r': r,
          'nitsche': nitsche,
          'stabilized': stabilized,
          'f': as_vector([0, 0, 2.5e-2]),
          'g': as_vector([conditional(lt(x[0], 2), x[0] + a, x[0] - a), x[1], 0]),
          'phi': as_matrix([[1, 0], [0, 1], [0, 0]]),
          'sub_domain': (1, 2),
          'solver_parameters': {
              'snes_rtol': 1e-4, 'snes_atol': 1e-50,
              'snes_converged_reason': None,
              'ksp_monitor': None,
              'snes_monitor': None,
          }}

problem = IsometricBendingProblem(config)
z, y_list = problem.solve(verbose=True, fname='./outputs/solution.pvd')
y, w, p = z.subfunctions
print(abs(compute_surface_area(y) - 4.))
print(compute_isometry_defect(y))

x0 = np.linspace(0, 4, ny * 4)
x1 = np.linspace(0, 1, ny)
plot_deformation(x0, x1, y, './outputs/figures/compressive-buckling.png')
plot_deformation_anim(
    x0,
    x1,
    y_list,
    xlim=(0, 4),
    ylim=(0, 1),
    zlim=(0, 1.8),
    fname='./outputs/figures/compressive-buckling.gif')
