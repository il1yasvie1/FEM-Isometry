from isometric_bending_solver.problem import IsometricBendingProblem
from isometric_bending_solver.utils import compute_surface_area, compute_isometry_defect, plot_deformation
from firedrake import *
import numpy as np


nx = 16
family, degree = ('CG', 2)
mesh = UnitSquareMesh(nx, nx)
x = SpatialCoordinate(mesh)

config = {'mesh': mesh,
          'function_space': {'family': family,
                             'degree': degree},
          'f': as_vector([0, 0, 2.5e-4]),
          'g': as_vector([x[0], x[1], 0]),
          'phi': as_matrix([[1, 0], [0, 1], [0, 0]]),
          'sub_domain': (1, 3),
          'solver_parameters': {}
          }

problem = IsometricBendingProblem(config)
z = problem.solve()
y, w, p = z.subfunctions
print(abs(compute_surface_area(y) - 1.0))
print(compute_isometry_defect(y))

x0 = np.linspace(0, 1, nx)
x1 = np.linspace(0, 1, nx)
plot_deformation(x0, x1, y, './outputs/figures/vertical-load-square.png')
