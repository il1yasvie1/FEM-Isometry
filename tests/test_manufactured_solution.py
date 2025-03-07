from isometric_bending_solver.problem import IsometricBendingProblem
from isometric_bending_solver.utils import compute_surface_area, compute_isometry_defect
from firedrake import *
import numpy as np
import pytest


@pytest.mark.parametrize('nx', [4, 8, 16])
@pytest.mark.parametrize('family, nitsche, degree',
                         [('CG', False, 2), ('CG', True, 2), ('DG', True, 2)])
@pytest.mark.parametrize('isRegularised', [{'beta': 1e-3}, False])
def test_manufactured_solution(nx, isRegularised, family, nitsche, degree):
    mesh = UnitSquareMesh(nx, nx)
    x = SpatialCoordinate(mesh)
    theta = np.pi / 4
    config = {'mesh': mesh,
              'function_space': {'family': family,
                                 'degree': degree},
              'continuation': False,
              'nitsche': nitsche,
              'isRegularised': isRegularised,
              'f': as_vector([(theta**3) * sin(theta * x[0]), 0, (-1) * (theta**3) * cos(theta * x[0])]),
              'g': as_vector([sin(theta * x[0]) / theta, x[1], 1 - cos(theta * x[0]) / theta]),
              'phi': as_matrix([[cos(theta * x[0]), 0], [0, 1], [sin(theta * x[0]), 0]]),
              'sub_domain': (1, 2, 3, 4),
              'solver_parameters': {
                  'snes_rtol': 1e-7, 'snes_atol': 1e-10,
                  'pc_type': 'lu',
                  'pc_factor_mat_solver_type': 'mumps',
                  'snes_type': 'newtonls',
                  'ksp_type': 'preonly',
              }}
    problem = IsometricBendingProblem(config)
    z = problem.solve()
    y, w, p = z.subfunctions
    assert abs(compute_surface_area(y) - 1.0) < 1e-2
    assert compute_isometry_defect(y) < 1e-2
    assert norm(y - problem.g) < 1e-3


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
