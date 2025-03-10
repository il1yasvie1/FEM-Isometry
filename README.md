# Isometric-Bending-Solver
A Firedrake-based numerical solver for the isometric bending deformation on the Kirchhoff thin plate model.

**Author:** Sanyang Liu

**Supervisor:** Prof. Colin Cotter

> ⚠️ **Warning:** This project may be part of unpublished work. Please reference it appropriately if used. The code is currently under development and is NOT finished yet. Some features may be incomplete or subject to changes. Contributions and feedback are welcome!

## Contents
- [Installation and Dependency](#installation-and-dependency)
- [Overview](#overview)
- [Implementation](#implementation)
- [5-min Tutorial](#5-min-tutorial)
- [References](#references)

## Installation and Dependency

You could directly download the package from Github source:
```bash
git clone https://github.com/il1yasvie1/FEM-Isometry.git; cd FEM-Isometry
pip install .
```

Or you could install and test the package in Colab by the code block:
```bash
!pip install git+https://github.com/il1yasvie1/FEM-Isometry.git
from isometric_bending_solver import *
```

This project is mainly dependent on the following repository:
- [Firedrake](https://www.firedrakeproject.org/) for solving partial differential equations.

## Overview
This project focuses on developing a numerical solver for the following minimisation problem ([Bonito et al. (2020)](#bonito2020), [Friesecke et al. (2002)](#friesecke2002)) :
```math
\begin{aligned}
\min_{y \in [H^2(\Omega)]^3}  E[y]:= & \frac{1}{2} \int_\Omega \|D^2 y \|^2_F  - \int_\Omega f \cdot y \\
\text{s.t.} & \nabla y^T \nabla y = I \quad \text{a.e. in } \Omega, \\
            & y|_{\partial \Omega_D} = g, \\
            & \nabla y|_{\partial \Omega_D} = \Phi
\end{aligned}
```
where $`\Omega \subset \mathbb{R}^2 `$, $` g \in [H^1(\Omega)]^3 `$ and $` \Phi \in [H^1(\Omega)]^{3\times 2} `$ are the Dirichlet data on part of the boundary $`\partial \Omega_D`$ of $`\Omega`$ and $`f \in [L^2(\Omega)]^3`$ is the forcing data. Notice that there is a nonlinear isometry constraint requiring the Jacobian of the deformation field to have orthonormal columns. In this work, we mainly introduce a novel approach to enforcing this constraint using the exponential map of a skew-symmetric matrix which generates an orthonormal matrix. To adapt this approach, an appropriate mixed formulation using interior penalty methods or higher order elements) should be considered. There are mainly 3 challenges for this project:

1. The fourth-order biharmonic equation $`\Delta^2 y = f`$.
2. The isometry constraint implemented by the exponential map and a skew-symmetrize map.
3. The numerical continuation method to solve the large isometric bending or difficult problem.

## Implementation
In [Bonito et al. (2020)](#bonito2020), the author(s) derived that the strong form of the Euler-Lagrange equation on the minimizer of the energy functional satisfies a biharmonic equation. There are 4 classical methods to solve the biharmonic problem ([Brenner (2011)](#brenner2011)) and some of them are used in isometric bending problem:

1. conforming method, e.g. Argyris, Bell elements.
2. nonconforming method, e.g. quadratic Morley elements.
3. interior penalty methods, e.g. $`C^0`$ or Discontinuous Lagrange elements.
4. mixed formulation on two Poisson's equations. We have not explored this approach.

We mainly develop the theory for $`C^0`$-interior penalty method.

## 5-min Tutorial
Here we use the example from `examples/mobius.py` to show the whole process of solving. You could also directly test this example with default setting by
```bash
python -m examples.mobius
```

### Theory
Suppose we want to find the shape of the Möbius strip at the equilibrium state from a rectangular strip $`\Omega = [0, 2\pi]\times [0, 2l]`$ (here we only consider an 'easy-bending strip', i.e. $`l \ll \pi`$) by two steps:

1. Clamp $`x_1=0`$, bend the strip along its lengthwise direction so that $`x_1=0`$ and $`x_1=2\pi`$ are brought together and connected, forming a cylindrical surface.
2. Clamp $`x_1=2\pi`$, rotating the other end $`x_1=0`$ about the axis $x_2 = l$ clockwise (using right-hand rule) from $`\theta \in [0, 2\pi] `$.

Notice that we can solve the first step explicitly and exactly. Therefore we can then set the exact solution as the initial guess for the second step. The exact solution for the first step is:
```math
y = [\sin(x_1), x_2, 1 - \cos(x_1)], \quad w = [0, -x_1, 0], \quad p= 0_{3\times 2}
```
Then for the second step, we have the Dirichlet data on $`\partial \Omega_D := \{0\} \times [0, 2l] \cup \{2\pi\} \times [0, 2l]`$ and the forcing data:
```math
\begin{aligned}
g|_{x_1=0} = [0, l + (x_2 - l)\cos(a), (x_2 - l)\sin(a)] , \quad & g|_{x_1=2\pi} = [0, x_2, 0]\\
\Phi |_{x_1=0} = \begin{bmatrix}
    1 & 0 & 0\\
    0 & \cos(a) &\sin(a)
\end{bmatrix}^T ,\quad &
\Phi|_{x_1=2\pi} =\begin{bmatrix}
    1 & 0 & 0\\
    0 & 1 & 0
\end{bmatrix}^T
\end{aligned}
```
```math
f = [\sin(x_1), 0, -\cos(x_1)]
```
where `a` is the continuation parameter from $`0`$ to $`\pi`$.

### Code
Here we use `isometric-bending-solver` to build and solve the problem. The main functionalities are integrated into the class `IsometricBendingProblem` where all parameters are passed via a `config` dictionary. Here are config options:

- `mesh`: 
- `function_space`: use the most stable $`C^0`$-interior penalty formulation with `degree = d`
- `continuation`: we first define the continuation parameter `a` and set `range` for it as $`(0, \pi)`$ with an appropriate `step_size` for continuation. Moreover, set `tangent` as `True` to use tangent continuation method. To visualise the continuation process, set `saved` to `True`.
- `g`, `phi`, `f`, `sub_domain`: set functions as their `ufl_expr`s and `sub_domain` as `(1, 2)` for selecting the corresponding part of boundary. Notice that the continuation parameter `a` should be explicitly appear in the definition of expressions.
-  `initial_guess`: exact solution to the cylindrical surface.
-  `stabilized`: turn it on to make solver converged more easily.
-  `nitsche`: not necessary when using $`C^0`$ interior penalty method. Hence we set it as `False` or exclude it from `config`.
-  `solver_parameters`: the interface for the parameters setting for PETSc.

Here is the full definition for `config`:
```python
nx = 8
family, degree = ('CG', 2)
  mesh = RectangleMesh(nx, ny, 2*np.pi, 2*l)
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
          'nitsche': False,
          'f': as_vector([sin(x[0]), 0, -cos(x[0])]),
          'g': as_vector([sin(x[0]),
                          conditional(lt(x[0], np.pi), 0.5 + (x[1] - 0.5) * cos(a), x[1]),
                          conditional(lt(x[0], np.pi), (x[1] - 0.5) * sin(a), 0)]),
          'phi': as_matrix([[1, 0],
                            [0, conditional(lt(x[0], np.pi), cos(a), 1)],
                            [0, conditional(lt(x[0], np.pi), sin(a), 0)]]),
          'sub_domain': (1, 2),
          'initial_guess': {'y0': as_vector([sin(x[0]), x[1], 1 - cos(x[0])]),
                            'w0': as_vector([0, -x[0], 0])},
          'solver_parameters': {},
          }
```

After we fully define the dictionary `config`, we could build and solve the problem as the following:
```python
problem = IsometricBendingProblem(config)
problem.solve()
```

### Visuals
The visualisation tools are implemented as `plot_deformation` or `plot_deformation_anim`. Saving results to the `.pvd` file is also avaliable. Here are the visuals for the Möbius strip:
![The equilibrium state](/examples/figures/mobius.png)
![Continuation mobius](/examples/figures/mobius.gif)

## References
- <a id="bonito2020"></a> Bonito, A., Nochetto, R. H., & Ntogkas, D. (2020). *DG Approach to Large Bending Plate Deformations with Isometry Constraint*. arXiv preprint, [arXiv:1912.03812](https://arxiv.org/abs/1912.03812).
- <a id="friesecke2002"></a> Friesecke, G., Müller, S., & James, R. D. (2002). Rigorous derivation of nonlinear plate theory and geometric rigidity. *Comptes Rendus Mathematique*, 334(2), 173–178. https://doi.org/10.1016/S1631-073X(02)02133-7
- <a id="brenner2011"></a> Brenner, S.C. (2011). *C₀ Interior Penalty Methods*. In: Blowey, J., Jensen, M. (eds) Frontiers in Numerical Analysis - Durham 2010. Lecture Notes in Computational Science and Engineering, vol 85. Springer, Berlin, Heidelberg. [https://doi.org/10.1007/978-3-642-23914-4_2](https://doi.org/10.1007/978-3-642-23914-4_2)
