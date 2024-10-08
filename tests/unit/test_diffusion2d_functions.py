"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest
import numpy as np

def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(
        w=5.,
        h=4.,
        dx=0.2,
        dy=0.5
    )
    
    # check assigned values
    assert solver.w == 5.
    assert solver.h == 4.
    assert solver.dx == 0.2
    assert solver.dy == 0.5
    
    # check computed nx, ny
    assert solver.nx == 25
    assert solver.ny == 8


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_physical_parameters
    """
    solver = SolveDiffusion2D()
    solver.w = 5.
    solver.h = 5.
    solver.dx = 0.2
    solver.dy = 0.5
    
    solver.nx = 25
    solver.ny = 10
    
    solver.initialize_physical_parameters(
        d=5.,
        T_cold=-300.5,
        T_hot=800.
    )
    
    # check assigned values
    assert solver.D == 5.
    assert solver.T_cold == -300.5
    assert solver.T_hot == 800.
    
    # check dt
    print(solver.dt)
    expected_dt = 0.003448275
    pytest.approx(solver.dt, expected_dt, 1e-6)


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    
    solver.w = 5.
    solver.h = 5.
    solver.dx = 0.2
    solver.dy = 0.5
    
    solver.nx = 25
    solver.ny = 10
    
    solver.D = 5.
    solver.T_cold = -300.5
    solver.T_hot = 800.
    
    u = solver.set_initial_condition()
    
    # create expected shape
    expected_u = np.ones((25, 10)) * -300.5
    r = min(solver.h, solver.w) / 4.0
    cx = solver.w / 2.0
    cy = solver.h / 2.0
    r2 = r ** 2
    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                expected_u[i, j] = solver.T_hot
                    
    # check values
    assert np.allclose(u, expected_u)
