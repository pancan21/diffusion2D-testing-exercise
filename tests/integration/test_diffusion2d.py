"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest
import numpy as np

def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    w = 5.
    h = 4.
    
    dx = 0.2
    dy = 0.5
    
    D = 5.
    
    T_cold = -300.5
    T_hot = 800.
    
    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    solver.initialize_physical_parameters(d=D, T_cold=T_cold, T_hot=T_hot)
    
    expected_dt = 0.003448275
    pytest.approx(solver.dt, expected_dt)


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    w = 5.
    h = 4.
    
    dx = 0.2
    dy = 0.5
    
    D = 5.
    
    T_cold = -300.5
    T_hot = 800.
    
    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    solver.initialize_physical_parameters(d=D, T_cold=T_cold, T_hot=T_hot)
    
    u = solver.set_initial_condition()
        
    # create expected shape
    expected_u = np.ones((solver.nx, solver.ny)) * T_cold
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
