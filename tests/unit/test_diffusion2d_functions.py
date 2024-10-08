"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest
import numpy as np

from unittest import TestCase

class TestDiffusion2DFunctions(TestCase):
    
    def setUp(self):
        # create an instance of SolveDiffusion2D
        self.solver = SolveDiffusion2D()
        
        # define fixture values
        self.w = 5.
        self.h = 4.
        
        self.dx = 0.2
        self.dy = 0.5
        
        self.D = 5.
        
        self.T_cold = -300.5
        self.T_hot = 800.
    
    def test_initialize_domain(self):
        """
        Check function SolveDiffusion2D.initialize_domain
        """
        self.solver.initialize_domain(
            w=self.w,
            h=self.h,
            dx=self.dx,
            dy=self.dy
        )
        
        # check assigned values
        assert self.solver.w == self.w
        assert self.solver.h == self.h
        assert self.solver.dx == self.dx
        assert self.solver.dy == self.dy
        
        # check nx and ny
        assert self.solver.nx == 25
        assert self.solver.ny == 8


    def test_initialize_physical_parameters(self):
        """
        Checks function SolveDiffusion2D.initialize_physical_parameters
        """
        self.solver.w = self.w
        self.solver.h = self.h
        self.solver.dx = self.dx
        self.solver.dy = self.dy
        
        self.solver.initialize_physical_parameters(
            d=self.D,
            T_cold=self.T_cold,
            T_hot=self.T_hot
        )
        
        # check assigned values
        assert self.solver.D == self.D
        assert self.solver.T_cold == self.T_cold
        assert self.solver.T_hot == self.T_hot
        
        # check dt      
        expected_dt = 0.003448275
        pytest.approx(self.solver.dt, expected_dt, 1e-6)


    def test_set_initial_condition(self):
        """
        Checks function SolveDiffusion2D.get_initial_function
        """
        self.solver.w = self.w
        self.solver.h = self.h
        self.solver.dx = self.dx
        self.solver.dy = self.dy
        
        self.solver.T_cold = self.T_cold
        self.solver.T_hot = self.T_hot
        
        self.solver.nx = 25
        self.solver.ny = 10
        
        u = self.solver.set_initial_condition()
        
        # create expected shape
        expected_u = np.ones((25, 10)) * -300.5
        r = min(self.solver.h, self.solver.w) / 4.0
        cx = self.solver.w / 2.0
        cy = self.solver.h / 2.0
        r2 = r ** 2
        for i in range(self.solver.nx):
            for j in range(self.solver.ny):
                p2 = (i * self.solver.dx - cx) ** 2 + (j * self.solver.dy - cy) ** 2
                if p2 < r2:
                    expected_u[i, j] = self.solver.T_hot
                    
        # check values
        assert np.allclose(u, expected_u)
