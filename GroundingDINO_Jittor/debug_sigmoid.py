#!/usr/bin/env python3
"""
Debug sigmoid issue specifically
"""

import jittor as jt
import numpy as np

def test_sigmoid():
    print("Testing sigmoid function...")
    
    # Test with normal values
    x_normal = jt.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    sigmoid_normal = jt.sigmoid(x_normal)
    print(f"Normal input: {x_normal}")
    print(f"Normal sigmoid: {sigmoid_normal}")
    print(f"Expected: [[0.119, 0.269, 0.500, 0.731, 0.881]]")
    
    # Test with extreme values
    x_extreme = jt.array([[-10.0, -1e9, 1e9, 10.0]])
    sigmoid_extreme = jt.sigmoid(x_extreme)
    print(f"\nExtreme input: {x_extreme}")
    print(f"Extreme sigmoid: {sigmoid_extreme}")
    print(f"Expected: [[~0.000, ~0.000, ~1.000, ~1.000]]")
    
    # Test with clamping
    x_clamped = jt.clamp(x_extreme, -10.0, 10.0)
    sigmoid_clamped = jt.sigmoid(x_clamped)
    print(f"\nClamped input: {x_clamped}")
    print(f"Clamped sigmoid: {sigmoid_clamped}")
    
    # Test specific problematic values
    x_problem = jt.array([[-1e9, 8.5]])
    sigmoid_problem = jt.sigmoid(x_problem)
    print(f"\nProblem input: {x_problem}")
    print(f"Problem sigmoid: {sigmoid_problem}")
    
    # Try manual sigmoid
    def manual_sigmoid(x):
        return 1.0 / (1.0 + jt.exp(-x))
    
    sigmoid_manual = manual_sigmoid(x_problem)
    print(f"Manual sigmoid: {sigmoid_manual}")

if __name__ == "__main__":
    test_sigmoid()
