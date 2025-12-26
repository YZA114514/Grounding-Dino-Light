#!/usr/bin/env python3
"""
Debug Jittor argmax behavior
"""

import sys
sys.path.append('/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor')

import jittor as jt
import numpy as np

def debug_argmax():
    print("Debugging Jittor argmax behavior...")
    
    # Create a simple test tensor
    test_data = np.array([
        [0.1, 0.9, 0.2],  # Should argmax to index 1
        [0.8, 0.3, 0.7],  # Should argmax to index 0  
        [0.2, 0.4, 0.9],  # Should argmax to index 2
    ], dtype=np.float32)
    
    test_tensor = jt.array(test_data)
    print(f"Test tensor shape: {test_tensor.shape}")
    print(f"Test tensor dtype: {test_tensor.dtype}")
    print(f"Test tensor contents:\n{test_tensor.numpy()}")
    
    # Test argmax
    print("\n--- Testing jt.argmax ---")
    argmax_result = jt.argmax(test_tensor, dim=-1)
    print(f"argmax result: {argmax_result}")
    print(f"argmax type: {type(argmax_result)}")
    
    # Check if it's a tuple
    if isinstance(argmax_result, tuple):
        print(f"argmax tuple length: {len(argmax_result)}")
        print(f"First element (values): {argmax_result[0].numpy()}")
        print(f"Second element (indices): {argmax_result[1].numpy()}")
        print(f"Values shape: {argmax_result[0].shape}")
        print(f"Indices shape: {argmax_result[1].shape}")
        print(f"Values type: {type(argmax_result[0])}")
        print(f"Indices type: {type(argmax_result[1])}")
        print(f"Values: {argmax_result[0].numpy()}")
        print(f"Indices: {argmax_result[1].numpy()}")
        
        # Try to convert indices to numpy
        indices_np = argmax_result[1].numpy()
        print(f"Indices as numpy: {indices_np}")
        print(f"Indices dtype: {indices_np.dtype}")
    else:
        print("argmax returned a single tensor (not tuple)")
        print(f"argmax shape: {argmax_result.shape}")
        print(f"argmax numpy: {argmax_result.numpy()}")
    
    # Test with max too
    print("\n--- Testing jt.max ---")
    max_result = jt.max(test_tensor, dim=-1)
    print(f"max result: (values, indices)")
    print(f"max type: {type(max_result)}")
    
    if isinstance(max_result, tuple):
        print(f"max tuple length: {len(max_result)}")
        print(f"Max values: {max_result[0].numpy()}")
        print(f"Max indices: {max_result[1].numpy()}")
    else:
        print("max returned a single tensor (not tuple)")
        print(f"max shape: {max_result.shape}")
        print(f"max numpy: {max_result.numpy()}")

if __name__ == "__main__":
    debug_argmax()
