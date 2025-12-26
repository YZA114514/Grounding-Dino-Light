import pickle
import numpy as np
import sys

def inspect_weights():
    try:
        with open('weights/groundingdino_swint_jittor.pkl', 'rb') as f:
            state_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    print(f"Type of loaded data: {type(state_dict)}")
    print("Keys:")
    for k in state_dict.keys():
        print(k)

if __name__ == "__main__":
    inspect_weights()
