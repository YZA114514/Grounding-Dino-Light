
import pickle
import sys

def check_pickle():
    print("Checking label_enc.weight in pickle...")
    try:
        with open('weights/groundingdino_swint_jittor.pkl', 'rb') as f:
            data = pickle.load(f)
            if 'label_enc.weight' in data:
                print(f"Pickle file 'label_enc.weight' shape: {data['label_enc.weight'].shape}")
            else:
                print("Pickle file does NOT contain 'label_enc.weight'")
            
            # Also check for other label_enc related keys
            keys = [k for k in data.keys() if 'label_enc' in k]
            if keys:
                print(f"Found other label_enc keys: {keys}")
            else:
                print("No keys containing 'label_enc' found.")
                
    except Exception as e:
        print(f"Error reading pickle: {e}")

if __name__ == "__main__":
    check_pickle()
