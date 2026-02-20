import scipy.io
import numpy as np
import sys
import os

def analyze_mat_file(file_path):
    print(f"\n{'='*60}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*60}")
    
    # Try relative path from script location if necessary
    if not os.path.isabs(file_path) and not os.path.exists(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, file_path)
        if os.path.exists(potential_path):
            file_path = potential_path

    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        print("Try providing the full path or check if the PACKETS folder is in the current directory.")
        return

    try:
        data = scipy.io.loadmat(file_path)
        
        # Filter out metadata keys
        useful_keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Keys found in file: {useful_keys}")
        
        # Mapping derived from Forward_problem_PINO.py
        INPUT_FEATURES = [
            "0: Permeability (mD)",
            "1: Source Term Q (Total Flow Rate)",
            "2: Source Term Qw (Water Flow Rate)",
            "3: Porosity (fraction)",
            "4: Time Index (Normalized)",
            "5: Initial Pressure (psia)",
            "6: Initial Water Saturation (fraction)"
        ]
        
        for key in useful_keys:
            val = data[key]
            if isinstance(val, np.ndarray):
                shape = val.shape
                print(f"\nData Label: '{key}'")
                print(f"  - Shape: {shape}")
                
                # Interpret dimensions
                if len(shape) == 5:
                    # Likely (N_samples, Channels, Nx, Ny, Nz)
                    print(f"  - Structure: [Samples: {shape[0]}, Channels: {shape[1]}, X: {shape[2]}, Y: {shape[3]}, Z: {shape[4]}]")
                    
                    if key == "INPUT" and shape[1] == 7:
                        print("  - Input Features:")
                        for feature in INPUT_FEATURES:
                            print(f"    * {feature}")
                    elif key == "OUTPUT" and shape[1] == 60:
                        print("  - Output Features (30 Time-Steps):")
                        print("    * 0-29: Pressure evolution (psia)")
                        print("    * 30-59: Water Saturation evolution (fraction)")
                
                print(f"  - Total Ensemble Models: {shape[0]}")
            else:
                print(f"\nData Label: '{key}' (Not an array, type: {type(val)})")

    except Exception as e:
        print(f"ERROR: Failed to load .mat file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Default to the training file in the PACKETS folder relative to the script location or common path
        target_path = "PACKETS/Training4.mat"
        
    analyze_mat_file(target_path)
