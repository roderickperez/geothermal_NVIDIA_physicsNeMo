import scipy.io
import numpy as np
import pandas as pd
import sys
import os
import argparse

def convert_mat_to_csv(mat_path, output_csv, num_models=None):
    """
    Converts a .mat reservoir simulation file to CSV.
    
    Args:
        mat_path (str): Path to the .mat file.
        output_csv (str): Path to save the CSV.
        num_models (int): Number of models to export (default: None = all).
    """
    print(f"Loading {mat_path}...")
    
    # Handle paths
    if not os.path.exists(mat_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, mat_path)
        if os.path.exists(potential_path):
            mat_path = potential_path
        else:
            print(f"ERROR: File not found: {mat_path}")
            return

    try:
        data = scipy.io.loadmat(mat_path)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return

    if 'INPUT' not in data:
        print("Error: 'INPUT' key not found in .mat file.")
        return

    # Shape: (N_samples, 7, 40, 40, 3)
    # 7 Features: Perm, Q, Qw, Phi, Time, Pini, Sini
    input_data = data['INPUT']
    total_models = input_data.shape[0]
    
    print(f"Dataset Dimensions: {input_data.shape}")
    
    if num_models is None or num_models > total_models:
        num_models = total_models
        print(f"Processing ALL {total_models} models.")
        print("WARNING: This will generate a very large file (~1-2 GB).")
    else:
        print(f"Processing first {num_models} models out of {total_models}.")

    # Column names mapping
    features = [
        "Permeability", 
        "Source_Q", 
        "Source_Qw", 
        "Porosity", 
        "Time_Index", 
        "Pini", 
        "Sini"
    ]

    # Pre-calculate Grid Indices
    # Grid is 40 x 40 x 3
    nx, ny, nz = 40, 40, 3
    
    # Create coordinate arrays
    # meshgrid order can be tricky, let's just stick to the array iterations to match data layout
    # The data is [channel, x, y, z] per model
    
    # Efficiently create the DataFrame using reshaping
    # We want rows: Model_ID, X, Y, Z, Feature1, Feature2...
    
    dfs = []
    
    for m in range(num_models):
        if m % 10 == 0:
            print(f"Converting Model {m}/{num_models}...")
            
        # Extract single model: (7, 40, 40, 3)
        model_cube = input_data[m]
        
        # Reshape to (7, 40*40*3) = (7, 4800)
        # We need to render the grid coordinates too.
        # Let's flatten the spatial dimensions
        flat_cube = model_cube.reshape(7, -1).T # Now (4800, 7)
        
        # Create DataFrame
        df_model = pd.DataFrame(flat_cube, columns=features)
        
        # Add Model ID
        df_model.insert(0, "Model_ID", m)
        
        # We need to generate coordinate columns X, Y, Z that match the reshape(-1) logic
        # Numpy reshape "C" order (default) goes last axis fast. 
        # So Z changes fastest, then Y, then X.
        # But we need to verify the dimension order of the input (7, 40, 40, 3)
        # Indices are [feat, x, y, z] -> Flattening x,y,z means:
        # x=0, y=0, z=0
        # x=0, y=0, z=1 
        # ...
        
        # Generating coordinates
        # using indices
        if m == 0:
            # Generate coords once
            x_idx, y_idx, z_idx = np.indices((nx, ny, nz))
            # Flatten in same order as data
            x_flat = x_idx.reshape(-1)
            y_flat = y_idx.reshape(-1)
            z_flat = z_idx.reshape(-1)
        
        df_model["X"] = x_flat
        df_model["Y"] = y_flat
        df_model["Z"] = z_flat
        
        # Reorder columns: Model_ID, X, Y, Z, ...features
        cols = ["Model_ID", "X", "Y", "Z"] + features
        df_model = df_model[cols]
        
        dfs.append(df_model)

    print("Concatenating data...")
    final_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Saving to {output_csv}...")
    final_df.to_csv(output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mat ensemble data to CSV")
    parser.add_argument("mat_path", help="Path to input .mat file")
    parser.add_argument("output_csv", help="Path to output .csv file")
    parser.add_argument("-n", "--num_models", type=int, default=5, 
                        help="Number of models to export (default: 5). Set to 0 for all.")
    
    args = parser.parse_args()
    
    n = args.num_models if args.num_models > 0 else None
    
    convert_mat_to_csv(args.mat_path, args.output_csv, n)
