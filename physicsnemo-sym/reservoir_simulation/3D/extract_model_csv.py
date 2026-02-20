import scipy.io
import numpy as np
import pandas as pd
import sys
import os

def extract_to_csv(mat_path, model_idx=0):
    if not os.path.exists(mat_path):
        # Try relative to PACKETS if needed
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(mat_path):
            mat_path = os.path.join(script_dir, mat_path)
            
    if not os.path.exists(mat_path):
        print(f"ERROR: File not found: {mat_path}")
        return

    print(f"Extracting Model {model_idx} from {mat_path}...")
    data = scipy.io.loadmat(mat_path)
    
    input_data = data['INPUT'] # (2000, 7, 40, 40, 3)
    
    properties = [
        "Permeability", "Source_Q", "Source_Qw", 
        "Porosity", "Time_Index", "Pini", "Sini"
    ]
    
    # Extract model 0, 7 properties
    model_data = input_data[model_idx] # (7, 40, 40, 3)
    
    records = []
    
    # We will iterate and create a flat list for CSV
    # Columns: Property, X, Y, Z, Value
    for p_idx, p_name in enumerate(properties):
        for x in range(model_data.shape[1]): # 40
            for y in range(model_data.shape[2]): # 40
                for z in range(model_data.shape[3]): # 3
                    val = model_data[p_idx, x, y, z]
                    records.append({
                        "Property": p_name,
                        "X": x,
                        "Y": y,
                        "Z": z,
                        "Value": val
                    })
    
    df = pd.DataFrame(records)
    csv_name = f"model_{model_idx}_input.csv"
    df.to_csv(csv_name, index=False)
    print(f"Successfully saved to {csv_name}")
    
    # Print summary to terminal
    print("\nSample Data Summary (Property averages for Model 0):")
    summary = df.groupby("Property")["Value"].mean()
    print(summary)

if __name__ == "__main__":
    mat_file = sys.argv[1] if len(sys.argv) > 1 else "PACKETS/Training4.mat"
    extract_to_csv(mat_file)
