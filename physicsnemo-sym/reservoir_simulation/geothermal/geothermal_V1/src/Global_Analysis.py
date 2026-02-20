import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    results_dir = "COMPARE_RESULTS/Detailed"
    models = {
        "FNO": os.path.join(results_dir, "FNO"),
        "PINO Base": os.path.join(results_dir, "PINO/PINO_base"),
        "PINO CL": os.path.join(results_dir, "PINO/PINO_CL"),
        "PINO RL": os.path.join(results_dir, "PINO/PINO_RL")
    }
    
    os.makedirs("COMPARE_RESULTS/Global", exist_ok=True)
    
    # 1. Bar Chart (RMSE summary at last timestep)
    rmses = []
    for name, path in models.items():
        if os.path.exists(os.path.join(path, "RSM_MODULUS.csv")):
            df_mod = pd.read_csv(os.path.join(path, "RSM_MODULUS.csv"))
            df_num = pd.read_csv(os.path.join(path, "RSM_NUMERICAL.csv"))
            # Calculate RMSE over all wells for the last step
            rmse = np.sqrt(np.mean((df_mod.iloc[-1, 1:] - df_num.iloc[-1, 1:])**2))
            rmses.append(rmse)
        else:
            rmses.append(0)

    plt.figure(figsize=(10, 6))
    plt.bar(models.keys(), rmses, color=['orange', 'blue', 'green', 'red'])
    plt.ylabel("RMSE at Final Timestep")
    plt.title("Model Performance Comparison")
    plt.savefig("COMPARE_RESULTS/Global/Bar_chat.png")
    plt.close()

    # 2. Production Profile Comparison (Example: Well P1 Temperature)
    plt.figure(figsize=(12, 8))
    # First plot Numerical (Truth) - assuming identical for all since it's the same dataset
    ref_path = os.path.join(models["FNO"], "RSM_NUMERICAL.csv")
    if os.path.exists(ref_path):
        df_num = pd.read_csv(ref_path)
        plt.plot(df_num["Time(DAY)"], df_num["P1 - T(C)"], 'k--', label="Numerical (Truth)", linewidth=3)
        
    for name, path in models.items():
        mod_path = os.path.join(path, "RSM_MODULUS.csv")
        if os.path.exists(mod_path):
            df_mod = pd.read_csv(mod_path)
            plt.plot(df_mod["Time(DAY)"], df_mod["P1 - T(C)"], label=f"{name} (Pred)")
            
    plt.xlabel("Time (Days)")
    plt.ylabel("Temperature (C) at P1")
    plt.title("Production Profile: Well P1 Temperature Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("COMPARE_RESULTS/Global/Compare_models.png")
    plt.close()
    
    print("Saved Global summaries to COMPARE_RESULTS/Global/")

if __name__ == "__main__":
    main()
