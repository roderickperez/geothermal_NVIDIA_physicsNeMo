import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from PIL import Image
import glob
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import FNOArch, ConvFullyConnectedArch
from utilities import load_FNO_dataset2, to_absolute_path

def compute_metrics(true, pred):
    true = true.flatten()
    pred = pred.flatten()
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    l2 = np.sqrt(np.mean((true - pred) ** 2)) / (np.sqrt(np.mean(true**2)) + 1e-10)
    return r2 * 100, l2 * 100

def extract_geothermal_wells(q_tensor):
    q_sum = np.sum(q_tensor, axis=2) # Sum over Z layers to find surface loc
    inj_loc = np.unravel_index(np.argmax(q_sum), q_sum.shape)
    prod_loc = np.unravel_index(np.argmin(q_sum), q_sum.shape)
    # Ensure they are valid (non-zero Q)
    if q_sum[inj_loc] <= 0: inj_loc = None
    if q_sum[prod_loc] >= 0: prod_loc = None
    return inj_loc, prod_loc

def generate_evolution_frame(p_truth, p_pred, t_truth, t_pred, timestep, day, out_path, inj_loc, prod_loc):
    fig = plt.figure(figsize=(15, 10), dpi=100)
    
    # Grid for plotting (X and Y dimensions)
    nx, ny = p_truth.shape[0], p_truth.shape[1]
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    
    # Layer to visualize (Middle layer)
    layer = 2
    
    # Row 1: Pressure
    vars_p = [p_pred[:, :, layer].T, p_truth[:, :, layer].T, np.abs(p_pred[:, :, layer] - p_truth[:, :, layer]).T]
    titles_p = ["Pressure Pred", "Pressure Truth", "Pressure Diff"]
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+1)
        
        if i == 2: # Diff specific bounded scale
            im = ax.pcolormesh(XX, YY, vars_p[i], cmap='YlOrRd', vmin=0, vmax=30, shading='auto')
        else: # Main field bounded scale
            im = ax.pcolormesh(XX, YY, vars_p[i], cmap='viridis', shading='auto')
            
        ax.set_title(titles_p[i])
        plt.colorbar(im, ax=ax)
        if inj_loc: ax.scatter(inj_loc[0], inj_loc[1], c='blue', marker='o', s=100, edgecolors='white', label='INJ')
        if prod_loc: ax.scatter(prod_loc[0], prod_loc[1], c='red', marker='X', s=100, edgecolors='white', label='PROD')
        if i == 0: ax.legend(loc='upper right')

    # Row 2: Temperature
    vars_t = [t_pred[:, :, layer].T, t_truth[:, :, layer].T, np.abs(t_pred[:, :, layer] - t_truth[:, :, layer]).T]
    titles_t = ["Temp Pred", "Temp Truth", "Temp Diff"]
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+4)
        
        if i == 2: # Diff specific bounded scale
            im = ax.pcolormesh(XX, YY, vars_t[i], cmap='YlOrRd', vmin=0, vmax=25, shading='auto')
        else: # Main field bounded scale
            im = ax.pcolormesh(XX, YY, vars_t[i], cmap='viridis', shading='auto')
            
        ax.set_title(titles_t[i])
        plt.colorbar(im, ax=ax)
        if inj_loc: ax.scatter(inj_loc[0], inj_loc[1], c='blue', marker='o', s=100, edgecolors='white')
        if prod_loc: ax.scatter(prod_loc[0], prod_loc[1], c='red', marker='X', s=100, edgecolors='white')

    plt.suptitle(f"Day {int(day)}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model_idx", type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    output_keys_p = ["pressure"]
    output_keys_t = ["temperature"]
    
    data_path = args.data
    if data_path.endswith(".mat"):
        from utilities import preprocess_FNO_mat
        preprocess_FNO_mat(data_path)
        data_path = data_path.replace(".mat", ".hdf5")
        
    invar, out_p_truth, out_t_truth = load_FNO_dataset2(
        data_path, input_keys, output_keys_p, output_keys_t, n_examples=None
    )
    
    # 2. Setup Model
    steppi = out_p_truth["pressure"].shape[1]
    decoder_p = ConvFullyConnectedArch([Key("z", size=32)], [Key("pressure", size=steppi)])
    model_p = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)
    
    decoder_t = ConvFullyConnectedArch([Key("z", size=32)], [Key("temperature", size=steppi)])
    model_t = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_t).to(device)
    
    # Load weights (robust logic)
    base_dir = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.dirname(args.checkpoint)
    for pref in ["pino_forward_model", "fno_forward_model"]:
        p_path = os.path.join(base_dir, f"{pref}_pressure.0.pth")
        t_path = os.path.join(base_dir, f"{pref}_temperature.0.pth")
        if os.path.exists(p_path) and os.path.exists(t_path):
            model_p.load_state_dict(torch.load(p_path, map_location=device, weights_only=False))
            model_t.load_state_dict(torch.load(t_path, map_location=device, weights_only=False))
            print(f"Loaded {pref} weights.")
            break

    # 3. Inference
    idx = args.model_idx
    test_in = {k: torch.from_numpy(v[idx:idx+1]).to(device) for k, v in invar.items()}
    with torch.no_grad():
        pred_p = model_p(test_in)["pressure"].cpu().numpy()[0] # [T, Z, X, Y]
        pred_t = model_t(test_in)["temperature"].cpu().numpy()[0]
    
    true_p = out_p_truth["pressure"][idx]
    true_t = out_t_truth["temperature"][idx]
    
    # 4. Extract Dynamic Wells
    q_tensor_test = test_in["Q"].cpu().numpy()[0, 0] # Extract the base Q tensor [X,Y,Z] 
    inj_loc, prod_loc = extract_geothermal_wells(q_tensor_test)
    print(f"Detected Wells -> Injector: {inj_loc}, Producer: {prod_loc}")

    # 5. Metrics & GIF
    print("Generating Evolution GIF and Metrics...")
    
    metrics_p = []
    metrics_t = []
    frames = []
    
    # [START OF CORRECTION]
    for t in range(steppi):
        day = (t + 1) * 365.0
        frame_path = os.path.join(args.out, f"frame_{t:03d}.png")
        
        # 1. UN-NORMALIZE (Convert 0-1 back to Bar/Celsius)
        # Pressure: x * 200 + 100
        p_pred_phys = pred_p[t] * 200.0 + 100.0
        p_truth_phys = true_p[t] * 200.0 + 100.0
        
        # Temperature: Normalized output is exactly T_Celsius / 100
        t_pred_phys = pred_t[t] * 100.0 
        t_truth_phys = true_t[t] * 100.0
        
        # 2. GENERATE FRAME WITH PHYSICAL VALUES
        generate_evolution_frame(p_truth_phys, p_pred_phys, t_truth_phys, t_pred_phys, t, day, frame_path, inj_loc, prod_loc)
        frames.append(Image.open(frame_path))
        
        # 3. COMPUTE METRICS (Optional: Keep normalized for standard comparison, or use phys)
        rp, lp = compute_metrics(true_p[t], pred_p[t])
        rt, lt = compute_metrics(true_t[t], pred_t[t])
        metrics_p.append([rp, lp])
        metrics_t.append([rt, lt])
    # [END OF CORRECTION]

    # Save GIF
    frames[0].save(os.path.join(args.out, "Evolution.gif"), save_all=True, append_images=frames[1:], duration=200, loop=0)
    for f in glob.glob(os.path.join(args.out, "frame_*.png")): os.remove(f)
    print(f"Saved Evolution.gif to {args.out}")

    # 5. R2L2 Plot
    metrics_p = np.array(metrics_p)
    metrics_t = np.array(metrics_t)
    days = [(t + 1) * 365.0 for t in range(steppi)]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(days, metrics_p[:, 0], label='P R2', marker='o')
    plt.plot(days, metrics_t[:, 0], label='T R2', marker='x')
    plt.title("R2 Accuracy (%)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(days, metrics_p[:, 1], label='P L2', marker='o')
    plt.plot(days, metrics_t[:, 1], label='T L2', marker='x')
    plt.title("L2 Error (%)")
    plt.legend()
    plt.savefig(os.path.join(args.out, "R2L2.png"))
    plt.close()

    # 6. RSM CSV
    
    # We will log the Injector and Producer dynamically
    cols = ["Time(DAY)"]
    if inj_loc:
        cols.extend([f"INJ_P(Bar)", f"INJ_T(C)"])
    if prod_loc:
        cols.extend([f"PROD_P(Bar)", f"PROD_T(C)"])
        
    data_mod = []
    data_num = []
    for t in range(steppi):
        row_mod = [days[t]]
        row_num = [days[t]]
        
        # Extract at middle layer (index 2) for well reporting
        if inj_loc:
            wx, wy = inj_loc
            row_mod.extend([pred_p[t, wx, wy, 2] * 200.0 + 100.0, pred_t[t, wx, wy, 2] * 100.0])
            row_num.extend([true_p[t, wx, wy, 2] * 200.0 + 100.0, true_t[t, wx, wy, 2] * 100.0])
            
        if prod_loc:
            wx, wy = prod_loc
            row_mod.extend([pred_p[t, wx, wy, 2] * 200.0 + 100.0, pred_t[t, wx, wy, 2] * 100.0])
            row_num.extend([true_p[t, wx, wy, 2] * 200.0 + 100.0, true_t[t, wx, wy, 2] * 100.0])
            
        data_mod.append(row_mod)
        data_num.append(row_num)
        
    pd.DataFrame(data_mod, columns=cols).to_csv(os.path.join(args.out, "RSM_MODULUS.csv"), index=False)
    pd.DataFrame(data_num, columns=cols).to_csv(os.path.join(args.out, "RSM_NUMERICAL.csv"), index=False)
    print(f"Saved RSM tables to {args.out}")

if __name__ == "__main__":
    main()
