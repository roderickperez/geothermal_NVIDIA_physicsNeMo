import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

# PhysicsNeMo imports
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import FNOArch, ConvFullyConnectedArch
from utilities import load_FNO_dataset2

# ==========================================
# 1. CONFIGURATION
# ==========================================

def extract_geothermal_wells(q_tensor):
    q_sum = np.sum(q_tensor, axis=2) # Sum over Z layers to find surface loc
    inj_loc = np.unravel_index(np.argmax(q_sum), q_sum.shape)
    prod_loc = np.unravel_index(np.argmin(q_sum), q_sum.shape)
    # Ensure they are valid (non-zero Q)
    if q_sum[inj_loc] <= 0: inj_loc = None
    if q_sum[prod_loc] >= 0: prod_loc = None
    return inj_loc, prod_loc

# Model Paths
MODELS_CONFIG = {
    "FNO":       {"checkpoint": "outputs/Forward_problem_FNO/ResSim", "tag": "FNO"},
    "PINO_Base": {"checkpoint": "outputs/Forward_problem_PINO/ResSim", "tag": "PINO_Base"},
    "PINO_CL":   {"checkpoint": "outputs/Forward_problem_PINO_CL/ResSim_M3_Curriculum", "tag": "PINO_CL"},
    "PINO_RL":   {"checkpoint": "outputs/Forward_problem_PINO_RL/ResSim_M4_Reverse", "tag": "PINO_RL"}
}

# Dataset Statistics (From your Logs) to fix the "Wrong Colors" bug
DATA_STATS = {
    "Training": {
        "p_mean": 155.66, "p_std": 26.50,
        "t_mean": 349.41, "t_std": 30.47,
        "k_mean": 677.37, "k_std": 724.16,
        "phi_mean": 0.195, "phi_std": 0.063
    },
    "Test": {
        "p_mean": 153.57, "p_std": 28.14,
        "t_mean": 349.01, "t_std": 30.68,
        "k_mean": 708.81, "k_std": 744.18,
        "phi_mean": 0.198, "phi_std": 0.065
    }
}

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================

def get_stats_for_file(filepath):
    """Auto-selects correct stats based on filename."""
    if "Training" in filepath:
        print(f"[INFO] Using TRAINING stats for {os.path.basename(filepath)}")
        return DATA_STATS["Training"]
    else:
        print(f"[INFO] Using TEST stats for {os.path.basename(filepath)}")
        return DATA_STATS["Test"]

def load_dataset(data_path):
    print(f"\nLoading Data from {data_path}...")
    if data_path.endswith(".mat"):
        from utilities import preprocess_FNO_mat
        preprocess_FNO_mat(data_path)
        data_path = data_path.replace(".mat", ".hdf5")

    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    inv, out_p, out_t = load_FNO_dataset2(data_path, input_keys, ["pressure"], ["temperature"], n_examples=None)
    return inv, out_p, out_t

def load_model(checkpoint_path, device):
    # Architecture
    steppi = 30
    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    
    decoder_p = ConvFullyConnectedArch([Key("z", size=32)], [Key("pressure", size=steppi)])
    model_p = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)

    decoder_t = ConvFullyConnectedArch([Key("z", size=32)], [Key("temperature", size=steppi)])
    model_t = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_t).to(device)

    # Load Weights
    base_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    loaded = False
    
    # Try multiple naming conventions
    prefixes = ["fno_forward_model", "pino_forward_model"]
    for prefix in prefixes:
        p_path = os.path.join(base_dir, f"{prefix}_pressure.0.pth")
        t_path = os.path.join(base_dir, f"{prefix}_temperature.0.pth")
        if os.path.exists(p_path) and os.path.exists(t_path):
            try:
                model_p.load_state_dict(torch.load(p_path, map_location=device, weights_only=False))
                model_t.load_state_dict(torch.load(t_path, map_location=device, weights_only=False))
                print(f"Loaded: {prefix}")
                loaded = True
                break
            except Exception as e:
                print(f"Error loading {prefix}: {e}")

    # Try Optim checkpoint
    if not loaded:
        opt_path = os.path.join(base_dir, "optim_checkpoint.0.pth")
        if os.path.exists(opt_path):
            try:
                cp = torch.load(opt_path, map_location=device, weights_only=False)
                if 'models' in cp:
                    pk = next((k for k in cp['models'] if 'pressure' in k), None)
                    tk = next((k for k in cp['models'] if 'temperature' in k), None)
                    if pk and tk:
                        model_p.load_state_dict(cp['models'][pk])
                        model_t.load_state_dict(cp['models'][tk])
                        print("Loaded from optim_checkpoint.")
                        loaded = True
            except: pass

    if not loaded:
        print(f"WARNING: Could not load model from {checkpoint_path}")
        return None, None
        
    return model_p, model_t

def run_inference(inv, model_p, model_t, idx, device):
    model_p.eval()
    model_t.eval()
    batch = {k: torch.from_numpy(v[idx:idx+1]).to(device) for k, v in inv.items()}
    with torch.no_grad():
        pp = model_p(batch)["pressure"].cpu().numpy()[0]
        tt = model_t(batch)["temperature"].cpu().numpy()[0]
    return pp, tt

def plot_sample(model_name, dataset_name, idx, true_vol, pred_vol, layer, var_name, out_dir, inj_loc, prod_loc):
    """
    Plots Map, Inline, and Xline views for a specific variable.
    """
    # 1. Unpack Time-Series (Take Final Step)
    if true_vol.ndim == 4: # [T, Z, X, Y]
        true_slice = true_vol[-1]
        pred_slice = pred_vol[-1]
        diff_slice = np.abs(true_slice - pred_slice)
        time_label = "Final Step (Day 10950)"
    else: # Static
        # Handle explicitly missing time dimension [Z, X, Y]
        true_slice = true_vol
        pred_slice = pred_vol
        diff_slice = np.zeros_like(true_vol)
        time_label = "Static"

    nz, nx, ny = true_slice.shape
    map_layer = min(layer, nz-1)
    cx, cy = nx//2, ny//2

    # 2. Extract Slices (Note: Transpose Map for imshow)
    # Map (XY)
    map_t = true_slice[map_layer].T; map_p = pred_slice[map_layer].T; map_d = diff_slice[map_layer].T
    # Inline (XZ - X horizontal, Z vertical)
    in_t = true_slice[:, cx, :]; in_p = pred_slice[:, cx, :]; in_d = diff_slice[:, cx, :]
    # Xline (YZ - Y horizontal, Z vertical)
    xl_t = true_slice[:, :, cy]; xl_p = pred_slice[:, :, cy]; xl_d = diff_slice[:, :, cy]

    # 3. Plot
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    vmin = min(true_slice.min(), pred_slice.min())
    vmax = max(true_slice.max(), pred_slice.max())
    
    # User Request: Smooth the Colorbars and Set Fixed Difference Scales
    if var_name == "Temperature":
        cmap_main = 'viridis'
        cmap_diff = 'YlOrRd'
        vmax_diff = 25.0
    elif var_name == "Pressure":
        cmap_main = 'viridis'
        cmap_diff = 'YlOrRd'
        vmax_diff = 30.0
    elif var_name == "Porosity":
        cmap_main = 'bone'
        cmap_diff = 'YlOrRd'
        vmax_diff = 0.05
    else: # Permeability
        cmap_main = 'copper'
        cmap_diff = 'YlOrRd'
        vmax_diff = 500.0

    def draw(ax, data, title, is_diff=False, inj_loc=None, prod_loc=None, view_type="map"):
        cmap = cmap_diff if is_diff else cmap_main
        
        if view_type == "map":
            ext = [0, 32, 0, 32]
        else:
            ext = [0, 32, 0, 5]
            
        if is_diff:
            im = ax.imshow(data, cmap=cmap, origin='lower', vmin=0, vmax=vmax_diff, extent=ext, aspect='equal' if view_type=="map" else 'auto')
        else:
            im = ax.imshow(data, cmap=cmap, origin='lower', extent=ext, aspect='equal' if view_type=="map" else 'auto')
            
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Add Dynamic Wells and Dashed Intersections
        if view_type == "map":
            if inj_loc is not None:
                ax.scatter(inj_loc[0], inj_loc[1], c='blue', s=200, edgecolors='white', marker='o')
            if prod_loc is not None:
                ax.scatter(prod_loc[0], prod_loc[1], c='red', s=200, edgecolors='white', marker='X')
            ax.axhline(y=16, color='k', linestyle='--', linewidth=1.5)
            ax.axvline(x=16, color='k', linestyle='--', linewidth=1.5)
        elif view_type == "inline" or view_type == "xline":
            ax.axhline(y=2, color='k', linestyle='--', linewidth=1.5)
            ax.axvline(x=16, color='k', linestyle='--', linewidth=1.5)

    # Row 1: Truth (Simulation)
    draw(axes[0,0], map_t, f"Truth Map (L{map_layer})", inj_loc=inj_loc, prod_loc=prod_loc, view_type="map")
    axes[0,0].set_ylabel("Simulation", fontsize=14, fontweight='bold', rotation=90, labelpad=15)
    draw(axes[0,1], in_t, f"Truth Inline (x=16)", view_type="inline")
    draw(axes[0,2], xl_t, f"Truth Xline (y=16)", view_type="xline")

    # Row 2: Pred
    draw(axes[1,0], map_p, f"Pred Map (L{map_layer})", inj_loc=inj_loc, prod_loc=prod_loc, view_type="map")
    axes[1,0].set_ylabel(model_name, fontsize=14, fontweight='bold', rotation=90, labelpad=15)
    draw(axes[1,1], in_p, f"Pred Inline", view_type="inline")
    draw(axes[1,2], xl_p, f"Pred Xline", view_type="xline")

    # Row 3: Diff
    draw(axes[2,0], map_d, f"Diff Map", is_diff=True, inj_loc=inj_loc, prod_loc=prod_loc, view_type="map")
    axes[2,0].set_ylabel("Difference", fontsize=14, fontweight='bold', rotation=90, labelpad=15)
    draw(axes[2,1], in_d, f"Diff Inline", is_diff=True, view_type="inline")
    draw(axes[2,2], xl_d, f"Diff Xline", is_diff=True, view_type="xline")

    plt.suptitle(f"{model_name} on {dataset_name} (Idx {idx}) | {var_name} | {time_label}", fontsize=14)
    plt.tight_layout()
    
    fname = f"{dataset_name}_Idx{idx}_{model_name}_{var_name}.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()
    print(f"Saved: {fname}")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../PACKETS/Training4.mat")
    parser.add_argument("--test_data", type=str, default="../PACKETS/Test4.mat")
    parser.add_argument("--out", type=str, default="COMPARE_RESULTS/Full_Report")
    parser.add_argument("--idx_train", type=int, default=-1, help="-1 for random")
    parser.add_argument("--idx_test", type=int, default=-1, help="-1 for random")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Prepare Datasets
    datasets = {}
    
    # Load Training
    if os.path.exists(args.train_data):
        inv_tr, out_p_tr, out_t_tr = load_dataset(args.train_data)
        idx_tr = args.idx_train if args.idx_train >= 0 else random.randint(0, len(out_p_tr["pressure"])-1)
        stats_tr = get_stats_for_file(args.train_data)
        datasets["Training"] = (inv_tr, out_p_tr, out_t_tr, idx_tr, stats_tr)
    else:
        print(f"Skipping Training data (not found: {args.train_data})")

    # Load Test
    if os.path.exists(args.test_data):
        inv_te, out_p_te, out_t_te = load_dataset(args.test_data)
        idx_te = args.idx_test if args.idx_test >= 0 else random.randint(0, len(out_p_te["pressure"])-1)
        stats_te = get_stats_for_file(args.test_data)
        datasets["Test"] = (inv_te, out_p_te, out_t_te, idx_te, stats_te)
    else:
        print(f"Skipping Test data (not found: {args.test_data})")

    # 2. Iterate Models
    for model_name, cfg in MODELS_CONFIG.items():
        print(f"\n--- Running {model_name} ---")
        
        # Load Model
        mod_p, mod_t = load_model(cfg["checkpoint"], device)
        if mod_p is None: continue

        # 3. Process Each Dataset (Train/Test)
        for d_name, (inv, out_p, out_t, idx, stats) in datasets.items():
            print(f"Processing {d_name} Sample [{idx}]...")
            
            # Predict
            pred_p, pred_t = run_inference(inv, mod_p, mod_t, idx, device)
            true_p = out_p["pressure"][idx]
            true_t = out_t["temperature"][idx]

            # Un-Normalize using utilities.py formulas
            # Pressure formula: x = (P - 100) / 200 => P = x * 200 + 100
            pred_p = pred_p * 200.0 + 100.0
            true_p = true_p * 200.0 + 100.0
            
            # Temperature formula: x = (T_Kelvin - 273) / 100 => T_Celsius = x * 100
            pred_t = pred_t * 100.0
            true_t = true_t * 100.0 

            # Fix Dimensions [T,X,Y,Z] -> [T,Z,X,Y]
            if pred_p.shape[-1] == 5:
                pred_p = pred_p.transpose(0, 3, 1, 2)
                true_p = true_p.transpose(0, 3, 1, 2)
                pred_t = pred_t.transpose(0, 3, 1, 2)
                true_t = true_t.transpose(0, 3, 1, 2)

            # Output Dir with Train/Test Subfolders
            model_out = os.path.join(args.out, model_name, d_name)
            os.makedirs(model_out, exist_ok=True)

            # Find Dynamic Wells
            q_tensor = inv["Q"][idx]
            if q_tensor.ndim == 4: q_tensor = q_tensor[0] # Removes Front Singleton. Shape becomes [X,Y,Z] (32,32,5)
            inj_loc, prod_loc = extract_geothermal_wells(q_tensor)

            # Plot Target State Variables
            plot_sample(model_name, d_name, idx, true_p, pred_p, 2, "Pressure", model_out, inj_loc, prod_loc)
            plot_sample(model_name, d_name, idx, true_t, pred_t, 2, "Temperature", model_out, inj_loc, prod_loc)

            # Plot Static Input Permeability and Porosity (for all models!)
            if "perm" in inv:
                perm = inv["perm"][idx]
                if perm.ndim==4: perm=perm[0] # remove front singleton
                
                # Permeability formula: x = log10(Perm) => Perm = 10 ** x
                perm = 10 ** perm 
                
                if perm.shape[-1] == 5: perm = perm.transpose(2,0,1)    
                plot_sample(model_name, d_name, idx, perm, perm, 2, "Permeability", model_out, inj_loc, prod_loc)
                
            if "Phi" in inv:
                phi = inv["Phi"][idx]
                if phi.ndim==4: phi=phi[0] # remove front singleton
                
                # Porosity has NO normalization in utilities.py (x = data["Phi"])
                if phi.shape[-1] == 5: phi = phi.transpose(2,0,1)    
                plot_sample(model_name, d_name, idx, phi, phi, 2, "Porosity", model_out, inj_loc, prod_loc)