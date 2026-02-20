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

WELLS = [
    [1, 24, 'I1'], [3, 3, 'I2'], [31, 1, 'I3'], [31, 31, 'I4'],  # Injectors
    [7, 9, 'P1'], [14, 12, 'P2'], [28, 19, 'P3'], [14, 27, 'P4'] # Producers
]

MODELS_CONFIG = {
    "FNO": {"checkpoint": "outputs/Forward_problem_FNO_V2/ResSim", "tag": "FNO"},
}

def load_dataset(data_path):
    print(f"\nLoading Data from {data_path}...")
    if data_path.endswith(".mat"):
        from utilities import preprocess_FNO_mat
        preprocess_FNO_mat(data_path)
        data_path = data_path.replace(".mat", ".hdf5")

    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    
    # We load Absolute Pressure & Temp as Ground Truth
    inv, out_p, out_t = load_FNO_dataset2(data_path, input_keys, ["pressure"], ["temperature"], n_examples=None)
    return inv, out_p, out_t

def load_model(checkpoint_path, device):
    steppi = 30
    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    
    # [CRITICAL FIX]: Architecture must match the Delta Output Keys!
    decoder_p = ConvFullyConnectedArch([Key("z", size=32)], [Key("delta_pressure", size=steppi)])
    model_p = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)

    decoder_t = ConvFullyConnectedArch([Key("z", size=32)], [Key("delta_temperature", size=steppi)])
    model_t = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_t).to(device)

    base_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    loaded = False
    
    prefixes = ["fno_forward_model", "pino_forward_model"]
    for prefix in prefixes:
        p_path = os.path.join(base_dir, f"{prefix}_pressure.0.pth")
        t_path = os.path.join(base_dir, f"{prefix}_temperature.0.pth")
        
        # If delta weights have the new name
        p_path_alt = os.path.join(base_dir, f"{prefix}_delta_pressure.0.pth")
        t_path_alt = os.path.join(base_dir, f"{prefix}_delta_temperature.0.pth")

        if os.path.exists(p_path_alt): p_path = p_path_alt
        if os.path.exists(t_path_alt): t_path = t_path_alt

        if os.path.exists(p_path) and os.path.exists(t_path):
            try:
                model_p.load_state_dict(torch.load(p_path, map_location=device, weights_only=False))
                model_t.load_state_dict(torch.load(t_path, map_location=device, weights_only=False))
                print(f"Loaded: {prefix}")
                loaded = True
                break
            except Exception as e:
                print(f"Error loading {prefix}: {e}")

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

def run_inference_and_calculate_absolute(inv, out_p, out_t, model_p, model_t, idx, device):
    """
    Runs the model, extracts Delta, un-normalizes, and converts back to Absolute Physical Units.
    """
    model_p.eval()
    model_t.eval()
    batch = {k: torch.from_numpy(v[idx:idx+1]).to(device) for k, v in inv.items()}
    
    with torch.no_grad():
        # 1. Model Predicts Normalized Delta
        pred_delta_p = model_p(batch)["delta_pressure"].cpu().numpy()[0]
        pred_delta_t = model_t(batch)["delta_temperature"].cpu().numpy()[0]
    
    # 2. Extract Ground Truth Absolute (Still Normalized by utilities.py)
    true_p = out_p["pressure"][idx]
    true_t = out_t["temperature"][idx]
    
    # 3. Extract Initial Conditions (Still Normalized)
    # Shape is [X, Y, Z], we add [Time] axis to match the 30 timesteps
    pini_norm = inv["Pini"][idx]
    if pini_norm.ndim == 3: pini_norm = np.expand_dims(pini_norm, axis=0)
    
    tini_norm = inv["Tini"][idx]
    if tini_norm.ndim == 3: tini_norm = np.expand_dims(tini_norm, axis=0)

    # ===================================================================
    # 4. UN-NORMALIZE MATHEMATICS (Based on utilities.py logic)
    # utilities.py does: x = (x - 100) / 200 for Pressure
    # Therefore: Delta_Norm = (P_true - 100)/200 - (Pini - 100)/200 = (P_true - Pini) / 200
    # ===================================================================
    
    # A) Physical Un-normalized Initial Conditions
    pini_phys = pini_norm * 200.0 + 100.0
    tini_phys = tini_norm * 100.0 + 273.0

    # B) Physical Un-normalized Deltas (Multiply by the denominator)
    pred_delta_p_phys = pred_delta_p * 200.0
    pred_delta_t_phys = pred_delta_t * 100.0

    # C) Calculate Final Absolute Prediction
    pred_p_phys = pini_phys + pred_delta_p_phys
    pred_t_phys = tini_phys + pred_delta_t_phys

    # D) Un-normalize Ground Truth Absolute for direct comparison
    true_p_phys = true_p * 200.0 + 100.0
    true_t_phys = true_t * 100.0 + 273.0

    # Convert Temp to Celsius
    pred_t_phys -= 273.15
    true_t_phys -= 273.15

    return true_p_phys, pred_p_phys, true_t_phys, pred_t_phys

def plot_sample(model_name, dataset_name, idx, true_vol, pred_vol, layer, var_name, out_dir):
    if true_vol.ndim == 4: # [T, Z, X, Y]
        true_slice = true_vol[-1]
        pred_slice = pred_vol[-1]
        diff_slice = np.abs(true_slice - pred_slice)
        time_label = "Final Step (Day 10950)"
    else: 
        true_slice = true_vol
        pred_slice = pred_vol
        diff_slice = np.zeros_like(true_vol)
        time_label = "Static"

    nz, nx, ny = true_slice.shape
    map_layer = min(layer, nz-1)
    cx, cy = nx//2, ny//2

    # Map (XY)
    map_t = true_slice[map_layer].T; map_p = pred_slice[map_layer].T; map_d = diff_slice[map_layer].T
    # Inline (YZ)
    in_t = true_slice[:, cx, :]; in_p = pred_slice[:, cx, :]; in_d = diff_slice[:, cx, :]
    # Xline (XZ)
    xl_t = true_slice[:, :, cy]; xl_p = pred_slice[:, :, cy]; xl_d = diff_slice[:, :, cy]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))

    def draw(ax, data, title, is_diff=False, wells=False):
        cmap = 'inferno' if is_diff else 'turbo'
        im = ax.imshow(data, cmap=cmap, origin='lower', aspect='auto')
        
        # Calculate Range for Title
        d_min, d_max = data.min(), data.max()
        ax.set_title(f"{title}\n[{d_min:.1f} - {d_max:.1f}]", fontsize=9)
        
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        if wells:
            for wx, wy, name in WELLS:
                c = 'cyan' if 'I' in name else 'white'
                ax.scatter(wx, wy, c=c, s=30, edgecolors='k')
                ax.text(wx, wy, name, fontsize=7, color='k')

    draw(axes[0,0], map_t, f"Truth Map (L{map_layer})", wells=True)
    draw(axes[0,1], in_t, f"Truth Inline (X={cx})")
    draw(axes[0,2], xl_t, f"Truth Xline (Y={cy})")

    draw(axes[1,0], map_p, f"Pred Map (L{map_layer})", wells=True)
    draw(axes[1,1], in_p, f"Pred Inline")
    draw(axes[1,2], xl_p, f"Pred Xline")

    draw(axes[2,0], map_d, f"Diff Map", is_diff=True, wells=True)
    draw(axes[2,1], in_d, f"Diff Inline", is_diff=True)
    draw(axes[2,2], xl_d, f"Diff Xline", is_diff=True)

    plt.suptitle(f"{model_name} on {dataset_name} (Idx {idx}) | {var_name} | {time_label}", fontsize=14)
    plt.tight_layout()
    
    fname = f"{dataset_name}_Idx{idx}_{model_name}_{var_name}.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()
    print(f"Saved: {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="../PACKETS/Test4.mat")
    parser.add_argument("--out", type=str, default="COMPARE_RESULTS/Delta_Viz")
    parser.add_argument("--idx_test", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inv_te, out_p_te, out_t_te = load_dataset(args.test_data)
    idx_te = args.idx_test

    for model_name, cfg in MODELS_CONFIG.items():
        print(f"\n--- Running {model_name} ---")
        
        mod_p, mod_t = load_model(cfg["checkpoint"], device)
        if mod_p is None: continue

        # Run Inference and Do the Absolute Math
        true_p, pred_p, true_t, pred_t = run_inference_and_calculate_absolute(
            inv_te, out_p_te, out_t_te, mod_p, mod_t, idx_te, device
        )

        # Fix Dimensions [T,X,Y,Z] -> [T,Z,X,Y]
        if pred_p.shape[-1] == 5:
            pred_p = pred_p.transpose(0, 3, 1, 2)
            true_p = true_p.transpose(0, 3, 1, 2)
            pred_t = pred_t.transpose(0, 3, 1, 2)
            true_t = true_t.transpose(0, 3, 1, 2)

        model_out = os.path.join(args.out, model_name)
        os.makedirs(model_out, exist_ok=True)

        plot_sample(model_name, "Test", idx_te, true_p, pred_p, 2, "Pressure", model_out)
        plot_sample(model_name, "Test", idx_te, true_t, pred_t, 2, "Temperature", model_out)
