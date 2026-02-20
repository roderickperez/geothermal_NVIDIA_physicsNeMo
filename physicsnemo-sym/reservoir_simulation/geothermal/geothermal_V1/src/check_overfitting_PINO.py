import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

# PhysicsNeMo imports
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import FNOArch, ConvFullyConnectedArch
from utilities import load_FNO_dataset2

# ==========================================
# CONFIGURATION & STATS
# ==========================================

DATA_STATS = {
    "Training": {"p_mean": 155.66, "p_std": 26.50},
    "Test":     {"p_mean": 153.57, "p_std": 28.14}
}

def load_data(data_path):
    print(f"Loading {data_path}...")
    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    inv, out_p, _ = load_FNO_dataset2(data_path, input_keys, ["pressure"], ["temperature"], n_examples=2)
    return inv, out_p

def load_pino_model(checkpoint_dir, device):
    print(f"Loading OLD 60K Model from {checkpoint_dir}...")
    steppi = 30
    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    decoder_p = ConvFullyConnectedArch([Key("z", size=32)], [Key("pressure", size=steppi)])
    model_p = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)

    p_path = os.path.join(checkpoint_dir, "pino_forward_model_pressure.0.pth")
    if os.path.exists(p_path):
        model_p.load_state_dict(torch.load(p_path, map_location=device, weights_only=False))
        print("Model loaded successfully.")
    else:
        print("Could not find weights.")
        return None
    return model_p

def predict_and_unnormalize(inv, out_p, model_p, idx, stats, device):
    model_p.eval()
    batch = {k: torch.from_numpy(v[idx:idx+1]).to(device) for k, v in inv.items()}
    with torch.no_grad():
        pred_p = model_p(batch)["pressure"].cpu().numpy()[0]
    
    true_p = out_p["pressure"][idx]

    # Un-normalize using the EXACT stats for the dataset
    pred_p = pred_p * stats["p_std"] + stats["p_mean"]
    true_p = true_p * stats["p_std"] + stats["p_mean"]

    # Transpose [T, X, Y, Z] -> [T, Z, X, Y]
    if pred_p.shape[-1] == 5:
        pred_p = pred_p.transpose(0, 3, 1, 2)
        true_p = true_p.transpose(0, 3, 1, 2)

    return true_p[-1], pred_p[-1] # Return Final Timestep [Z, X, Y]

def draw_map(ax, data, title, q_map_3d, is_diff=False):
    cmap = 'inferno' if is_diff else 'viridis'
    
    # Layer 2 data
    layer_data = data[2].T
    im = ax.imshow(layer_data, cmap=cmap, origin='lower', aspect='auto') 
    
    # Calculate Range for Title
    d_min, d_max = layer_data.min(), layer_data.max()
    ax.set_title(f"{title}\nRange: [{d_min:.2f} - {d_max:.2f}]", fontsize=11)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # dynamically find true wells in this exact sample from Q-Tensor
    # Sum across Z (dim 2) because wells perforate multiple layers
    q_surface = np.sum(q_map_3d, axis=2)
    q_surface = q_surface.T # Match XY transposing of imshow
    
    # Positive Total Q = Injector
    inj_idx = np.unravel_index(np.argmax(q_surface), q_surface.shape)
    if q_surface[inj_idx] > 0.001:
        ax.scatter(inj_idx[1], inj_idx[0], c='red', s=60, edgecolors='white', marker='^', zorder=5)
        ax.text(inj_idx[1]+1, inj_idx[0]+1, 'INJ', fontsize=9, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))
        
    # Negative Total Q = Producer
    prod_idx = np.unravel_index(np.argmin(q_surface), q_surface.shape)
    if q_surface[prod_idx] < -0.001:
        ax.scatter(prod_idx[1], prod_idx[0], c='black', s=60, edgecolors='white', marker='v', zorder=5)
        ax.text(prod_idx[1]+1, prod_idx[0]+1, 'PROD', fontsize=9, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = "COMPARE_RESULTS/Global"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
    inv_train, out_train = load_data("../PACKETS/Training4.hdf5")
    inv_test, out_test = load_data("../PACKETS/Test4.hdf5")

    # 2. Load ORIGINAL 60K Model (PINO)
    model = load_pino_model("outputs/Forward_problem_PINO/ResSim", device)

    # 3. Predict (Using Sample Index 0 for both)
    idx = 0
    true_tr, pred_tr = predict_and_unnormalize(inv_train, out_train, model, idx, DATA_STATS["Training"], device)
    true_te, pred_te = predict_and_unnormalize(inv_test, out_test, model, idx, DATA_STATS["Test"], device)

    # Extract dynamic well topology Q[idx] -> Shape: (NX, NY, NZ)
    q_train = inv_train["Q"][idx, 0, ...]
    q_test = inv_test["Q"][idx, 0, ...]
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ROW 1: TRAINING SET
    draw_map(axes[0,0], true_tr, "TRAINING Ground Truth", q_train)
    draw_map(axes[0,1], pred_tr, "TRAINING Old 60K Prediction", q_train)
    draw_map(axes[0,2], np.abs(true_tr - pred_tr), "TRAINING Absolute Error", q_train, is_diff=True)

    # ROW 2: TEST SET
    draw_map(axes[1,0], true_te, "TEST Ground Truth", q_test)
    draw_map(axes[1,1], pred_te, "TEST Old 60K Prediction", q_test)
    draw_map(axes[1,2], np.abs(true_te - pred_te), "TEST Absolute Error", q_test, is_diff=True)

    plt.suptitle("Original 60,000 Step Model Check: Dynamic True Wells View\n(Layer 2 Map View - Final Timestep)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "Overfitting_Analysis_Old_60k.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison to {save_path}")
