import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import FNOArch, ConvFullyConnectedArch
from utilities import load_FNO_dataset2

MODELS_CONFIG = {
    "FNO_V3": {"checkpoint": "outputs/Forward_problem_FNO_V3/ResSim_V3", "tag": "FNO_V3"},
}

# --- EXACT STATS FROM YOUR 60K LOGS ---
DATA_STATS = {
    "Training": {
        "P_MEAN": 155.818, "P_STD": 26.6224, "PINI_MEAN": 249.433, "PINI_STD": 7.31539,
        "T_MEAN": 349.321, "T_STD": 30.4247, "TINI_MEAN": 372.725, "TINI_STD": 10.6776,
        "DP_MEAN": -1910.8816, "DP_STD": 53.7091, "DT_MEAN": -10.8319, "DT_STD": 9.0834
    },
    "Test": {
        "P_MEAN": 153.567, "P_STD": 28.1408, "PINI_MEAN": 250.026, "PINI_STD": 6.7608,
        "T_MEAN": 349.012, "T_STD": 30.6824, "TINI_MEAN": 372.647, "TINI_STD": 10.8107,
        "DP_MEAN": -1779.2994, "DP_STD": 45.2766, "DT_MEAN": -11.0853, "DT_STD": 9.2635
    }
}

def load_data_and_model(checkpoint_path, data_path, device):
    print(f"\nLoading Data from {data_path}...")
    if data_path.endswith(".mat"):
        from utilities import preprocess_FNO_mat
        preprocess_FNO_mat(data_path)
        data_path = data_path.replace(".mat", ".hdf5")

    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    inv, out_p, out_t = load_FNO_dataset2(data_path, input_keys, ["pressure"], ["temperature"], n_examples=None)

    steppi = 30
    decoder_p = ConvFullyConnectedArch([Key("z", size=32)], [Key("delta_pressure", size=steppi)])
    model_p = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[32,32,4], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)

    decoder_t = ConvFullyConnectedArch([Key("z", size=32)], [Key("delta_temperature", size=steppi)])
    model_t = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_t).to(device)

    base_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    p_path = os.path.join(base_dir, "fno_forward_model_delta_pressure.0.pth")
    t_path = os.path.join(base_dir, "fno_forward_model_delta_temperature.0.pth")
    
    try:
        model_p.load_state_dict(torch.load(p_path, map_location=device, weights_only=False))
        model_t.load_state_dict(torch.load(t_path, map_location=device, weights_only=False))
        print("V2 Delta Models loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, None
    
    return inv, out_p, out_t, model_p, model_t

def find_dynamic_wells(inv, idx):
    q_norm = inv["Q"][idx]
    if q_norm.ndim == 4: q_norm = q_norm[0]
    
    q_variation = np.abs(q_norm - np.median(q_norm)).sum(axis=-1)
    threshold = q_variation.max() * 0.3 
    
    wells = []
    for x in range(q_variation.shape[0]):
        for y in range(q_variation.shape[1]):
            if q_variation[x, y] > threshold:
                val = q_norm[x, y, :].sum()
                w_type = 'I' if val > np.median(q_norm)*5 else 'P'
                wells.append((x, y, w_type))
    return wells

def predict_and_unnormalize(inv, out_p, out_t, model_p, model_t, idx, stats, device):
    model_p.eval()
    model_t.eval()
    batch = {k: torch.from_numpy(v[idx:idx+1]).to(device) for k, v in inv.items()}
    
    with torch.no_grad():
        pred_delta_p_norm = model_p(batch)["delta_pressure"].cpu().numpy()[0]
        pred_delta_t_norm = model_t(batch)["delta_temperature"].cpu().numpy()[0]
    
    true_p_norm = out_p["pressure"][idx]
    true_t_norm = out_t["temperature"][idx]
    pini_norm = np.expand_dims(inv["Pini"][idx], axis=0) if inv["Pini"][idx].ndim == 3 else inv["Pini"][idx]
    tini_norm = np.expand_dims(inv["Tini"][idx], axis=0) if inv["Tini"][idx].ndim == 3 else inv["Tini"][idx]

    pred_p_phys = (pini_norm * stats["PINI_STD"] + stats["PINI_MEAN"]) + (pred_delta_p_norm * stats["DP_STD"] + stats["DP_MEAN"])
    pred_t_phys = (tini_norm * stats["TINI_STD"] + stats["TINI_MEAN"]) + (pred_delta_t_norm * stats["DT_STD"] + stats["DT_MEAN"])

    true_p_phys = true_p_norm * stats["P_STD"] + stats["P_MEAN"]
    true_t_phys = true_t_norm * stats["T_STD"] + stats["T_MEAN"]

    pred_t_phys -= 273.15
    true_t_phys -= 273.15

    if pred_p_phys.shape[-1] == 5:
        pred_p_phys = pred_p_phys.transpose(0, 3, 1, 2)
        true_p_phys = true_p_phys.transpose(0, 3, 1, 2)
        pred_t_phys = pred_t_phys.transpose(0, 3, 1, 2)
        true_t_phys = true_t_phys.transpose(0, 3, 1, 2)

    return true_p_phys, pred_p_phys, true_t_phys, pred_t_phys

def plot_slice(model_name, true_vol, pred_vol, layer, var_name, out_dir, dynamic_wells, idx, dataset_type):
    true_slice = true_vol[-1]
    pred_slice = pred_vol[-1]
    diff_slice = np.abs(true_slice - pred_slice)

    nz, nx, ny = true_slice.shape
    map_layer = min(layer, nz-1)
    cx, cy = nx//2, ny//2

    map_t = true_slice[map_layer]; map_p = pred_slice[map_layer]; map_d = diff_slice[map_layer]
    in_t = true_slice[:, cx, :]; in_p = pred_slice[:, cx, :]; in_d = diff_slice[:, cx, :]
    xl_t = true_slice[:, :, cy]; xl_p = pred_slice[:, :, cy]; xl_d = diff_slice[:, :, cy]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # SHARED SCALES FOR TRUTH AND PREDICTION
    vmin_map = min(map_t.min(), map_p.min()); vmax_map = max(map_t.max(), map_p.max())
    vmin_in = min(in_t.min(), in_p.min()); vmax_in = max(in_t.max(), in_p.max())
    vmin_xl = min(xl_t.min(), xl_p.min()); vmax_xl = max(xl_t.max(), xl_p.max())

    def draw(ax, data, title, vmin, vmax, is_diff=False, show_wells=False):
        cmap = 'magma' if is_diff else 'turbo'
        im = ax.imshow(data, cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n[{data.min():.1f} - {data.max():.1f}]", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        if show_wells and "Map" in title:
            for x, y, wtype in dynamic_wells:
                color = 'blue' if wtype == 'I' else 'red'
                ax.scatter(y, x, c=color, s=200, edgecolors='white', linewidths=1.5, marker='o') 
                ax.text(y+1.5, x+1.5, wtype, color='black', fontweight='bold', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # ROW 1 & 2: Shared Scales
    draw(axes[0,0], map_t, f"Truth Map (L{map_layer})", vmin_map, vmax_map, show_wells=True)
    draw(axes[0,1], in_t, f"Truth Inline (X={cx})", vmin_in, vmax_in)
    draw(axes[0,2], xl_t, f"Truth Xline (Y={cy})", vmin_xl, vmax_xl)
    draw(axes[1,0], map_p, f"Pred Map (L{map_layer})", vmin_map, vmax_map, show_wells=True)
    draw(axes[1,1], in_p, f"Pred Inline", vmin_in, vmax_in)
    draw(axes[1,2], xl_p, f"Pred Xline", vmin_xl, vmax_xl)

    # ROW 3: Difference (0 to Max Diff)
    draw(axes[2,0], map_d, f"Diff Map", 0, map_d.max(), is_diff=True, show_wells=True)
    draw(axes[2,1], in_d, f"Diff Inline", 0, in_d.max(), is_diff=True)
    draw(axes[2,2], xl_d, f"Diff Xline", 0, xl_d.max(), is_diff=True)

    # ROTATED ROW LABELS
    axes[0, 0].set_ylabel("Simulation | Ground Truth", fontsize=14, fontweight='bold', labelpad=15)
    axes[1, 0].set_ylabel("Prediction", fontsize=14, fontweight='bold', labelpad=15)
    axes[2, 0].set_ylabel("Difference", fontsize=14, fontweight='bold', labelpad=15)

    plt.suptitle(f"{model_name} V2 Delta | {dataset_type} Data (Idx {idx}) | {var_name} | Final Step", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"V5_Viz_{dataset_type}_{var_name}_Idx{idx}.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../PACKETS/Test4.mat")
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = "COMPARE_RESULTS/FNO_V3_Delta_Viz"
    os.makedirs(out_dir, exist_ok=True)

    dataset_type = "Training" if "Training" in args.data else "Test"
    stats_to_use = DATA_STATS[dataset_type]

    inv, out_p, out_t, mod_p, mod_t = load_data_and_model("outputs/Forward_problem_FNO_V3/ResSim_V3", args.data, device)
    
    if mod_p:
        idx = args.idx
        dyn_wells = find_dynamic_wells(inv, idx)
        true_p, pred_p, true_t, pred_t = predict_and_unnormalize(inv, out_p, out_t, mod_p, mod_t, idx, stats_to_use, device)

        plot_slice("FNO", true_p, pred_p, 2, "Pressure", out_dir, dyn_wells, idx, dataset_type)
        plot_slice("FNO", true_t, pred_t, 2, "Temperature", out_dir, dyn_wells, idx, dataset_type)
        print(f"Saved highly-formatted V5 physical plots for {dataset_type} data to {out_dir}")
