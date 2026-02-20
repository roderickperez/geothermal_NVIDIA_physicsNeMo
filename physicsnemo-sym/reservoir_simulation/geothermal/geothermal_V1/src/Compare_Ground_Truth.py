import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import pandas as pd
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import FNOArch, ConvFullyConnectedArch
from utilities import load_FNO_dataset2

DATA_STATS = {
    "Test": {
        "p_mean": 153.57, "p_std": 28.14,
        "t_mean": 349.01, "t_std": 30.68,
        "k_mean": 708.81, "k_std": 744.18,
        "phi_mean": 0.198, "phi_std": 0.065
    }
}

MODELS_CONFIG = {
    "FNO":       {"checkpoint": "outputs/Forward_problem_FNO/ResSim"},
    "PINO_Base": {"checkpoint": "outputs/Forward_problem_PINO/ResSim"},
    "PINO_CL":   {"checkpoint": "outputs/Forward_problem_PINO_CL/ResSim_M3_Curriculum"},
    "PINO_RL":   {"checkpoint": "outputs/Forward_problem_PINO_RL/ResSim_M4_Reverse"}
}

def compute_metrics(true, pred):
    true = true.flatten()
    pred = pred.flatten()
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    l2 = np.sqrt(np.mean((true - pred) ** 2)) / (np.sqrt(np.mean(true**2)) + 1e-10)
    return r2 * 100, l2 * 100

def load_model(checkpoint_path, device):
    steppi = 30
    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    decoder_p = ConvFullyConnectedArch([Key("z", size=32)], [Key("pressure", size=steppi)])
    model_p = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)
    decoder_t = ConvFullyConnectedArch([Key("z", size=32)], [Key("temperature", size=steppi)])
    model_t = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_t).to(device)

    base_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    loaded = False
    
    for prefix in ["fno_forward_model", "pino_forward_model"]:
        p_path = os.path.join(base_dir, f"{prefix}_pressure.0.pth")
        t_path = os.path.join(base_dir, f"{prefix}_temperature.0.pth")
        if os.path.exists(p_path) and os.path.exists(t_path):
            try:
                model_p.load_state_dict(torch.load(p_path, map_location=device, weights_only=False))
                model_t.load_state_dict(torch.load(t_path, map_location=device, weights_only=False))
                loaded = True
                break
            except: pass

    if not loaded:
        opt_path = os.path.join(base_dir, "optim_checkpoint.0.pth")
        if os.path.exists(opt_path):
            try:
                cp = torch.load(opt_path, map_location=device, weights_only=False)
                pk = next((k for k in cp['models'] if 'pressure' in k), None)
                tk = next((k for k in cp['models'] if 'temperature' in k), None)
                if pk and tk:
                    model_p.load_state_dict(cp['models'][pk])
                    model_t.load_state_dict(cp['models'][tk])
                    loaded = True
            except: pass

    if not loaded: return None, None
    return model_p, model_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="../PACKETS/Test4.mat")
    parser.add_argument("--out", type=str, default="COMPARE_RESULTS/Ground_Truth_Benchmark")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Loading Ground Truth from {args.test_data}")
    data_path = args.test_data
    if data_path.endswith(".mat"):
        from utilities import preprocess_FNO_mat
        preprocess_FNO_mat(data_path)
        data_path = data_path.replace(".mat", ".hdf5")
        
    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    inv, out_p, out_t = load_FNO_dataset2(data_path, input_keys, ["pressure"], ["temperature"], n_examples=None)
    stats = DATA_STATS["Test"]

    n_samples = len(out_p["pressure"])
    idx = random.randint(0, n_samples - 1)
    print(f"Evaluating Sample Index: {idx} (representing Ground Truth)")

    true_p = out_p["pressure"][idx] * 200.0 + 100.0
    true_t = out_t["temperature"][idx] * 100.0 
    
    # Store aggregated metrics
    results_p_r2, results_p_l2 = {}, {}
    results_t_r2, results_t_l2 = {}, {}
    steps = np.arange(1, 31)

    for model_name, cfg in MODELS_CONFIG.items():
        print(f"Benchmarking {model_name}...")
        mod_p, mod_t = load_model(cfg["checkpoint"], device)
        if mod_p is None:
            print(f"  [!] Missing weights for {model_name}")
            continue
            
        mod_p.eval(); mod_t.eval()
        batch = {k: torch.from_numpy(v[idx:idx+1]).to(device) for k, v in inv.items()}
        with torch.no_grad():
            pred_p = mod_p(batch)["pressure"].cpu().numpy()[0]
            pred_t = mod_t(batch)["temperature"].cpu().numpy()[0]

        # Un-normalize appropriately
        pred_p = pred_p * 200.0 + 100.0
        pred_t = pred_t * 100.0 
        
        # Calculate dynamic physical metrics per timestep
        p_r2_loc, p_l2_loc = [], []
        t_r2_loc, t_l2_loc = [], []
        
        for t in range(30):
            r2p, l2p = compute_metrics(true_p[t], pred_p[t])
            r2t, l2t = compute_metrics(true_t[t], pred_t[t])
            p_r2_loc.append(r2p); p_l2_loc.append(l2p)
            t_r2_loc.append(r2t); t_l2_loc.append(l2t)
            
        results_p_r2[model_name] = p_r2_loc
        results_p_l2[model_name] = p_l2_loc
        results_t_r2[model_name] = t_r2_loc
        results_t_l2[model_name] = t_l2_loc

    # Plot Consolidated Benchmark
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    markers = ['o', 'x', '^', 's']
    
    for i, (model_name, r2_data) in enumerate(results_p_r2.items()):
        axes[0, 0].plot(steps, r2_data, label=model_name, marker=markers[i], lw=2)
        axes[0, 1].plot(steps, results_p_l2[model_name], label=model_name, marker=markers[i], lw=2)
        axes[1, 0].plot(steps, results_t_r2[model_name], label=model_name, marker=markers[i], lw=2)
        axes[1, 1].plot(steps, results_t_l2[model_name], label=model_name, marker=markers[i], lw=2)

    axes[0,0].set_title("Pressure R2 Accuracy vs Truth (%)")
    axes[0,0].set_ylabel("R2 (%)")
    axes[0,1].set_title("Pressure L2 True-Error vs Truth (%)")
    axes[0,1].set_ylabel("L2 (%)")
    axes[1,0].set_title("Temperature R2 Accuracy vs Truth (%)")
    axes[1,0].set_ylabel("R2 (%)")
    axes[1,1].set_title("Temperature L2 True-Error vs Truth (%)")
    axes[1,1].set_ylabel("L2 (%)")

    for ax in axes.flat:
        ax.set_xlabel("Timestep (Month)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"Ground_Truth_Benchmark_Sample{idx}.png"))
    print(f"\n=> Benchmark saved to {args.out}/Ground_Truth_Benchmark_Sample{idx}.png")
    
if __name__ == "__main__":
    main()
