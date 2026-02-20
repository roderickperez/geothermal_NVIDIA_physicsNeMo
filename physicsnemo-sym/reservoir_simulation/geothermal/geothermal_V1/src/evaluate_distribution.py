import os
import torch
import numpy as np

from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import FNOArch, ConvFullyConnectedArch
from utilities import load_FNO_dataset2

DATA_STATS = {
    "Test": {"p_mean": 153.57, "p_std": 28.14}
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_fno_model(checkpoint_dir, device):
    steppi = 30
    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    decoder_p = ConvFullyConnectedArch([Key("z", size=32)], [Key("pressure", size=steppi)])
    model_p = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)
    p_path = os.path.join(checkpoint_dir, "fno_forward_model_pressure.0.pth")
    model_p.load_state_dict(torch.load(p_path, map_location=device, weights_only=False))
    model_p.eval()
    return model_p

inv_test, out_test, _ = load_FNO_dataset2("../PACKETS/Test4.hdf5", ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"], ["pressure"], ["temperature"], n_examples=2)
model = load_fno_model("outputs/Forward_problem_FNO/ResSim", device)

idx = 0
batch = {k: torch.from_numpy(v[idx:idx+1]).to(device) for k, v in inv_test.items()}
with torch.no_grad():
    pred_p = model(batch)["pressure"].cpu().numpy()[0]

pred_p = pred_p * DATA_STATS["Test"]["p_std"] + DATA_STATS["Test"]["p_mean"]
true_p = out_test["pressure"][idx] * DATA_STATS["Test"]["p_std"] + DATA_STATS["Test"]["p_mean"]

# Select Layer 2 Final Timestep exactly as plotted
if pred_p.shape[-1] == 5:
    pred_p = pred_p.transpose(0, 3, 1, 2)
    true_p = true_p.transpose(0, 3, 1, 2)
p_pred_2d = pred_p[-1, 2, :, :]
p_true_2d = true_p[-1, 2, :, :]

print("\n--- GROUND TRUTH DISTRIBUTION ---")
print(f"Min: {p_true_2d.min():.4f}, Max: {p_true_2d.max():.4f}")
print(f"1st: {np.percentile(p_true_2d, 1):.4f}, 5th: {np.percentile(p_true_2d, 5):.4f}")
print(f"50th (Median): {np.percentile(p_true_2d, 50):.4f}")
print(f"95th: {np.percentile(p_true_2d, 95):.4f}, 99th: {np.percentile(p_true_2d, 99):.4f}")

print("\n--- FNO PREDICTION DISTRIBUTION ---")
print(f"Min: {p_pred_2d.min():.4f}, Max: {p_pred_2d.max():.4f}")
print(f"1st: {np.percentile(p_pred_2d, 1):.4f}, 5th: {np.percentile(p_pred_2d, 5):.4f}")
print(f"50th (Median): {np.percentile(p_pred_2d, 50):.4f}")
print(f"95th: {np.percentile(p_pred_2d, 95):.4f}, 99th: {np.percentile(p_pred_2d, 99):.4f}")
