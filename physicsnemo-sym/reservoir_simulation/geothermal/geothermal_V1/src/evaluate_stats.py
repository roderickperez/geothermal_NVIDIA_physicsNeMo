import os
import torch
import numpy as np

# PhysicsNeMo imports
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import FNOArch, ConvFullyConnectedArch
from utilities import load_FNO_dataset2

DATA_STATS = {
    "Training": {"p_mean": 155.66, "p_std": 26.50},
    "Test":     {"p_mean": 153.57, "p_std": 28.14}
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_data(data_path):
    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    inv, out_p, _ = load_FNO_dataset2(data_path, input_keys, ["pressure"], ["temperature"], n_examples=2)
    return inv, out_p

def load_fno_model(checkpoint_dir, device):
    steppi = 30
    input_keys = ["perm", "Q", "Qw", "Phi", "Time", "Pini", "Tini"]
    decoder_p = ConvFullyConnectedArch([Key("z", size=32)], [Key("pressure", size=steppi)])
    model_p = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)
    p_path = os.path.join(checkpoint_dir, "fno_forward_model_pressure.0.pth")
    model_p.load_state_dict(torch.load(p_path, map_location=device, weights_only=False))
    model_p.eval()
    return model_p

inv_test, out_test = load_data("../PACKETS/Test4.hdf5")
model = load_fno_model("outputs/Forward_problem_FNO/ResSim", device)

idx = 0
batch = {k: torch.from_numpy(v[idx:idx+1]).to(device) for k, v in inv_test.items()}
with torch.no_grad():
    pred_p = model(batch)["pressure"].cpu().numpy()[0]

pred_p = pred_p * DATA_STATS["Test"]["p_std"] + DATA_STATS["Test"]["p_mean"]
true_p = out_test["pressure"][idx] * DATA_STATS["Test"]["p_std"] + DATA_STATS["Test"]["p_mean"]

print(f"\n--- PERFORMANCE VERDICT ---")
print(f"Ground Truth Min: {true_p.min():.2f} / Max: {true_p.max():.2f} / Mean: {true_p.mean():.2f}")
print(f"FNO Prediction Min: {pred_p.min():.2f} / Max: {pred_p.max():.2f} / Mean: {pred_p.mean():.2f}")
print(f"Std Deviation (Spread): {pred_p.std():.2f}")
print(f"Mean Abs Error (MAE): {np.abs(true_p - pred_p).mean():.2f}")

diff = pred_p.max() - pred_p.min()
if diff < 1.0:
    print("VERDICT: COLLAPSE DETECTED. The model is outputting a constant field.")
elif diff > 100:
    print("VERDICT: EXPLOSION DETECTED. The model output variance is unnaturally high.")
else:
    print("VERDICT: MODEL IS RESPONDING. It has a functional dynamic range.")
