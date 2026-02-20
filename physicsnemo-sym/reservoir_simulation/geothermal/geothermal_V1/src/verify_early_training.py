import os
import torch
import numpy as np
import h5py
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import FNOArch, ConvFullyConnectedArch
from utilities import load_FNO_dataset2

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_keys = [
        Key("perm"), Key("Q"), Key("Qw"), Key("Phi"), 
        Key("Time"), Key("Pini"), Key("Tini")
    ]
    out_p_key = [Key("pressure")]
    out_t_key = [Key("temperature")]
    
    print("Loading scaled dataset...")
    # NOTE: utilities.py inherently maps P: (x-100)/200 and T: (x-273)/100
    inv, out_p, out_t = load_FNO_dataset2(
        '../PACKETS/Test4.hdf5',
        [k.name for k in input_keys],
        [k.name for k in out_p_key],
        [k.name for k in out_t_key],
        n_examples=1
    )
    
    decoder_p = ConvFullyConnectedArch([Key('z', size=32)], [Key('pressure', size=30)])
    model_p = FNOArch([Key(k.name, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)
    
    print("\nAttempting to load weights...")
    try:
        model_p.load_state_dict(torch.load('outputs/Forward_problem_FNO/ResSim/fno_forward_model_pressure.0.pth', map_location=device, weights_only=False))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Weight load failed: {e}")
        return
        
    batch = {k: torch.from_numpy(v[:1]).to(device) for k, v in inv.items()}
    with torch.no_grad():
        pred_p = model_p(batch)['pressure'].cpu().numpy()[0]
        
    print(f"\nRAW MODEL OUTPUT TENSOR P: min={pred_p.min():.4f}, max={pred_p.max():.4f}, mean={pred_p.mean():.4f}")
    
    # Verify un-normalization physics mappings match utilities.py expectations
    physical_p = (pred_p * 200.0) + 100.0
    print(f"UN-NORMALIZED PHYSICAL P: min={physical_p.min():.4f}, max={physical_p.max():.4f}, mean={physical_p.mean():.4f} Bar")
    
    truth_p = out_p['pressure'][0]
    physical_truth_p = (truth_p * 200.0) + 100.0
    print(f"TRUTH PHYSICAL P: min={physical_truth_p.min():.4f}, max={physical_truth_p.max():.4f}, mean={physical_truth_p.mean():.4f} Bar")
    
    # Assert variance to prove it's not a dead tensor anymore
    variance = pred_p.max() - pred_p.min()
    if variance > 0.05:
        print("\nSUCCESS: Model tensor is ALIVE and actively predicting varying physical structures.")
    else:
        print("\nWARNING: Model tensor is STILL DEAD (Variance < 0.05).")

if __name__ == '__main__':
    main()
