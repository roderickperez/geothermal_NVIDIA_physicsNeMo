import os
import torch
import numpy as np
import h5py
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import FNOArch, ConvFullyConnectedArch

def load_raw_data(path, input_keys, output_keys, output_keys2):
    data = h5py.File(path, "r")
    invar, outvar, outvar2 = dict(), dict(), dict()
    for d, keys in [(invar, input_keys), (outvar, output_keys), (outvar2, output_keys2)]:
        for k in keys:
            x = data[k][:]
            # ABSOLUTELY NO GEOTHERMAL NORMALIZATION HERE
            d[k] = x
    return (invar, outvar, outvar2)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_keys = ['perm', 'Q', 'Qw', 'Phi', 'Time', 'Pini', 'Tini']
    inv, out_p, out_t = load_raw_data('../PACKETS/Test4.hdf5', input_keys, ['pressure'], ['temperature'])
    
    decoder_p = ConvFullyConnectedArch([Key('z', size=32)], [Key('pressure', size=30)])
    model_p = FNOArch([Key(k, size=1) for k in input_keys], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)
    
    model_p.load_state_dict(torch.load('outputs/Forward_problem_FNO/ResSim/fno_forward_model_pressure.0.pth', map_location=device, weights_only=False))
    
    batch = {k: torch.from_numpy(v[:1]).to(device) for k, v in inv.items()}
    with torch.no_grad():
        pred_p = model_p(batch)['pressure'].cpu().numpy()[0]
        
    print('RAW Unnormalized Pred P: min={:.4f}, max={:.4f}, mean={:.4f}'.format(pred_p.min(), pred_p.max(), pred_p.mean()))

if __name__ == '__main__':
    main()
