import os
import torch
import numpy as np
from utilities import load_FNO_dataset2
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import FNOArch, ConvFullyConnectedArch

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inv, out_p, out_t = load_FNO_dataset2('../PACKETS/Test4.hdf5', ['perm', 'Q', 'Qw', 'Phi', 'Time', 'Pini', 'Tini'], ['pressure'], ['temperature'], n_examples=1)
    
    decoder_p = ConvFullyConnectedArch([Key('z', size=32)], [Key('pressure', size=30)])
    model_p = FNOArch([Key(k, size=1) for k in ['perm', 'Q', 'Qw', 'Phi', 'Time', 'Pini', 'Tini']], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_p).to(device)
    
    decoder_t = ConvFullyConnectedArch([Key('z', size=32)], [Key('temperature', size=30)])
    model_t = FNOArch([Key(k, size=1) for k in ['perm', 'Q', 'Qw', 'Phi', 'Time', 'Pini', 'Tini']], fno_modes=[16,16,2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder_t).to(device)
    
    model_p.load_state_dict(torch.load('outputs/Forward_problem_FNO/ResSim/fno_forward_model_pressure.0.pth', map_location=device, weights_only=False))
    model_t.load_state_dict(torch.load('outputs/Forward_problem_FNO/ResSim/fno_forward_model_temperature.0.pth', map_location=device, weights_only=False))
    
    batch = {k: torch.from_numpy(v[:1]).to(device) for k, v in inv.items()}
    with torch.no_grad():
        pred_p = model_p(batch)['pressure'].cpu().numpy()[0]
        pred_t = model_t(batch)['temperature'].cpu().numpy()[0]
        
    print('RAW Pred P: min={:.4f}, max={:.4f}, mean={:.4f}'.format(pred_p.min(), pred_p.max(), pred_p.mean()))
    print('RAW Pred T: min={:.4f}, max={:.4f}, mean={:.4f}'.format(pred_t.min(), pred_t.max(), pred_t.mean()))
    
    true_p = out_p['pressure'][0]
    true_t = out_t['temperature'][0]
    print('RAW True P (Normed?): min={:.4f}, max={:.4f}, mean={:.4f}'.format(true_p.min(), true_p.max(), true_p.mean()))
    print('RAW True T (Normed?): min={:.4f}, max={:.4f}, mean={:.4f}'.format(true_t.min(), true_t.max(), true_t.mean()))

if __name__ == '__main__':
    main()
