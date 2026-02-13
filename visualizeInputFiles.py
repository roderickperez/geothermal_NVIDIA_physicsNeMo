
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os

def visualize_mat_raw(filepath, sample_idx=0):
    """Visualizes raw Training4.mat or Test4.mat"""
    print(f"Visualizing Raw MAT: {filepath}")
    try:
        data = sio.loadmat(filepath)
        input_data = data['INPUT'][sample_idx]
        output_data = data['OUTPUT'][sample_idx]
        
        # Plot Permeability (Channel 0) and Porosity (Channel 3) from INPUT
        perm = input_data[:, :, 0, 0] # Z-slice 0, Time 0 (Input has no time, just channels)
        # Wait, shape is (N, 7, nx, ny, nz) -> (2000, 7, 40, 40, 3) based on inspection
        # Let's assume (N, C, X, Y, Z)
        perm = input_data[0, :, :, 0] # Channel 0, Z 0
        phi = input_data[3, :, :, 0]  # Channel 3, Z 0
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im1 = axes[0].imshow(perm, cmap='jet', origin='lower')
        axes[0].set_title(f'Sample {sample_idx}: Input Permeability (Slice 0)')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(phi, cmap='jet', origin='lower')
        axes[1].set_title(f'Sample {sample_idx}: Input Porosity (Slice 0)')
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        plt.savefig(f'vis_raw_sample_{sample_idx}.png')
        print(f"Saved vis_raw_sample_{sample_idx}.png")
        plt.close()
        
        # Plot Pressure (last step)
        # Output shape (N, 60, 40, 40, 3) 
        # Channels 0-29: Pressure, 30-59: Saturation? Or alternating?
        # Usually first half is one variable, second is another.
        steps = output_data.shape[0] // 2
        final_pressure = output_data[steps-1, :, :, 0] 
        final_saturation = output_data[-1, :, :, 0]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im1 = axes[0].imshow(final_pressure, cmap='jet', origin='lower')
        axes[0].set_title(f'Sample {sample_idx}: Final Output Pressure (Slice 0)')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(final_saturation, cmap='jet', origin='lower')
        axes[1].set_title(f'Sample {sample_idx}: Final Output Saturation (Slice 0)')
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        plt.savefig(f'vis_raw_output_{sample_idx}.png')
        print(f"Saved vis_raw_output_{sample_idx}.png")
        plt.close()

    except Exception as e:
        print(f"Error visualizing raw {filepath}: {e}")
        import traceback
        traceback.print_exc()

def visualize_processed(filepath, sample_idx=0):
    """Visualizes processed .mat or .hdf5 files"""
    print(f"Visualizing Processed File: {filepath}")
    
    data = {}
    try:
        if filepath.endswith('.mat'):
            mat = sio.loadmat(filepath)
            for k in mat:
                if not k.startswith('__'): data[k] = mat[k]
        elif filepath.endswith('.hdf5'):
            with h5py.File(filepath, 'r') as f:
                for k in f.keys():
                    data[k] = f[k][:] # Load into memory
        
        if 'perm' in data:
            # Shape (N, 1, Z, X, Y) or similar -> (2000, 1, 3, 40, 40)
            perm = data['perm'][sample_idx, 0, 0, :, :] 
            plt.figure(figsize=(5, 4))
            plt.imshow(perm, cmap='jet', origin='lower')
            plt.title(f'Processed Sample {sample_idx}: Permeability')
            plt.colorbar()
            plt.savefig(f'vis_proc_perm_{sample_idx}.png')
            print(f"Saved vis_proc_perm_{sample_idx}.png")
            plt.close()
            
        if 'pressure' in data:
            # Shape (2000, 30, 3, 40, 40)
            press = data['pressure'][sample_idx] 
            steps = [0, press.shape[0]//2, press.shape[0]-1]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for i, step in enumerate(steps):
                im = axes[i].imshow(press[step, 0, :, :], cmap='jet', origin='lower')
                axes[i].set_title(f'Pressure Step {step}')
                plt.colorbar(im, ax=axes[i])
            plt.suptitle(f'Processed Sample {sample_idx} Pressure Evolution')
            plt.tight_layout()
            plt.savefig(f'vis_proc_press_{sample_idx}.png')
            print(f"Saved vis_proc_press_{sample_idx}.png")
            plt.close()
            
    except Exception as e:
        print(f"Error visualizing processed {filepath}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    base_path = "physicsnemo-sym/examples/reservoir_simulation/3D/PACKETS/"
    
    # 1. Visualize Raw Training Data
    training_file = os.path.join(base_path, "Training4.mat")
    if os.path.exists(training_file):
        visualize_mat_raw(training_file, sample_idx=0)
    
    # 2. Visualize Processed Training Data (MAT)
    sim_mat = os.path.join(base_path, "simulationstrain.mat")
    if os.path.exists(sim_mat):
        # Flatten name for clarity
        visualize_processed(sim_mat, sample_idx=0)
