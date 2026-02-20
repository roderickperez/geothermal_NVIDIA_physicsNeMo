import os
import glob
import matplotlib.pyplot as plt
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def read_tb_log(file):
    if not os.path.exists(file):
        print(f"Warning: File not found {file}")
        return None
    size_mb = os.path.getsize(file) / (1024 * 1024)
    print(f"Reading: {file} ({size_mb:.2f} MB)")
    
    ea = EventAccumulator(file)
    ea.Reload()
    
    data = {}
    tags = ea.Tags()
    
    # Check scalars
    if 'scalars' in tags:
        for tag in tags['scalars']:
            events = ea.Scalars(tag)
            data[tag] = ([e.step for e in events], [e.value for e in events])
            
    # Check tensors (PhysicsNeMo-Sym often stores metrics here as 0-d tensors)
    if 'tensors' in tags:
        for tag in tags['tensors']:
            events = ea.Tensors(tag)
            try:
                # Direct protobuf float_val extraction (no tensorflow required)
                steps = [e.step for e in events]
                values = [e.tensor_proto.float_val[0] for e in events if len(e.tensor_proto.float_val) > 0]
                if values:
                    data[tag] = (steps, values)
            except Exception as e:
                pass
                
    return data

def plot_all(models, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    logs = {}
    for name, path in models.items():
        logs[name] = read_tb_log(path)

    # Key Metrics to compare
    metrics = [
        "Train/loss_aggregated", 
        "Train/loss_pressure", 
        "Train/loss_temperature", 
        "Train/loss_f_temperature"
    ]
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        has_data = False
        for name, data in logs.items():
            if data and metric in data:
                steps, values = data[metric]
                plt.plot(steps, values, label=name, linewidth=2, alpha=0.8)
                has_data = True
        
        if not has_data:
            plt.close()
            continue

        plt.yscale('log')
        plt.xlabel("Steps")
        plt.ylabel(metric)
        plt.title(f"Comparison: {metric}")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        
        safe_name = metric.replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"compare_{safe_name}.png"))
        print(f"Saved compare_{safe_name}.png")
        plt.close()

if __name__ == "__main__":
    MODELS = {
        "M0: FNO": "outputs/Forward_problem_FNO/ResSim/events.out.tfevents.1771263223.DESKTOP-VBGINCF.13730.0",
        "M1: PINO (Baseline)": "outputs/Forward_problem_PINO/ResSim/events.out.tfevents.1771218602.DESKTOP-VBGINCF.8431.0",
        "M3: Curriculum (Data->Phys)": "outputs/Forward_problem_PINO_CL/ResSim_M3_Curriculum/events.out.tfevents.1771292080.DESKTOP-VBGINCF.4023.0",
        "M4: Reverse (Phys->Data)": "outputs/Forward_problem_PINO_RL/ResSim_M4_Reverse/events.out.tfevents.1771317196.DESKTOP-VBGINCF.9710.0"
    }
    
    plot_all(MODELS, "COMPARE_RESULTS/Global_Comparison")
