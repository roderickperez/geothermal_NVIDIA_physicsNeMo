import subprocess
import time
import sys

def run_model(name, cmd_args):
    print(f"\n[{name}] Launching Training Pipeline...")
    with open(f"log_{name}.txt", "w") as f:
        process = subprocess.Popen(
            cmd_args,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    print(f"[{name}] Process running with PID {process.pid}")
    return process

def main():
    print("=== GEOTHERMAL BATCH TRAINING SEQUENCER ===")
    print("Executing FNO, PINO, PINO_M3, and PINO_M4...")
    
    # We will launch FNO and PINO sequentially first
    print("\n[BATCH 1] FNO")
    p_fno = run_model("FNO", ["bash", "train.sh", "fno"])
    p_fno.wait()
    print("[BATCH 1] FNO Complete.")
    
    print("\n[BATCH 2] PINO Baseline")
    p_pino = run_model("PINO", ["bash", "train.sh", "pino"])
    p_pino.wait()
    print("[BATCH 2] PINO Baseline Complete.")
    
    print("\n[BATCH 3] PINO M3 Curriculum")
    p_m3 = run_model("PINO_M3", ["bash", "train.sh", "pino_m3"])
    p_m3.wait()
    print("[BATCH 3] PINO M3 Complete.")
    
    print("\n[BATCH 4] PINO M4 Reverse Transfer")
    p_m4 = run_model("PINO_M4", ["bash", "train.sh", "pino_m4"])
    p_m4.wait()
    print("[BATCH 4] PINO M4 Complete.")
    
    print("\n=== ALL TRAINING ROUNDS COMPLETE ===")

if __name__ == '__main__':
    main()
