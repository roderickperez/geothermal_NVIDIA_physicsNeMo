import numpy as np
from utilities import load_FNO_dataset2

inv, _, _ = load_FNO_dataset2("../PACKETS/Test4.hdf5", ["perm", "Q"], ["pressure"], ["temperature"], n_examples=1)
q_map = inv["Q"][0, 0, :, :, 1] # First sample, Q channel, Layer 2

# Find injector (max positive Q)
inj_idx = np.unravel_index(np.argmax(q_map), q_map.shape)
inj_rate = q_map[inj_idx]

# Find producer (min negative Q)
prod_idx = np.unravel_index(np.argmin(q_map), q_map.shape)
prod_rate = q_map[prod_idx]

print(f"Layer 2 Injector at {inj_idx} with rate {inj_rate}")
print(f"Layer 2 Producer at {prod_idx} with rate {prod_rate}")
