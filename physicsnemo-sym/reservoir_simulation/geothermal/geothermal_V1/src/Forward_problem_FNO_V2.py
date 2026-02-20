# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import physicsnemo
import torch
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import DictGridDataset
from utilities import load_FNO_dataset2, preprocess_FNO_mat
from physicsnemo.sym.models.fno import *
import scipy.io as sio

torch.set_default_dtype(torch.float32)

@physicsnemo.sym.main(config_path="conf", config_name="config_FNO_V2_V2")
def run(cfg: PhysicsNeMoConfig) -> None:
    print("\n|-----------------------------------------------------------------|")
    print("|          TRAIN THE MODEL USING A 3D FNO APPROACH (V7 GEO)       |")
    print("|-----------------------------------------------------------------|")

    # Simplified Path Handling
    packets_dir = to_absolute_path("../PACKETS")
    if not os.path.exists(packets_dir):
        os.makedirs(packets_dir)

    # [FIX] POINT TO THE CORRECT V7 FILES
    train_mat = os.path.join(packets_dir, "Training4.mat")
    test_mat = os.path.join(packets_dir, "Test4.mat")
    
    train_hdf5 = train_mat.replace(".mat", ".hdf5")
    test_hdf5 = test_mat.replace(".mat", ".hdf5")

    # --- STEP 1: DIRECT HDF5 CONVERSION ---
    print(f"\n[1/3] Checking Data Integrity...")
    
    if not os.path.isfile(train_mat):
        # Fallback check for the original names if Training4 isn't found
        fallback = os.path.join(packets_dir, "Training1000.mat")
        if os.path.isfile(fallback):
            print(f"   -> Found Training1000.mat, using that.")
            train_mat = fallback
            train_hdf5 = train_mat.replace(".mat", ".hdf5")
        else:
            raise FileNotFoundError(f"CRITICAL: Could not find {train_mat}. Did you copy the V7 output to PACKETS?")
        
    # Convert Training Data if HDF5 is missing or old
    if not os.path.isfile(train_hdf5) or os.path.getmtime(train_mat) > os.path.getmtime(train_hdf5):
        print(f"   -> Converting {train_mat} to HDF5 (This may take a minute)...")
        preprocess_FNO_mat(train_mat)
    else:
        print(f"   -> Using existing {train_hdf5}")

    # Convert Test Data
    if not os.path.isfile(test_mat):
         # Fallback check
        fallback_test = os.path.join(packets_dir, "Test1000.mat")
        if os.path.isfile(fallback_test):
            test_mat = fallback_test
            test_hdf5 = test_mat.replace(".mat", ".hdf5")

    if not os.path.isfile(test_hdf5) or os.path.getmtime(test_mat) > os.path.getmtime(test_hdf5):
        print(f"   -> Converting {test_mat} to HDF5...")
        preprocess_FNO_mat(test_mat)
    else:
        print(f"   -> Using existing {test_hdf5}")

    # --- STEP 2: DEFINE KEYS & NORMALIZATION ---
    print(f"\n[2/3] Configuring Keys & Normalization...")
    
    # Input Keys: Scale parameters are used for internal normalization info
    input_keys = [
        Key("perm", scale=(2.5, 0.8)),  # Log-scaled in utilities.py
        Key("Q", scale=(1.2, 42.0)),    # Sparse Injector/Producer rates
        Key("Qw", scale=(0.24, 8.5)),   # Injection Enthalpy
        Key("Phi", scale=(0.22, 0.05)), # Porosity
        Key("Time", scale=(0.5, 0.29)), # Normalized Time
        Key("Pini", scale=(250.0, 7.5)),# Initial Pressure
        Key("Tini", scale=(373.0, 22.0))# Initial Temp
    ]
    
    # Output Keys (Pressure and Temperature are normalized in load_FNO_dataset2)
    output_keys_pressure = [Key("pressure", scale=(0.0, 1.0))]
    output_keys_temperature = [Key("temperature", scale=(0.0, 1.0))]

    # --- STEP 3: LOAD DATASETS ---
    print(f"\n[3/3] Loading Datasets...")
    
    # Load Training Data
    invar_train, outvar_train_p, outvar_train_t = load_FNO_dataset2(
        train_hdf5,
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_temperature],
        n_examples=cfg.custom.ntrain,
    )
    
    # Load Test Data
    invar_test, outvar_test_p, outvar_test_t = load_FNO_dataset2(
        test_hdf5,
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_temperature],
        n_examples=cfg.custom.ntest,
    )

    # ===================================================================
    # [CRITICAL FIX] DELTA FORMULATION: AVOID MEAN COLLAPSE
    # ===================================================================
    def convert_to_delta(outvar, invar, target_key, ini_key, delta_key):
        true_val = outvar[target_key]
        ini_val = invar[ini_key]
        
        # Pini is [N, X, Y, Z]. True Pressure is [N, Time, X, Y, Z].
        # We expand Pini to [N, 1, X, Y, Z] so it subtracts across all timesteps.
        if ini_val.ndim == 4:
            ini_val = np.expand_dims(ini_val, axis=1)
            
        # Calculate Delta and assign to new key
        outvar[delta_key] = true_val - ini_val
        
        # Delete the absolute key so Modulus doesn't use it
        del outvar[target_key]

    # Convert Training Set
    convert_to_delta(outvar_train_p, invar_train, "pressure", "Pini", "delta_pressure")
    convert_to_delta(outvar_train_t, invar_train, "temperature", "Tini", "delta_temperature")

    # Convert Test Set
    convert_to_delta(outvar_test_p, invar_test, "pressure", "Pini", "delta_pressure")
    convert_to_delta(outvar_test_t, invar_test, "temperature", "Tini", "delta_temperature")
    # ===================================================================

    # Initialize Datasets
    train_dataset_pressure = DictGridDataset(invar_train, outvar_train_p)
    train_dataset_temperature = DictGridDataset(invar_train, outvar_train_t)
    test_dataset_pressure = DictGridDataset(invar_test, outvar_test_p)
    test_dataset_temperature = DictGridDataset(invar_test, outvar_test_t)

    # --- STEP 4: DEFINE MODEL ---
    steppi = 30 # Matches V7 output time steps
    
    output_keys_p = ["delta_pressure"]
    output_keys_t = ["delta_temperature"]
    
    # Specialized FNO for Pressure
    decoder1 = ConvFullyConnectedArch([Key("z", size=32)], [Key("delta_pressure", size=steppi)])
    fno_pressure = FNOArch(
        [Key(k.name, size=1) for k in input_keys],
        fno_modes=[16, 16, 2], # [FIX] Compat with NZ=5
        dimension=3,
        padding=11,            # [FIX] Safer padding for 3D
        nr_fno_layers=4,
        decoder_net=decoder1,
    )

    # Specialized FNO for Temperature
    decoder2 = ConvFullyConnectedArch([Key("z", size=32)], [Key("delta_temperature", size=steppi)])
    fno_temperature = FNOArch(
        [Key(k.name, size=1) for k in input_keys],
        fno_modes=[16, 16, 2], # [FIX] Compat with NZ=5
        dimension=3,
        padding=11,            # [FIX] Safer padding for 3D
        nr_fno_layers=4,
        decoder_net=decoder2,
    )

    nodes = [
        fno_pressure.make_node("fno_forward_model_delta_pressure"),
        fno_temperature.make_node("fno_forward_model_delta_temperature")
    ]

    # --- STEP 5: SETUP SOLVER ---
    domain = Domain()

    # Constraints (Supervised Learning)
    domain.add_constraint(
        SupervisedGridConstraint(
            nodes=nodes,
            dataset=train_dataset_pressure,
            batch_size=cfg.batch_size.grid,
        ), "supervised_pressure"
    )

    domain.add_constraint(
        SupervisedGridConstraint(
            nodes=nodes,
            dataset=train_dataset_temperature,
            batch_size=cfg.batch_size.grid,
        ), "supervised_temperature"
    )

    # Validation
    domain.add_validator(
        GridValidator(
            nodes,
            dataset=test_dataset_pressure,
            batch_size=cfg.batch_size.test,
            requires_grad=False,
        ), "test_pressure"
    )

    domain.add_validator(
        GridValidator(
            nodes,
            dataset=test_dataset_temperature,
            batch_size=cfg.batch_size.test,
            requires_grad=False,
        ), "test_temperature"
    )

    # Solve
    slv = Solver(cfg, domain)
    slv.solve()

if __name__ == "__main__":
    run()
