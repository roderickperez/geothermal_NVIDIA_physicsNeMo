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
from physicsnemo.sym.node import Node
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import DictGridDataset
from utilities import load_FNO_dataset2, preprocess_FNO_mat
from ops import dx, ddx
from physicsnemo.sym.models.fno import *
from typing import Dict

torch.set_default_dtype(torch.float32)

# --- PHYSICS DEFINITION (Synchronized with V7 Normalization) ---
class Geothermal_Physics(torch.nn.Module):
    "Custom Geothermal Physics PDE definition for PINO"

    def __init__(self, UIR, pini_alt, tini_alt, LUB, HUB, aay, bby, MAXZ, nx, ny, nz):
        super().__init__()
        self.UIR = UIR
        self.pini_alt = pini_alt
        self.tini_alt = tini_alt
        self.LUB = LUB
        self.HUB = HUB
        self.aay = aay
        self.bby = bby
        self.MAXZ = MAXZ
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [B, T, NZ, NX, NY]
        u = input_var["pressure"] * 200.0 + 100.0 
        sat = input_var["temperature"] * 100.0 + 273.0 
        
        # Sources and Properties (Static or Broadcastable)
        finwater = input_var["Qw"] * self.UIR 
        poro = input_var["Phi"]
        dt = input_var["Time"]
        
        B, T, NZ, NX, NY = u.shape
        dtin = dt * self.MAXZ
        dxf = 1.0 / NX # X spacing

        # Flatten B,T to process all steps at once
        u_flat = u.view(-1, 1, NZ, NX, NY)
        sat_flat = sat.view(-1, 1, NZ, NX, NY)
        
        # Horizontal Derivatives (NX=dim 1, NY=dim 2 in spatial space)
        # Spatial dims in ops.py are 0, 1, 2 for NZ, NX, NY
        dudx = dx(u_flat, dx=dxf, channel=0, dim=1, order=1, padding="replication")
        dudy = dx(u_flat, dx=dxf, channel=0, dim=2, order=1, padding="replication")
        dduddx = ddx(u_flat, dx=dxf, channel=0, dim=1, order=1, padding="replication")
        dduddy = ddx(u_flat, dx=dxf, channel=0, dim=2, order=1, padding="replication")

        # Darcy Residual (Simplified Physics)
        p_loss_flat = (dudx + dduddx + dudy + dduddy) * 1e-5
        p_loss = p_loss_flat.view(B, T, NZ, NX, NY)

        # Energy Residual (Temporal Derivative)
        dta = dtin # Total time period handled by Time key
        dsw = sat[:, 1:, ...] - sat[:, :-1, ...] 
        dsw = torch.cat([dsw, dsw[:, -1:, ...]], dim=1)
        
        # Broadcast Qw if it's static (B, 1, NZ, NX, NY)
        if finwater.shape[1] == 1 and T > 1:
            finwater = finwater.expand(-1, T, -1, -1, -1)
            
        energy_residual = poro * (dsw / (dta + 1e-6)) - finwater
        
        return {
            "f_pressure": p_loss,
            "f_temperature": energy_residual
        }

@physicsnemo.sym.main(config_path="conf", config_name="config_PINO")
def run(cfg: PhysicsNeMoConfig) -> None:
    print("\n|-----------------------------------------------------------------|")
    print("|          TRAIN THE MODEL USING A 3D PINO APPROACH (V7 GEO)      |")
    print("|-----------------------------------------------------------------|")

    # Path Handling
    packets_dir = to_absolute_path("../PACKETS")
    train_mat = os.path.join(packets_dir, "Training4.mat")
    test_mat = os.path.join(packets_dir, "Test4.mat")
    
    # Fallbacks
    if not os.path.isfile(train_mat):
        train_mat = os.path.join(packets_dir, "Training1000.mat")
    if not os.path.isfile(test_mat):
        test_mat = os.path.join(packets_dir, "Test1000.mat")
        
    train_hdf5 = train_mat.replace(".mat", ".hdf5")
    test_hdf5 = test_mat.replace(".mat", ".hdf5")

    # Preprocessing
    if not os.path.isfile(train_hdf5) or os.path.getmtime(train_mat) > os.path.getmtime(train_hdf5):
        preprocess_FNO_mat(train_mat)
    if not os.path.isfile(test_hdf5) or os.path.getmtime(test_mat) > os.path.getmtime(test_hdf5):
        preprocess_FNO_mat(test_mat)

    # Keys (Same as FNO)
    input_keys = [
        Key("perm", scale=(2.5, 0.8)),
        Key("Q", scale=(1.2, 42.0)),
        Key("Qw", scale=(0.24, 8.5)),
        Key("Phi", scale=(0.22, 0.05)),
        Key("Time", scale=(0.5, 0.29)),
        Key("Pini", scale=(250.0, 7.5)),
        Key("Tini", scale=(373.0, 22.0))
    ]
    output_keys_pressure = [Key("pressure", scale=(0.0, 1.0))]
    output_keys_temperature = [Key("temperature", scale=(0.0, 1.0))]

    # Load Data
    invar_train, outvar_train_p, outvar_train_t = load_FNO_dataset2(
        train_hdf5, [k.name for k in input_keys], [k.name for k in output_keys_pressure], [k.name for k in output_keys_temperature], n_examples=cfg.custom.ntrain
    )
    invar_test, outvar_test_p, outvar_test_t = load_FNO_dataset2(
        test_hdf5, [k.name for k in input_keys], [k.name for k in output_keys_pressure], [k.name for k in output_keys_temperature], n_examples=cfg.custom.ntest
    )

    # Add zeros for residuals
    outvar_train_p["f_pressure"] = np.zeros_like(outvar_train_p["pressure"])
    outvar_train_t["f_temperature"] = np.zeros_like(outvar_train_t["temperature"])

    train_dataset_p = DictGridDataset(invar_train, outvar_train_p)
    train_dataset_t = DictGridDataset(invar_train, outvar_train_t)
    test_dataset_p = DictGridDataset(invar_test, outvar_test_p)
    test_dataset_t = DictGridDataset(invar_test, outvar_test_t)

    # Models
    steppi = 30
    decoder1 = ConvFullyConnectedArch([Key("z", size=32)], [Key("pressure", size=steppi)])
    fno_pressure = FNOArch([Key(k.name, size=1) for k in input_keys], fno_modes=[16, 16, 2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder1)
    
    decoder2 = ConvFullyConnectedArch([Key("z", size=32)], [Key("temperature", size=steppi)])
    fno_temperature = FNOArch([Key(k.name, size=1) for k in input_keys], fno_modes=[16, 16, 2], dimension=3, padding=11, nr_fno_layers=4, decoder_net=decoder2)

    # Physics Node
    nv = cfg.custom.NVRS
    geothermal_physics = Geothermal_Physics(nv.UIR, nv.pini_alt, nv.tini_alt, nv.LUB, nv.HUB, nv.aay, nv.bby, nv.MAXZ, nv.nx, nv.ny, nv.nz)
    physics_node = Node(inputs=[k.name for k in input_keys] + ["pressure", "temperature"], outputs=["f_pressure", "f_temperature"], evaluate=geothermal_physics, name="geothermal_physics")

    nodes = [fno_pressure.make_node("pino_forward_model_pressure"), fno_temperature.make_node("pino_forward_model_temperature"), physics_node]

    # Solver
    domain = Domain()
    domain.add_constraint(SupervisedGridConstraint(nodes=nodes, dataset=train_dataset_p, batch_size=cfg.batch_size.grid), "supervised_pressure")
    domain.add_constraint(SupervisedGridConstraint(nodes=nodes, dataset=train_dataset_t, batch_size=cfg.batch_size.grid), "supervised_temperature")
    
    domain.add_validator(GridValidator(nodes, dataset=test_dataset_p, batch_size=cfg.batch_size.test, requires_grad=False), "test_pressure")
    domain.add_validator(GridValidator(nodes, dataset=test_dataset_t, batch_size=cfg.batch_size.test, requires_grad=False), "test_temperature")

    slv = Solver(cfg, domain)
    slv.solve()

if __name__ == "__main__":
    run()
