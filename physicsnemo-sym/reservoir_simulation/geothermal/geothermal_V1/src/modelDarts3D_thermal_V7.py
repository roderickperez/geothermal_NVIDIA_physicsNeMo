# modelDarts3D_thermal_V7_FINAL.py
# Version 7 (Geothermalized, Smooth Geology & Clean Dashboard)

import os
import sys
import time
import signal
import warnings
import numpy as np
import pandas as pd
import scipy.io 
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
import traceback
import gc  # [NEW] For forcing memory cleanup

# Silence DARTS warnings
warnings.filterwarnings("ignore", message=".*number of cells looks too big.*")

# --- DARTS IMPORTS ---
from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.engines import value_vector, redirect_darts_output, well_control_iface, sim_params
from darts.physics.geothermal.physics import Geothermal as GeothermalPhysicsBase
from darts.physics.geothermal.geothermal import GeothermalIAPWSProperties
from darts.physics.properties.iapws.custom_rock_property import custom_rock_compaction_evaluator, custom_rock_energy_evaluator
from darts.physics.properties.iapws.iapws_property import (
    iapws_temperature_evaluator, iapws_water_enthalpy_evaluator, 
    iapws_water_density_evaluator, iapws_water_viscosity_evaluator,
    iapws_water_saturation_evaluator, iapws_water_relperm_evaluator,
    iapws_total_enthalpy_evalutor 
)
from darts.physics.properties.basic import ConstFunc

try:
    import pyvista as pv
except ImportError:
    pv = None

# Force software rendering for headless environments (V5 Compat)
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'swrast'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
os.environ['DISPLAY'] = ':99.0'

if pv:
    from pyvista.plotting.utilities.xvfb import PyVistaDeprecationWarning
    warnings.filterwarnings("ignore", category=PyVistaDeprecationWarning)
    try:
        pv.start_xvfb()
    except (OSError, RuntimeError, ImportError):
        pv.OFF_SCREEN = True

def export_wells(reservoir, run_dir):
    if not pv: return
    wells_data = []
    for well in reservoir.wells:
        perfs = [p for p in well.perforations if p[1] >= 0]
        if not perfs: continue
        
        depth_shift = 0.0
        if "depth" in reservoir.global_data:
            d = reservoir.global_data["depth"]
            if np.isscalar(d):
                depth_shift = float(d)
            elif isinstance(d, np.ndarray):
                depth_shift = np.mean(d) if d.size > 0 else 0.0
        
        first_perf_idx = perfs[0][1]
        coords = reservoir.discretizer.centroids_all_cells[first_perf_idx].copy()
        
        # Shift Z to negative depth for visualization
        coords[2] = -1 * (coords[2] + depth_shift)
        
        points_list = []
        surf_pt = coords.copy()
        surf_pt[2] = 0.0 # Wellhead at surface (Z=0)
        points_list.append(surf_pt)
        
        for p in perfs:
            pt = reservoir.discretizer.centroids_all_cells[p[1]].copy()
            pt[2] = -1 * (pt[2] + depth_shift)
            points_list.append(pt)
            
        points = np.array(points_list)
        n_pts = len(points)
        
        lines = np.full((n_pts + 1,), n_pts, dtype=np.int_)
        lines[1:] = np.arange(n_pts)
        
        poly = pv.PolyData()
        poly.points = points
        poly.lines = lines
        
        wtype = 0 if "INJ" in well.name else 1
        poly["WellType"] = np.full(n_pts, wtype, dtype=np.float32)
        
        radius = 10.0 
        tube = poly.tube(radius=radius)
        wells_data.append(tube)
        
    if wells_data:
        combined = wells_data[0]
        if len(wells_data) > 1:
            for w in wells_data[1:]:
                combined = combined.merge(w)
        
        vtk_dir = os.path.join(run_dir, 'vtk_files')
        os.makedirs(vtk_dir, exist_ok=True)
        combined.save(os.path.join(vtk_dir, 'wells.vtp'))

# --- CONFIGURATION (Optimized for Thermal Breakthrough) ---
NX, NY, NZ = 32, 32, 5  # 640m x 640m x 100m
DX, DY, DZ = 20.0, 20.0, 20.0 

TOTAL_DAYS = 10950 # 30 Years (10950)
NUM_TIMESTEPS = 30 # 1 Frame per Year
REPORT_STEP = TOTAL_DAYS / NUM_TIMESTEPS 

ENSEMBLE_SIZE = 1000 # Scaling up for FNO production (1000)
TRAIN_SPLIT = 0.8 

INJ_RATE_BASE = 1500.0 # Boosted for thermal sweep
INJ_TEMP_BASE = 303.15 # 30 C
PROD_BHP_BASE = 150.0 # bar

# Initial Conditions BASE (Will be jittered)
INIT_PRESSURE = 250.0          
INIT_TEMP = 373.15 # 100 C

USE_GPU = False 
# Reduced workers slightly to prevent zombie deadlocks on timeouts
MAX_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.7)) 

# ANSI Colors
C_RED = '\033[1;91m'
C_GREEN = '\033[1;92m'
C_CYAN = '\033[1;96m'
C_WHITE = '\033[1;97m'
C_YELLOW = '\033[1;93m'
C_END = '\033[0m'

DARTS_BANNER = f"""
{C_RED} ██████╗   █████╗  ██████╗  ████████╗ ███████╗
 ██╔══██╗ ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██╔════╝
 ██║  ██║ ███████║ ██████╔╝    ██║    ███████╗
 ██║  ██║ ██╔══██║ ██╔══██╗    ██║    ╚════██║
 ██████╔╝ ██║  ██║ ██║  ██║    ██║    ███████║
 ╚═════╝  ╚═╝  ╚═╝ ╚═╝  ╚═╝    ╚═╝    ███████║
 ╚═════╝  ╚═╝  ╚═╝ ╚═╝  ╚═╝    ╚═╝    ╚══════╝{C_END}

{C_CYAN} >>>>>>>>> {C_WHITE}[ FNO DATASET GENERATOR: V7.2 SMOOTH GEOLOGY ({NX}x{NY}x{NZ}) | GPU: {"ON" if USE_GPU else "OFF"} ]{C_CYAN} <<<<<<<<<<<< {C_END}
"""

# --- HELPER: PROPERTY GENERATOR (Updated V7.2: Smoothed Geology) ---
def generate_property_arrays(nx, ny, nz, seed):
    np.random.seed(seed)
    def jitter(val): return float(val * np.random.uniform(0.8, 1.2))

    # 1. Define available rock types (The "Palette")
    rock_types = [
        {'m_poro': 0.20, 'm_permx': 200.0,  'sigma': 4.0}, # Type A: Avg Sand
        {'m_poro': 0.22, 'm_permx': 800.0,  'sigma': 6.0}, # Type B: Good Sand
        {'m_poro': 0.28, 'm_permx': 2000.0, 'sigma': 6.5}, # Type C: High Perm Channel
        {'m_poro': 0.18, 'm_permx': 300.0,  'sigma': 5.0}, # Type D: Tight Sand
        {'m_poro': 0.10, 'm_permx': 100.0,  'sigma': 3.0}  # Type E: Barrier
    ]
    
    # 2. Random Stratigraphy: Shuffle vertical order per model
    layer_sequence = np.random.choice(rock_types, size=nz, replace=True)
    
    poro_3d  = np.zeros((nz, ny, nx))
    permx_3d = np.zeros((nz, ny, nx))
    permy_3d = np.zeros((nz, ny, nx)) 
    permz_3d = np.zeros((nz, ny, nx)) 
    
    for k in range(nz):
        cfg = layer_sequence[k] # Use randomized sequence
        
        # Base Parameters
        m_poro = jitter(cfg['m_poro'])
        m_permx = jitter(cfg['m_permx'])
        sigma_base = jitter(cfg['sigma'])
        
        # Anisotropy Factors
        sigma_y = sigma_base * np.random.uniform(1.0, 1.3)
        sigma_x = sigma_base * np.random.uniform(1.0, 1.3)
        
        # --- [NEW] POROSITY GENERATION (Multi-Scale) ---
        # 1. Macro Trend (Large blobs)
        macro_noise = gaussian_filter(np.random.normal(0, 1, (ny, nx)), sigma=(sigma_y*1.5, sigma_x*1.5))
        # 2. Micro Texture (Small details)
        micro_noise = gaussian_filter(np.random.normal(0, 1, (ny, nx)), sigma=(sigma_y*0.5, sigma_x*0.5))
        
        # Combine trends
        combined_structure = (macro_noise * 0.7) + (micro_noise * 0.3)
        
        poro_layer = m_poro + (combined_structure * 0.08)
        poro_layer = np.clip(poro_layer, 0.05, 0.40) 
        
        # --- [NEW] PERMEABILITY GENERATION (Smoothed Scatter) ---
        # Base correlation from Porosity
        norm_structure = (poro_layer - m_poro) / 0.08
        
        # [FIX] Smooth Scatter: Prevents pixelated "salt & pepper" look
        # Sigma=1.0 creates soft "patches" of variation instead of single-pixel spikes
        raw_scatter = np.random.normal(0, 0.15, (ny, nx))
        smooth_scatter = gaussian_filter(raw_scatter, sigma=1.0) 
        
        log_perm = np.log10(m_permx) + (norm_structure * 0.85) + smooth_scatter
        px = 10**log_perm
        px = np.clip(px, 5.0, 8000.0)
        
        # Anisotropy
        py = px * np.random.uniform(0.8, 1.2) 
        pz = px * 0.1 * np.random.uniform(0.5, 1.5)
        
        poro_3d[k,:,:]  = poro_layer
        permx_3d[k,:,:] = px
        permy_3d[k,:,:] = py
        permz_3d[k,:,:] = pz
        
    return (poro_3d.flatten().astype(np.float64), 
            permx_3d.flatten().astype(np.float64), 
            permy_3d.flatten().astype(np.float64), 
            permz_3d.flatten().astype(np.float64))

# --- PHYSICS CLASS ---
class CustomGeothermalPhysics(GeothermalPhysicsBase):
    def __init__(self, timer, platform='cpu'):
        super().__init__(timer, 200, 1.0, 1000.0, 10.0, 500000.0, platform=platform, cache=False)
        prop = GeothermalIAPWSProperties()
        prop.Mw = [18.015]
        prop.rock = [value_vector([200.0, 1e-5, 350.0])] 
        prop.rock_compaction_ev = custom_rock_compaction_evaluator(prop.rock)
        prop.rock_energy_ev = custom_rock_energy_evaluator(prop.rock)
        prop.temperature_ev = iapws_temperature_evaluator()
        
        enthalpy_ev = iapws_water_enthalpy_evaluator()
        prop.enthalpy_ev = {'water': enthalpy_ev, 'steam': enthalpy_ev, 'total': iapws_total_enthalpy_evalutor()} 
        
        prop.density_ev = {'water': iapws_water_density_evaluator(), 'steam': iapws_water_density_evaluator()}
        prop.viscosity_ev = {'water': iapws_water_viscosity_evaluator(), 'steam': iapws_water_viscosity_evaluator()}
        prop.saturation_ev = {'water': iapws_water_saturation_evaluator(), 'steam': iapws_water_saturation_evaluator()}
        prop.relperm_ev = {'water': iapws_water_relperm_evaluator(), 'steam': iapws_water_relperm_evaluator()}
        prop.conduction_ev = {'water': ConstFunc(172.8), 'steam': ConstFunc(0.0)}
        self.add_property_region(prop)

# --- MODEL CLASS ---
class SimulationModel(DartsModel):
    def __init__(self, nx, ny, nz, poro, permx, permy, permz, inj_loc, prod_loc, params):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.inj_loc = inj_loc
        self.prod_loc = prod_loc
        self.op_params = params
        
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=DX, dy=DY, dz=DZ,
                                         permx=permx, permy=permy, permz=permz, poro=poro,
                                         hcap=np.full(nx*ny*nz, 2200.0), rcond=np.full(nx*ny*nz, 180.0),
                                         depth=2000)
        self.physics = CustomGeothermalPhysics(self.timer, platform='gpu' if USE_GPU else 'cpu')
        
        if USE_GPU: 
            self.params.linear_type = sim_params.linear_solver_t.gpu_gmres_cpr_amgx_ilu
        else: 
            self.params.linear_type = sim_params.linear_solver_t.cpu_superlu
        
        self.set_sim_params(first_ts=1e-5, mult_ts=1.5, max_ts=10.0, runtime=TOTAL_DAYS,
                            tol_newton=1e-4, tol_linear=1e-5)

    def set_wells(self):
        self.reservoir.discretize()
        self.reservoir.add_well("INJ")
        inj_layers = self.op_params.get('inj_layers', range(1, self.nz + 1))
        for k in inj_layers: 
            self.reservoir.add_perforation("INJ", cell_index=(self.inj_loc[0]+1, self.inj_loc[1]+1, k))
        
        self.reservoir.add_well("PROD")
        prod_layers = self.op_params.get('prod_layers', range(1, self.nz + 1))
        for k in prod_layers: 
            self.reservoir.add_perforation("PROD", cell_index=(self.prod_loc[0]+1, self.prod_loc[1]+1, k))

    def set_well_controls(self):
        for w in self.reservoir.wells:
            if "INJ" in w.name:
                self.physics.set_well_controls(wctrl=w.control, 
                                               control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=self.op_params['inj_rate'], 
                                               phase_name='water', inj_composition=[], 
                                               inj_temp=self.op_params['inj_temp']) 
            else:
                self.physics.set_well_controls(wctrl=w.control, 
                                               control_type=well_control_iface.BHP,
                                               is_inj=False, target=self.op_params['prod_bhp'])

    def set_initial_conditions(self):
        # [UPDATE] USE RANDOMIZED INITIAL CONDITIONS
        self.physics.set_initial_conditions_from_array(
            self.reservoir.mesh, 
            {
                'pressure': self.op_params['init_press'], 
                'temperature': self.op_params['init_temp']
            }
        )

def randomize_well_locations(seed):
    np.random.seed(seed)
    while True:
        inj_loc = [np.random.randint(2, NX-2), np.random.randint(2, NY-2)]
        prod_loc = [np.random.randint(2, NX-2), np.random.randint(2, NY-2)]
        dist = np.sqrt((inj_loc[0]-prod_loc[0])**2 + (inj_loc[1]-prod_loc[1])**2)
        if dist > 16:
            return inj_loc, prod_loc

def randomize_operational_params(seed):
    np.random.seed(seed)
    def get_rand_layers(nz):
        num_perfs = np.random.randint(1, nz + 1)
        return sorted(np.random.choice(range(1, nz + 1), size=num_perfs, replace=False))

    return {
        'inj_rate': INJ_RATE_BASE * np.random.uniform(0.7, 1.3),
        'inj_temp': INJ_TEMP_BASE * np.random.uniform(0.9, 1.1),
        'prod_bhp': PROD_BHP_BASE * np.random.uniform(0.7, 1.3),
        
        # [UPDATE] RANDOMIZED INITIAL CONDITIONS
        'init_press': INIT_PRESSURE * np.random.uniform(0.95, 1.05),
        'init_temp': INIT_TEMP * np.random.uniform(0.95, 1.05),
        
        'inj_layers': get_rand_layers(NZ),
        'prod_layers': get_rand_layers(NZ)
    }

# --- MAIN GENERATOR ---
def timeout_handler(signum, frame):
    raise TimeoutError("Simulation timed out!")

def run_realization(i, realizations_root, progress_dict=None):
    MAX_RETRIES = 10 
    TIMEOUT_SECONDS = 900 
    
    # Stagger start to reduce load spikes
    time.sleep(np.random.uniform(0.5, 3.0)) 
    signal.signal(signal.SIGALRM, timeout_handler)
    
    for attempt in range(MAX_RETRIES):
        m = None # [NEW] Initialize to None for safer cleanup
        if progress_dict is not None:
            if attempt == 0:
                progress_dict[i] = "Init: Geo"
            else:
                progress_dict[i] = -1 * attempt
                time.sleep(2.0) 
            
        seed = 2000 + i + (attempt * 10000)
        np.random.seed(seed)
        
        # 1. Randomized Geology & Rates
        poro, permx, permy, permz = generate_property_arrays(NX, NY, NZ, seed)
        inj_loc, prod_loc = randomize_well_locations(seed)
        op_params = randomize_operational_params(seed)

        if progress_dict is not None and attempt == 0:
            progress_dict[i] = "Init: Phys"

        # 2. Setup Logging
        real_dir = os.path.join(realizations_root, f"model_{i}")
        os.makedirs(real_dir, exist_ok=True)
        
        log_name = "simulation.log" if attempt == 0 else f"simulation_retry_{attempt}.log"
        log_path = os.path.join(real_dir, log_name)
        log_file = open(log_path, "w")
        
        original_stdout_fd = os.dup(sys.stdout.fileno())
        original_stderr_fd = os.dup(sys.stderr.fileno())
        
        try:
            os.dup2(log_file.fileno(), sys.stdout.fileno())
            os.dup2(log_file.fileno(), sys.stderr.fileno())

            signal.alarm(TIMEOUT_SECONDS)
            
            # B. Simulation Model Setup
            m = SimulationModel(NX, NY, NZ, poro, permx, permy, permz, inj_loc, prod_loc, op_params)
            m.init(platform='gpu' if USE_GPU else 'cpu')
            
            if progress_dict is not None and attempt == 0:
                progress_dict[i] = "Init: Output"

            m.set_output(real_dir) 
            export_wells(m.reservoir, real_dir)
            
            # --- C. BUILD INPUT TENSOR ---
            input_tensor = np.zeros((7, NX, NY, NZ), dtype=np.float32)
            input_tensor[0, :, :, :] = permx.reshape(NZ, NY, NX).transpose(2, 1, 0)
            
            # [FIX] Map BOTH Injector (+) and Producer (-) to the Q Channel
            q_map = np.zeros((NX, NY, NZ))
            
            # 1. Map Injector (Positive Rate)
            inj_split_rate = op_params['inj_rate'] / len(op_params['inj_layers'])
            for k in op_params['inj_layers']:
                q_map[inj_loc[0], inj_loc[1], k-1] = inj_split_rate
                
            # 2. Map Producer (Negative Rate / Sink)
            prod_split_rate = -1.0 * op_params['inj_rate'] / len(op_params['prod_layers'])
            for k in op_params['prod_layers']:
                q_map[prod_loc[0], prod_loc[1], k-1] = prod_split_rate
                
            input_tensor[1, :, :, :] = q_map
            
            t_inj_map = np.zeros((NX, NY, NZ))
            for k in op_params['inj_layers']:
                t_inj_map[inj_loc[0], inj_loc[1], k-1] = op_params['inj_temp']
            input_tensor[2, :, :, :] = t_inj_map
            
            input_tensor[3, :, :, :] = poro.reshape(NZ, NY, NX).transpose(2, 1, 0)
            xv, yv, zv = np.meshgrid(np.linspace(0,1,NX), np.linspace(0,1,NY), np.linspace(0,1,NZ), indexing='ij')
            input_tensor[4, :, :, :] = xv 
            
            # [UPDATE] USE RANDOMIZED INITIAL CONDITIONS
            input_tensor[5, :, :, :] = op_params['init_press'] 
            # [FIX] Renamed 'Swini' -> 'Tini' (Geothermal)
            input_tensor[6, :, :, :] = op_params['init_temp'] 
            
            # --- D. BUILD OUTPUT TENSOR ---
            output_tensor = np.zeros((NUM_TIMESTEPS * 2, NX, NY, NZ), dtype=np.float32)
            
            vars_to_eval = m.physics.vars + ['temperature']
            ts, prop_arr = m.output.output_properties(output_properties=vars_to_eval, timestep=0, engine=True)
            for name, arr in [('poro', poro), ('permx', permx)]:
                prop_arr[name] = np.array([arr])
            m.output.output_to_vtk(output_data=[ts, prop_arr], ith_step=0)
    
            for step in range(NUM_TIMESTEPS):
                m.run(REPORT_STEP, verbose=False) 
                
                if progress_dict is not None:
                    progress_dict[i] = step + 1
                
                nb = NX * NY * NZ
                x_raw = np.array(m.physics.engine.X, copy=True)
                res_x = x_raw[:nb * 2] 
                p_raw = res_x[0::2]
                h_raw = res_x[1::2]
                t_raw = (h_raw / 18.015 / 4.187) + 273.15
                
                p_3d = p_raw.reshape(NZ, NY, NX).transpose(2, 1, 0)
                t_3d = t_raw.reshape(NZ, NY, NX).transpose(2, 1, 0)
                
                output_tensor[step, :, :, :] = p_3d
                output_tensor[step + NUM_TIMESTEPS, :, :, :] = t_3d
    
                ts, prop_arr = m.output.output_properties(output_properties=vars_to_eval, timestep=step+1, engine=True)
                for name, arr in [('poro', poro), ('permx', permx)]:
                    prop_arr[name] = np.array([arr])
                m.output.output_to_vtk(output_data=[ts, prop_arr], ith_step=step+1)
            
            signal.alarm(0)
            
            # Save raw numpy files just in case
            np.save(f"{real_dir}/input.npy", input_tensor)
            np.save(f"{real_dir}/output.npy", output_tensor)
            
            if progress_dict is not None:
                progress_dict[i] = "DONE"
                
            # Clean exit
            del m
            m = None
            gc.collect() # [NEW] Force clean memory
            return i, input_tensor, output_tensor

        except Exception as e:
            signal.alarm(0)
            sys.stderr.write(f"\nAttempt {attempt+1} failed: {str(e)}\n")
            traceback.print_exc()
            
            # [NEW] Force kill model on exception to prevent zombie C++ pointers
            if m:
                del m
                m = None
            gc.collect() 
            
            if attempt < MAX_RETRIES - 1:
                if progress_dict is not None:
                    progress_dict[i] = 0 
                continue 
            else:
                raise RuntimeError(f"Model {i} failed after {MAX_RETRIES} attempts.")
        finally:
            os.dup2(original_stdout_fd, sys.stdout.fileno())
            os.dup2(original_stderr_fd, sys.stderr.fileno())
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
            log_file.close()

# --- MAIN GENERATOR ---
def main():
    input_collection = np.zeros((ENSEMBLE_SIZE, 7, NX, NY, NZ), dtype=np.float32)
    output_collection = np.zeros((ENSEMBLE_SIZE, NUM_TIMESTEPS * 2, NX, NY, NZ), dtype=np.float32)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    base_output_dir = "/home/roderickperez/DataScienceProjects/openDARTS/output/geothermalOutput_model_files"
    ensemble_folder_name = f"ensemble_{timestamp}"
    ensemble_root = os.path.join(base_output_dir, ensemble_folder_name)
    realizations_root = ensemble_root
    
    os.makedirs(ensemble_root, exist_ok=True)

    print(DARTS_BANNER)
    print(f"Starting V7.2 GEOTHERMAL Ensemble Simulation: {ENSEMBLE_SIZE} realizations")
    print(f"Grid: {NX}x{NY}x{NZ} | Cell: {DX}x{DY}x{DZ}m")
    print(f"Parallel Workers: {MAX_WORKERS}")
    print(f"Ensemble Root: {ensemble_root}\n")

    print(f"{C_CYAN}Launch sequence initiated...{C_END}")
    start_time = time.time()
    session_start_time = time.time()
    
    # Track when models finish to hide them after X seconds
    completion_tracker = {}
    DONE_DISPLAY_SECONDS = 3.0
    
    # [NEW] Use Context Manager to ensure clean shutdown of Manager process
    with multiprocessing.Manager() as manager:
        progress_dict = manager.dict()
        console = Console()
        
        def generate_dashboard():
            current_time = time.time()
            elapsed = current_time - session_start_time
            
            done_count = sum(1 for v in progress_dict.values() if v == "DONE")
            
            safe_elapsed_min = max(elapsed / 60, 0.001)
            mod_min = done_count / safe_elapsed_min
            
            overall_progress = Progress(
                TextColumn("[bold blue]{task.description}", justify="right"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("completed: {task.completed}/{task.total}"),
                TimeRemainingColumn()
            )
            overall_progress.add_task("Ensemble", total=ENSEMBLE_SIZE, completed=done_count)
            
            table = Table(title="Simulation Cluster Status", show_header=True, header_style="bold magenta")
            table.add_column("Worker", justify="center")
            table.add_column("Model ID", justify="center")
            table.add_column("Progress (Year)", justify="right")
            table.add_column("Status", justify="center")

            # [NEW LOGIC] Filter Items based on completion time
            display_list = []
            current_state = list(progress_dict.items()) # Snapshot
            
            for k, v in current_state:
                if v == "DONE":
                    # If this is the first time we see it done, record time
                    if k not in completion_tracker:
                        completion_tracker[k] = current_time
                    
                    # Only show if finished recently
                    if current_time - completion_tracker[k] < DONE_DISPLAY_SECONDS:
                        display_list.append((k, v))
                else:
                    # Always show items that are pending/running/error
                    display_list.append((k, v))
            
            # Sort by ID
            display_list.sort(key=lambda x: x[0])
            
            # Show top 15 of the filtered list
            display_items = display_list[:15]
            
            for model_id, val in display_items:
                if isinstance(val, int) and val < 0:
                    status = f"[bold red]Timed Out (Retry {-val})[/]"
                    prog_str = " - "
                elif isinstance(val, str):
                    if val == "DONE":
                        status = "[bold green]DONE[/]"
                        prog_str = "30/30"
                    elif val == "Queued":
                        status = "[dim]Queued..."
                        prog_str = " - "
                    else:
                        status = f"[yellow]{val}..."
                        prog_str = " - "
                elif val == 0:
                    status = "[yellow]Initializing..."
                    prog_str = "0/30"
                else:
                    status = "[green]Simulating"
                    prog_str = f"{val}/30"
                    
                table.add_row(f"Model {model_id}", f"model_{model_id}", prog_str, status)

            if len(display_list) > 15:
                table.add_row("...", "...", "...", f"+ {len(display_list)-15} more")

            stats = Panel(
                f"Process Time: {elapsed/60:.1f} min | Throughput: [bold green]{mod_min:.1f}[/] mod/min | [bold cyan]{done_count}/{ENSEMBLE_SIZE}[/] Total",
                title="Performance Metrics", border_style="cyan"
            )
            
            layout = Layout()
            layout.split(Layout(overall_progress, size=3), Layout(table), Layout(stats, size=3))
            return layout

        for i in range(ENSEMBLE_SIZE):
            progress_dict[i] = "Queued"

        # [NEW] Wrap in Try/Except to catch Ctrl+C and kill Zombies
        try:
            with Live(generate_dashboard(), refresh_per_second=2, vertical_overflow="visible") as live:
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(run_realization, i, realizations_root, progress_dict): i for i in range(ENSEMBLE_SIZE)}
                    
                    # Dashboard loop
                    while True:
                        done_tasks = sum(1 for f in futures if f.done())
                        live.update(generate_dashboard())
                        if done_tasks >= ENSEMBLE_SIZE:
                            break
                        time.sleep(0.5)

                    for future in as_completed(futures):
                        try:
                            i, real_input, real_output = future.result()
                            input_collection[i] = real_input
                            output_collection[i] = real_output
                        except Exception as exc:
                            print(f"\n[ERROR] Model {futures[future]} failed: {exc}")
                            
        except KeyboardInterrupt:
            print(f"\n{C_RED}[!] INTERRUPTED BY USER. KILLING ZOMBIE PROCESSES...{C_END}")
            # ProcessPoolExecutor cleans up automatically on exit context, 
            # but manager needs the 'with' block exit to die.
            sys.exit(1)

    end_time = time.time()
    total_min = (end_time - session_start_time) / 60
    console.print(f"\n\n[bold green]Parallel Simulation Cluster Finished in {total_min:.2f} minutes.[/]")

    # 3. SPLIT AND SAVE (UPDATED: Unpacked Format)
    print(f"\nFinalizing Dataset Split and Unpacking...")
    split_idx = int(ENSEMBLE_SIZE * TRAIN_SPLIT)
    
    def save_unpacked_mat(filename, inputs, outputs):
        """Unpacks tensors into named variables expected by FNO/PINO (Geothermal Config)"""
        data_dict = {
            'perm': inputs[:, 0:1, ...], 
            'Q': inputs[:, 1:2, ...], 
            'Qw': inputs[:, 2:3, ...], 
            'Phi': inputs[:, 3:4, ...], 
            'Time': inputs[:, 4:5, ...], 
            'Pini': inputs[:, 5:6, ...], 
            
            # [FIX] Renamed 'Swini' -> 'Tini' (Initial Temperature)
            'Tini': inputs[:, 6:7, ...],
            
            'pressure': outputs[:, 0:30, ...], 
            
            # [FIX] Renamed 'water_sat' -> 'temperature' (The Geothermal Goal)
            'temperature': outputs[:, 30:60, ...] 
        }
        savemat(filename, data_dict)
        print(f"Saved {filename} with keys: {list(data_dict.keys())}")

    training_mat = os.path.join(ensemble_root, f"Training{ENSEMBLE_SIZE}.mat")
    save_unpacked_mat(training_mat, input_collection[:split_idx], output_collection[:split_idx])
    
    test_mat = os.path.join(ensemble_root, f"Test{ENSEMBLE_SIZE}.mat")
    save_unpacked_mat(test_mat, input_collection[split_idx:], output_collection[split_idx:])
    
    print(f"{C_GREEN}SUCCESS!{C_END}")
    print(f"  Ensemble: {ensemble_root}\n")
    
    # Run Verification
    print("Running Verification Script...")
    import subprocess
    subprocess.run(["python3", "scripts/verify_dataset.py", training_mat, test_mat])

if __name__ == "__main__":
    main()