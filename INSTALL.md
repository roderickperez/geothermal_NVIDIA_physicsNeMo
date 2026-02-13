# PhysicsNeMo Symbolic Installation Guide

This guide provides step-by-step instructions to set up the **NVIDIA PhysicsNeMo Symbolic** environment on **Windows (WSL 2)** and **Ubuntu Native**.

## Prerequisites

- **Python 3.10+** (Recommend 3.10 or 3.12)
- **NVIDIA GPU** with drivers installed.
- **Git LFS** (Large File Storage)

### 1. Install System Dependencies

Before creating the environment, you must install the necessary build tools and Git LFS.

**Ubuntu / WSL2:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential git-lfs
git lfs install
```

---

## Environment Setup (Using `uv`)

We recommend using `uv` for fast and reliable dependency management.

### 1. Install `uv` (if not installed)
```bash
driver_version=$(curl -s https://pypi.org/pypi/uv/json | grep -oP '"version":"\K[^"]+')
curl -LsSf https://astral.sh/uv/${driver_version}/install.sh | sh
```

### 2. Create and Activate Virtual Environment

Run the following commands in your project root:

```bash
# Create the virtual environment
uv venv .venv

# Activate the environment
source .venv/bin/activate
```

> **Tip:** You can use `source .venv/bin/activate` from anywhere if you provide the full path, e.g., `source /path/to/project/.venv/bin/activate`.

---

## Installation Steps

### 1. Install Core Dependencies
Install PyTorch with CUDA support and `ninja` for building extensions.

```bash
# Install PyTorch (CUDA 12.4) and Ninja
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install ninja Cython
```

### 2. Clone and Install PhysicsNeMo Symbolic
```bash
# Clone the repository
git clone https://github.com/NVIDIA/physicsnemo-sym.git
cd physicsnemo-sym

# Install the package (compiles CUDA extensions)
# Note: --no-build-isolation is critical to see the installed Torch/Ninja
uv pip install . --no-build-isolation -v
```

### 3. Install Additional Requirements (for 3D Reservoir Simulation)
If you are running the 3D reservoir simulation example, you need additional libraries:

```bash
# Navigate to the example directory
cd examples/reservoir_simulation/3D

# Install requirements
uv pip install -r requirements.txt

# Install Cupy (required for this specific example)
uv pip install cupy-cuda12x
```

---

## Running the Simulation

### Windows (WSL 2) Specifics
WSL 2 requires an extra step to ensure Numba can find the CUDA libraries.

**ALWAYS run this before starting your script:**
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### Ubuntu Native
No extra export is usually needed if your CUDA toolkit is in the standard path. If you encounter errors, verify your `LD_LIBRARY_PATH` includes your CUDA lib directory.

### Execution Command
```bash
# Ensure you are in the source directory
cd path/to/physicsnemo-sym/examples/reservoir_simulation/3D/src

# Run the Forward Problem
python Forward_problem_PINO.py
```

---

## Troubleshooting

- **`CUDA_ERROR_NO_DEVICE` in WSL2:**
  - Make sure you ran `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`.
- **Missing Data Errors (Training4.mat):**
  - The script may fail to download data automatically.
  - **Fix:** Manually download the files using `gdown` or from the Google Drive links provided in the script, and place them in the `PACKETS` folder.
