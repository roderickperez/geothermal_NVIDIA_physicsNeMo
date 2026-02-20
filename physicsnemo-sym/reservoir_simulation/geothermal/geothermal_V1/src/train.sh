#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. Environment Check
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${BLUE}[INFO]${NC} Activating environment..."
    source /home/roderickperez/DataScienceProjects/NVIDIA_physicsNemo_Sym/.venv/bin/activate || echo -e "${BLUE}[WARN]${NC} Could not find .venv."
fi

MODEL=$1

if [ "$MODEL" == "fno" ]; then
    echo -e "${BLUE}[INFO]${NC} Running FNO Baseline..."
    rm -rf outputs/Forward_problem_FNO
    python Forward_problem_FNO.py

elif [ "$MODEL" == "pino" ]; then
    echo -e "${BLUE}[INFO]${NC} Running PINO M1 (Baseline Data)..."
    rm -rf outputs/Forward_problem_PINO/ResSim
    python Forward_problem_PINO.py

elif [ "$MODEL" == "pino_m3" ]; then
    echo -e "${BLUE}[INFO]${NC} Running PINO M3 (Curriculum Learning)..."
    
    # Target Directory matches the one defined in the Config files
    DIR_M3="outputs/Forward_problem_PINO_CL/ResSim_M3_Curriculum"
    
    echo -e "${BLUE}[PHASE A]${NC} Data Only (0-30k steps)..."
    # 1. Clean directory to ensure START FROM ZERO
    rm -rf "$DIR_M3"
    
    # 2. Run Phase A using the correct Hydra flag
    python Forward_problem_PINO_CL.py --config-name config_PINO_M3_PhaseA
    
    echo -e "${BLUE}[PHASE B]${NC} Data + Physics (30k-60k steps)..."
    # 3. Resume (Hydra automatically finds the checkpoint in DIR_M3 and continues)
    python Forward_problem_PINO_CL.py --config-name config_PINO_M3_PhaseB

elif [ "$MODEL" == "pino_m4" ]; then
    echo -e "${BLUE}[INFO]${NC} Running PINO M4 (Reverse Transfer)..."
    
    DIR_M4="outputs/Forward_problem_PINO_RL/ResSim_M4_Reverse"
    
    echo -e "${BLUE}[PHASE A]${NC} Physics Only (0-30k steps)..."
    rm -rf "$DIR_M4"
    python Forward_problem_PINO_RL.py --config-name config_PINO_M4_Reverse_A
    
    echo -e "${BLUE}[PHASE B]${NC} Data Only (30k-60k steps)..."
    python Forward_problem_PINO_RL.py --config-name config_PINO_M4_Reverse_B

else
    echo "Usage: ./train.sh [fno | pino | pino_m3 | pino_m4]"
fi
