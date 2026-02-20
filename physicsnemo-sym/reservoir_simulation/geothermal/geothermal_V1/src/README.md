# Geothermal Reservoir Forward Modelling with Physics Informed Neural Operator (PINO) - 3D Adaptation

![Geothermal Energy](https://www.dgidocs.info/slider/images/media/resources_reservoirsim.jpg)

## Geothermal Pressure and Temperature Surrogate Modelling
This project adapts the NVIDIA PhysicsNeMo Black Oil simulator for Geothermal applications. The goal is to accelerate the prediction of **Pressure** and **Temperature** evolution in a reservoir, which is critical for optimization of geothermal assets and energy extraction.

### The "Geothermal Trick"
To bridge the gap between traditional Black Oil simulation (Pressure/Saturation) and Geothermal simulation (Pressure/Temperature) without a complete core code rewrite, we adopt a **Mental Mapping** strategy:

| NVIDIA Config Label | Geothermal Reality | Notes |
| :--- | :--- | :--- |
| **`pressure`** | **Pressure** | Same physics (Darcy flow). |
| **`temperature`** | **Temperature** | Native geothermal channel. |
| **`permeability`** | **Permeability** | Same petrophysical property. |
| **`Q` (Source)** | **Mass Rate** | Injector/Producer mass rates. |
| **`Qw` (Water Source)** | **Injection Enthalpy** | Energy injected into the system. |

### Why FNO (Fourier Neural Operator)?
We prioritize **FNO** for the initial geothermal breakthrough:
1.  **Purely Data-Driven**: FNO maps Input A to Output B by learning patterns. It does not "care" if the second output channel is labeled "Saturation" as long as it behaves consistently as Temperature.
2.  **Stable and Fast**: FNO avoids the hardcoded Buckley-Leverett (Saturation) physics constraints found in the current PINO implementation, which would otherwise conflict with Thermal Diffusion laws.

### Optimized Geothermal Sector Parameters (32x32x5)
To ensure the AI learns realistic subsurface physics for a sedimentary doublet (e.g., in the Netherlands or Munich), we adopt the following sector-specific configuration:

#### 1. Temporal Scale: 30-Year Lifecycle
- **Reasoning**: Geothermal projects are typically financed on a 30-year lifecycle. 
- **The Physics**: Heat moves slowly in porous media due to the **retardation factor**; the "Cold Front" travels much slower than the fluid itself.
- **AI Configuration**: 
### 5. Normalization & Preprocessing (On-the-Fly)
**Update**: We do **NOT** normalize the data in the generator. Instead, we apply transformations *during data loading* (`utilities.py`) to keep the raw data pristine. 

**A. Log-Scaling for Permeability**
*   **Why?**: Permeability varies by orders of magnitude (5 mD to 8000 mD). A neural network will naively focus on the large values (8000) and ignore the flow barriers (5), which is disastrous for fluid dynamics.
*   **How?**: We apply $x' = \log_{10}(x)$. This compresses the range to $\approx [0.7, 3.9]$, creating a balanced distribution for the AI to learn.

**C. Sparsity-Preserving Scaling (for Q and Qw)**
*   **The Problem**: Standard Min-Max scaling $(x - min) / (max - min)$ turns zeros into negative numbers if the minimum is positive (e.g., $303K$). For sparse fields, this creates a "false signal" in the reservoir background.
*   **The Fix**: We use **Ratio Scaling** for sparse channels:
    *   **Inj Rate (Q)**: $x' = x / 2000.0$ (Maps $0 \rightarrow 0$, $1500 \rightarrow 0.75$).
    *   **Inj Temp (Qw)**: $x' = x / 373.0$ (Maps $0 \rightarrow 0$, $303K \rightarrow 0.81$).

**D. Field Normalization (Shift & Scale)**
To keep values near the unit range $[0, 1]$, we use standardized mappings:
*   **Temperature (Tini & Output)**: $T_{norm} = (T - 273.0) / 100.0$ (Maps $373K \rightarrow 1.0$).
*   **Pressure**: $P_{norm} = (P - 100.0) / 200.0$ (Maps $150 \text{--} 300\text{ bar} \rightarrow \approx 0.25 \text{--} 1.0$).

### 6. Training Configuration
*   **Max Steps (Epochs)**: The config is set to **30,000 Steps**. 
    *   With a batch size of 32 and 800 samples, one "Epoch" is $\approx 25$ steps.
    *   Total Epochs $\approx 1200$.
*   **Early Stopping**: There is **NO explicit early stopping** configured in the default script. We rely on:
    1.  **Scheduler**: `tf_exponential_lr` decays the learning rate (0.95 every 1000 steps).
    2.  **Manual Monitoring**: Watch TensorBoard. When Validation Loss plateaus or rises (overfitting), stop the training manually.

## Loss Function Analysis: Black Oil vs Geothermal

### Original Black Oil Loss (PINO)
The original PINO uses a composite loss:
$$\mathcal{L} = \omega_{data} \mathcal{L}_{data} + \omega_{physics} (\mathcal{L}_{mass} + \mathcal{L}_{darcy})$$
*   **Physics Term**: Enforces the **Buckley-Leverett equation**, which describes immiscible displacement (Water pushing Oil).
*   **Goal**: Ensure water and oil conservation laws are met.

### Geothermal Adaptation (Current Setup)
For this version, we are solving for **Heat Diffusion**, which follows a different PDE than Oil/Water displacement.
**CRITICAL CHANGE**: We have **DISABLED** the Physics Loss (`f_pressure=0.0`, `f_temperature=0.0`) in `config_PINO.yaml`.

**Why?**
Using the Buckley-Leverett (Black Oil) physics loss on Temperature data is **physically wrong**. It would force the Temperature field to behave like a shock-front (water displacing oil) rather than a diffusive front (heat conduction/convection).

**Current Loss Function (for both FNO & PINO):**
$$\mathcal{L}_{geo} = \lambda_{P} || P_{pred} - P_{true} ||^2 + \lambda_{T} || T_{pred} - T_{true} ||^2$$
*   **Type**: Pure Data-Driven (L2/MSE Loss).
*   **Weights**:
    *   $\lambda_{P} = 1.0$ (Pressure is "easy" - diffuses fast).
    *   $\lambda_{T} = 10.0$ (Temperature is "hard" - diffuses slowly). We penalize Temperature errors **10x more** to force the AI to capture the sharp thermal front.
- **Total Days**: 10,950 Days (30 Years)
- **Output Frames**: 30 (to match NVIDIA tensor shapes)
- **Resolution**: **365 days/frame**. This allows the AI to visualize the thermal evolution year-by-year.

#### 2. Grid & Geometry
- **Dimensions**: $NX=32, NY=32, NZ=5$. 
- **Cell Size**: $20m \times 20m \times 20m$ (Total volume: $640m \times 640m \times 100m$).
- **Resolution**: This 32x32x5 grid is optimized for full-field simulation while maintaining manageable training times for FNO/PINO.

#### 2. Grid & Geometry
- **Dimensions**: $NX=32, NY=32, NZ=5$.
- **Cell Size**: $DX=20\text{m}, DY=20\text{m}, DZ=20\text{m}$.
- **Total Volume**: $640\text{m} \times 640\text{m} \times 100\text{m}$.
- **Well Configuration**: Doublet pattern with wells at $(5,5)$ and $(27,27)$, providing a distance of $\approx 450\text{m}$.

#### 3. Operational & Thermodynamic Constraints
- **Initial Conditions**: 250 bar (Hydrostatic) and 90°C - 100°C reservoir temperature.
- **Injection**: Cooled water re-injected at 30°C.
- **Rate Scaling**: To prevent instant thermal breakthrough in a small $640\text{m}$ sector, injection rates are scaled to **300 - 500 m³/day**. This ensures the "Cold Bubble" hits the producer between Year 15 and 20, providing the AI with a complex cooling curve to learn.

#### 4. Petrophysics
- **Porosity**: 15% - 25%.
- **Permeability**: 50 mD (Background) up to 500 mD (Channels).
- **Thermal Props**: Rock Heat Capacity of 2,200 kJ/m³/K and Conductivity of 2.1 W/m/K.

## Surrogate Forward Modelling Approach

The "Classic" Data Loss (Pure FNO) approach is a supervised learning method where the AI mimics the DARTS/Simulator results pixel-by-pixel. 

**Mathematical Formulation:**
$$\mathcal{L}_{total} = \lambda_{P} || P_{pred} - P_{true} ||^2 + \lambda_{T} || T_{pred} - T_{true} ||^2$$

*   **P**: Pressure (Normalized)
*   **T**: Temperature (Normalized).
*   **$\lambda$**: Weights (Importance). We use **$\lambda_{T} = 10.0$** to force the AI to focus on the slower, more complex thermal fronts.


## Getting Started:
These instructions will help you set up and run the geothermal training on your local machine.

### **1. Activate Virtual Environment**
Before running any scripts, you must activate the project's virtual environment from the root directory:
```bash
# From /home/roderickperez/DataScienceProjects/NVIDIA_physicsNemo_Sym/
source .venv/bin/activate
```

# Alternatively, if you are already inside 'src', use:
source ../../../../../../.venv/bin/activate

### **2. Execution (Correct Directory)**
To avoid path resolution errors (like corruption during redownload), you **must** be inside the `src` directory when executing the training scripts:

```bash
cd /home/roderickperez/DataScienceProjects/NVIDIA_physicsNemo_Sym/physicsnemo-sym/examples/reservoir_simulation/geothermal/geothermal_V1/src
```

### **3. Run Training**

#### **Step A: FNO Training (Data-Driven Learner)**
Run this first to learn the base patterns from the simulated dataset.
```bash
python Forward_problem_FNO.py
```

#### **Step B: PINO Training (Physics Finetuner)**
Run this after FNO has converged to incorporate physics-informed constraints (Mass & Energy conservation).
```bash
python Forward_problem_PINO.py
```

## Results
### Summary of Geothermal Model
The surrogate model was trained on 2000 geological models with a grid resolution of **40 x 40 x 3**. 
- **Goal**: Predict the "Thermal Front" and Pressure propagation.
- **Physical Context**: Production wells P1-P4 extract heat, while injectors I1-I4 recycle cooler water.
- **Accuracy**: The FNO surrogate matches the numerical results (DARTS/Simulator) with high precision, capturing the transition from the initial high-temperature state to the cooled state over 3,000 days of simulation.

The results compare the responses from the **FNO Surrogate** (labeled as Pressure and Water Saturation, representing **Temperature**) with the ground-truth numerical solver.

| FNO Pred (Pressure) | Ground Truth (Pressure) | Difference |
| :---: | :---: | :---: |
| ![FNO_P](COMPARE_RESULTS/FNO/Evolution_pressure_3D.gif) | ![True_P](Visuals/Evolution_pressure_3D.gif) | ![Diff_P](COMPARE_RESULTS/FNO/R2L2.png) |

| FNO Pred (Temperature) | Ground Truth (Temperature) | Difference |
| :---: | :---: | :---: |
| ![FNO_T](COMPARE_RESULTS/FNO/Evolution_water_3D.gif) | ![True_T](Visuals/Evolution_water_3D.gif) | ![Diff_T](COMPARE_RESULTS/FNO/R2L2.png) |



## Pretrained models

- Pre-trained models and all necessary files are provided in the script for rapid prototyping & reproduction

- The Inverse_problem.py/Compare_FVM_surrogate.py scripts can be ran without necessary running the forward problem steps.

## Setting up Tensorboard
Tensorboard is a great tool for visualization of machine learning experiments. To visualize the various training and validation losses, Tensorboard can be set up as follows:

- In a separate terminal window, navigate to the working directory of the forward problem run)

- Type in the following command on the command line:

##### RUN
```bash
cd src
tensorboard --logdir=./ --port=7007
```



- To view results, open a web browser and go to the url shown by the command prompt. An example would be: http://localhost:7007/#scalars. A window as shown in Fig. 7 should open up in the browser window.

## Results
### Summary of Numerical Model
The result for the surrogate is shown in Fig. 2(a–d); 500 training samples were used to compute the data losses. The fluid flows from the injectors (downwards facing arrows) towards the producers (upwards facing arrows). The size of the reservoir computational voxel is nx, ny, nz = 32,32,5. Single-phase geothermal physics are considered and the wells (5 wells) are arranged in the geothermal field pattern. The 2,340 days of simulation are simulated. The left column of Fig.2(a-b) are the responses from the surrogate, the middle column are the responses from the finite volume solver and the right column is the difference between each response. For all panels in Fig. 2(a-b), the first row is for the pressure and the second row is for the temperature.

|         FNO             | PINO            |
| --------------------|---------------------|
| ![Image 1][img1]     | ![Image 2][img2]     |
| **Figure 2(a) - Numerical implementation of Geothermal forward simulation. FNO based reservoir forwarding showing the 3D permeability and temperature fields with well locations**  | **Figure 2(b) - Numerical implementation of Geothermal forward simulation. PINO based reservoir forwarding showing the 3D permeability and temperature fields with well locations**  |


[img1]: COMPARE_RESULTS/FNO/Evolution.gif "Numerical implementation of Geothermal forward simulation. FNO based reservoir forwarding showing the 3D temperature evolution with well locations"
[img2]: COMPARE_RESULTS/PINO/Evolution.gif "Numerical implementation of Geothermal forward simulation. PINO based reservoir forwarding showing the 3D temperature evolution with well locations"



|         FNO             | PINO            |
| --------------------|---------------------|
| ![Image 1][img5]     | ![Image 2][img6]     |
| **Figure 2(c) -   FNO- Pressure and temperature R2 and L2 accuracy for the time steps**  | **Figure 2(d) - PINO- Pressure and temperature R2 and L2 accuracy for the time steps**  |
| --------------------|---------------------|


[img5]: COMPARE_RESULTS/FNO/R2L2.png "Numerical implementation of Geothermal forward simulation. FNO based reservoir forwarding showing the 3D temperature evolution with well locations"
[img6]: COMPARE_RESULTS/PINO/R2L2.png "Numerical implementation of Geothermal forward simulation. PINO based reservoir forwarding showing the 3D temperature evolution with well locations"





![alt text](COMPARE_RESULTS/Compare_models.png)*Figure 3(a): Production profile comparison. (red) True model and the 2 surrogates (FNO/PINO) (blue) PINO model. First row is for the bottom-hole-pressure of well injectors (I1-I4), second row is for the oil rate production for the well producers (P1-P4), third row is for the water rate production for the well producers (P1-P4) and the last row is for the water cut ratio of the 4 well producers (P1-P4)*

![alt text](COMPARE_RESULTS/Bar_chat.png)*Figure 3(b): RMSE values showng the 2 surrogates*

![alt text](COMPARE_RESULTS/simp.png)*Figure 4: The tensorboard output of the Run from the PINO experiment(blue) and FNO experiment (orange)*




## Release Notes

**23.01**
* First release 

## Author:
- Clement Etienam- Solution Architect-Energy @NVIDIA  Email: cetienam@nvidia.com

## Contributors:
- Oleg Ovcharenko- NVIDIA
- Issam Said- NVIDIA


## References:
[1] J.-Y. Zhu, R. Zhang, D. Pathak, T. Darrell, A. A. Efros, O. Wang, E. Shechtman, Toward multimodal image-to-image translation, in Advances in Neural Information Processing Systems, 2017, pp. 465–476.

[2] S. Rojas, J. Koplik, Nonlinear flow in porous media, Phys. Rev. E 58 (1998) 4776–4782.doi:10.1103/PhysRevE.58.4776.URL,https://link.aps.org/doi/10.1103/PhysRevE.58.4776

[3] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative adversarial nets, in: Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, K. Q. Weinberger (Eds.), Advances in Neural Information Processing Systems 27, Curran Associates, Inc., 2014, pp. 2672–2680. URL http://papers.nips.cc/paper/5423-generative-adversarial-nets. pdf

[4] M. Raissi, P. Perdikaris, G. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (2019) 686 – 707. doi:https://doi.org/10.1016/j.jcp.2018.10.045.URL,http://www.sciencedirect.com/science/article/pii/ S0021999118307125

[5] M. Raissi, Forward-backward stochastic neural networks: Deep learning of high-dimensional partial differential equations, arXiv preprint arXiv:1804.07010

[6] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics Informed Deep Learning (Part I): Data-driven solutions of nonlinear partial differential equations, arXiv preprint arXiv:1711.10561

[7] Bishop. Christopher M 2006. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag, Berlin, Heidelberg.


[8] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar. Fourier Neural Operator for Parametric Partial Differential Equations. https://doi.org/10.48550/arXiv.2010.088959] Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, Anima Anandkumar.Physics-Informed Neural Operator for Learning Partial Differential Equations. https://arxiv.org/pdf/2111.03794.pdf

# 3D Geothermal PINO Experiments

## Scientific Methodology: Data vs. Physics (Fixed Axis)
To ensure fair comparison, all models train for a total of **60,000 steps**.

| Model ID | Strategy | Phase A (0-30k) | Phase B (30k-60k) | Output Folder |
| :--- | :--- | :--- | :--- | :--- |
| **M0** | FNO Baseline | Data Only | Data Only (Cont.) | `Forward_problem_FNO/ResSim` |
| **M1** | PINO Baseline | Data Only | Data Only (Cont.) | `Forward_problem_PINO/ResSim` |
| **M3** | Curriculum | **Data Only** | **Data + Physics** | `Forward_problem_PINO_CL/ResSim_M3_Curriculum` |
| **M4** | Reverse | **Physics Only** | **Data Only** | `Forward_problem_PINO_RL/ResSim_M4_Reverse` |

### Execution
Run each model independently:
```bash
./train.sh fno      # Run M0
./train.sh pino     # Run M1
./train.sh pino_m3  # Run M3 (2 Phases, auto-resume)
./train.sh pino_m4  # Run M4 (2 Phases, auto-resume)
```

## Advanced Consideration: "Delta Prediction"
If you run the 60k training with `weight_decay: 0.0` and the model still produces flat colors, the issue is fundamental to how you are formulating the problem.

Currently, the model predicts Absolute Pressure (e.g., 250 Bar). Because the background pressure (250) is massive compared to the injection change (+5 Bar), the model ignores the change.

**The "Scientist" Fix:** In this code version, we changed the architecture so the model predicts $\Delta P$ (the change in pressure), not absolute pressure.

The model predicts: `delta_p` (Values ranging from -5 to +5).
You calculate absolute: `final_p = Pini + delta_p`

### The Mathematical Reason Why This Works
When your model predicts Absolute Pressure, it is looking at values around 250 Bar. The actual fluid flow (the injection plume) only changes the pressure by about ~1 to 5 Bar. Because the MSE loss function is trying to optimize the massive 250 Bar background, it treats the tiny 1 Bar plume as "mathematical noise" and ignores it, resulting in a flat, solid-color prediction.

By subtracting the Initial Pressure ($P_{ini}$) before training, the background becomes 0 Bar, and the plume becomes the only signal the network sees. The network is forced to learn the fluid flow, effectively eliminating the risk of Mean Collapse.


## Identifying Dynamic Well Locations for Plotting
Because the DARTS simulation randomizes well coordinates for every sample, you cannot hardcode the injector/producer locations when visualizing your results (e.g., `[1, 24]`).

To accurately overlay the wells on your prediction maps, you must dynamically read the sparse `Q` tensor of that exact sample:

```python
# Extract the Q mass-rate map for your test sample [X, Y, Z]
q_tensor = invar["Q"][idx]
q_sum = np.sum(q_tensor, axis=2) # Sum across Z layers

# Max Positive = Injector. Min Negative = Producer.
inj_loc = np.unravel_index(np.argmax(q_sum), q_sum.shape)
prod_loc = np.unravel_index(np.argmin(q_sum), q_sum.shape)

# Crucial: Matplotlib scatter(x, y) expects (Column, Row).
# unravel_index returns (Row, Column). You must swap them!
plt.scatter(inj_loc[1], inj_loc[0], c='blue', marker='o')
plt.scatter(prod_loc[1], prod_loc[0], c='red', marker='X')
```

