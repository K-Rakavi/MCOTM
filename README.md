# MCOTM: Mobility-aware Computation Offloading and Task Migration for Edge Computing in IIoT

This repository contains the full implementation of **MCOTM**, a deep reinforcement learning-based framework for intelligent task offloading and migration in **Industrial Internet of Things (IIoT)** environments using edge computing. The framework combines **trajectory prediction**, **resource usage forecasting**, and **policy optimization** to minimize task turnaround time, energy consumption, and migration overhead.

MCOTM leverages **Lagrange interpolation** for predicting device movement, **LSTM neural networks** for forecasting system resource usage, and **Deep Deterministic Policy Gradient (DDPG)** for learning optimal offloading and migration policies.

---

## Features

* Mobility Prediction using Lagrange interpolation based on historical device positions
* Resource State Forecasting with Long Short-Term Memory (LSTM) models
* Reinforcement Learning-based Optimization using DDPG to handle continuous action spaces
* Demonstrated improvements:

  * Up to 42% reduction in task turnaround time
  * At least 10% reduction in energy consumption
  * Maintains a task migration rate of approximately 50%

---

## Project Structure

```bash
.
├── original_final_mctom.py        # Core implementation of MCOTM algorithm
├── updated_pod_list.csv           # Input CSV file (based on Alibaba Cluster Trace)
├── mcotm_results.png              # Visualization of performance metrics
└── README.md                      # Project overview and documentation
```

---

## Dependencies

```bash
pip install tensorflow>=2.0
pip install numpy
pip install pandas
pip install scipy
pip install scikit-learn
pip install matplotlib
```

Alternatively, you can install all requirements with:

```bash
pip install -r requirements.txt
```

To generate `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## Input Dataset

The model uses a preprocessed version of the **Alibaba Cluster Trace** dataset. The input CSV should contain the following columns:

* cpu\_milli
* gpu\_milli
* memory\_mib

Ensure this file is accessible and update the path in the script if needed:

```python
csv_path = "updated_pod_list.csv"
```

---

## Execution

Run the main script:

```bash
python original_final_mctom.py
```

The script will:

* Train an LSTM model on system resource data
* Simulate edge server and device layout
* Predict device positions
* Optimize decisions using DDPG
* Generate a performance summary and output plot

Results are saved as `mcotm_results.png`.

---

## Output

The following metrics are visualized:

* Episode-wise average reward
* Number of task migrations
* Average task turnaround time
* Energy consumption per episode

---

## Citation

If you use this implementation in your research, please cite:

> W. Qin et al., "MCOTM: Mobility-aware computation offloading and task migration for edge computing in industrial IoT," *Future Generation Computer Systems*, vol. 151, pp. 232–241, 2024.

---






