# 🩺 Warm-Started Reinforcement Learning for Iterative 3D–2D Liver Registration

This repository contains code and data related to a **patient-specific 3D–2D liver registration** framework based on **reinforcement learning (RL)**.  
We take **patient 4** as an example to demonstrate the training and registration workflow.

---

## 📁 Folder Overview

- **`RL_discrete/`**  
  Contains registration results obtained using the RL-based method.  
  Please note that the folder also includes **`patient2`**, which is **not** part of our experimental results.  
  It is included **only to allow the computation of TRE (Target Registration Error)**, since the evaluation framework requires a complete patient dataset.  
  Therefore, the results for `patient2` are **not representative** and should **not** be used for analysis.

- **`mask/`** and **`contour/`**  
  Contain manually segmented data for **patient 4**.

- **`refinenetppo_auto.py`**  
  This is the **training script** for the reinforcement learning–based registration model.

---

## 🧠 Method Overview

This project implements a **warm-started RL strategy** for iterative 3D–2D liver registration, where reinforcement learning is used to optimize transformation parameters in a **patient-specific** manner.  
The framework demonstrates how RL can be applied to medical image registration tasks with limited data per patient.

---

## 📊 Dataset

The dataset used in this work originates from the **Liver Registration Evaluation dataset**, available at:  
🔗 [https://encov.ip.uca.fr/ab/code_and_datasets/datasets/llr_reg_evaluation_by_lus/index.php](https://encov.ip.uca.fr/ab/code_and_datasets/datasets/llr_reg_evaluation_by_lus/index.php)
