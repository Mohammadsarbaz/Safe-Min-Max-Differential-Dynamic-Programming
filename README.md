# Safe Min-Max Differential Dynamic Programming

Python implementation of **Safe Min-Max Differential Dynamic Programming (Safe Min-Max DDP)** for zero-sum differential games in continuous time. This repository provides the code associated with the following paper:

---

## 📖 Overview

Chance-Constrained Game-Theoretic Differential Dynamic Programming (CC-GT-DDP) is a safe and robust trajectory optimization algorithm that integrates chance constraints into a Min-Max differential game framework. It computes optimal control policies for nonlinear stochastic systems while providing probabilistic safety guarantees and robustness against worst-case disturbances.

---

## 📁 Repository Structure

### 1. `MPC_CC_GT_DDP_Quadrotor` — Safe Navigation of a Quadcopter
Implementation of MPC-CC-GT-DDP applied to a quadcopter system navigating toward a target while safely avoiding obstacles. The method accounts for stochastic uncertainties and enforces probabilistic safety guarantees through chance-constrained optimization.

### 2. `CC_GT_DDP_Pursuit_Evasion` — Pursuit-Evasion Differential Game
Implementation of CC-GT-DDP for a pursuit-evasion scenario, where a pursuer (first Quadcopter) seeks to intercept an evader (second Quadcopter) while accounting for uncertainty and adversarial interactions. The framework generates robust strategies that balance performance and safety under worst-case disturbances.


---

## 🛠️ Requirements
- Python 2022 or later

## 🚀 How to Run
1. Clone or download this repository
2. Open VS code and navigate to the desired system folder
3. Run the `.py` file

## 📊 Results
Figures and simulation results are available in the `figures/` folder.

## 🎥 Video
Implementation video is available in the `videos/` folder.

---

## 📬 Contact
**Mohammad Sarbaz** — mohammad.sarbaz@ou.edu
🔗 [LinkedIn](https://www.linkedin.com/in/mohammad-sarbaz-94256b1b7/) | 🎓 [Google Scholar](https://scholar.google.com/citations?user=St87OnMAAAAJ&hl=en&oi=ao)
