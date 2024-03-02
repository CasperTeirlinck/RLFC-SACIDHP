# Reinforcement Learning Flight Control: Hybrid Soft Actor-Critic and Incremental Dual Heuristic Programming for Fault-Tolerance

[![Python](https://shields.io/badge/python-3.9-blue.svg?style=for-the-badge)]()
[![License](https://shields.io/badge/Licence-MIT-green?style=for-the-badge)]()

Authors: [Casper Teirlinck](https://github.com/CasperTeirlinck) &nbsp; [![LinkedIn](https://shields.io/badge/LinkedIn--blue?style=social&logo=linkedin)](https://www.linkedin.com/in/casperteirlinck)

> Part of obtaining MSc Thesis at Control & Simulation Departement, Faculty of Aerospace Engineering at Delft University of Technology

Paper published by the American Institute of Aeronautics and Astronautics @ [doi.org/10.2514/6.2024-2406](https://doi.org/10.2514/6.2024-2406)

## System Dependencies

- Python + pip (tested with Python 3.9.9)

## Installation

1. Create and activate a python virtual environment in project root
   ```bash
   python -m venv venv
   ```
2. Install python module

   For use:

   ```bash
   python setup.py install
   ```

   For development:

   ```bash
   python setup.py develop
   ```

## Usage

### Evaluateing pre-trained agents:

- `scripts/evaluate_sac_inner.py`: evaluate SAC attitude controller
- `scripts/evaluate_idhpsac_inner.py`: evaluate hybrid SAC-IDHP attitude controller
- `scripts/evaluate_idhpsac_inner_dc.py`: evaluate hybrid SAC-IDHP attitude controller - decoupled version
- `scripts/evaluate_sac_outer.py`: evaluate SAC/SAC-IDHP cascaded altitude controller

### Training new agents:

- `scripts/train_sac_inner.py`: train a new SAC attitude controller
- `scripts/train_idhpsac_inner.py`: train a new hybrid SAC-IDHP attitude controller
- `scripts/train_idhpsac_inner_dc.py`: train a new hybrid SAC-IDHP attitude controller - decoupled version
- `scripts/train_sac_outer.py`: train a new SAC/SAC-IDHP cascaded altitude controller

## References

- K. Dally and E.-J. Van Kampen, “Soft Actor-Critic Deep Reinforcement Learning for Fault Tolerant Flight Control” [[paper](https://arxiv.org/abs/2202.09262)]

- T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor” [[paper](http://arxiv.org/abs/1801.01290)]

- T. Haarnoja et al., “Soft Actor-Critic Algorithms and Applications” [[paper](http://arxiv.org/abs/1812.05905)]

- S. Heyer, D. Kroezen, and E.-J. Van Kampen, “Online Adaptive Incremental Reinforcement Learning Flight Control for a CS-25 Class Aircraft” [[paper](https://arc.aiaa.org/doi/10.2514/6.2020-1844)]

- D. Kroezen, “Online Reinforcement Learning for Flight Control: An Adaptive Critic Design without prior model knowledge” [[paper](https://repository.tudelft.nl/islandora/object/uuid%3A38547b1d-0535-4b30-a348-67ac40c7ddcc)]

- J. Lee, “Longitudinal Flight Control by Reinforcement Learning: Online Adaptive Critic Design Approach to Altitude Control” [[paper](https://repository.tudelft.nl/islandora/object/uuid%3Ac1201f27-964c-4257-ad65-89224bef94a1)]
