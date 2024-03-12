# AR-SA
 Scenario Approximation based on A-priori Reduced (AR-SA) dataset for Jointly Chance-Constrained (JCC) Direct Currect approximation of Optimal Power Flow (DC-OPF) with Automated Generation Control (AGC). [Data-Driven Chance Constrained Programs over Wasserstein Balls (DD-DRO)](https://pubsonline.informs.org/doi/abs/10.1287/opre.2022.2330)

# Docker
Use `docker build -t optcontrol .` to build image.
Use `bash docker_run.sh` to run the container

# Usage

## Configuration
Configurate experiment with configuration files in `conf` folder ([hydra](https://hydra.cc) backended)

* `grid` - test case, available from `pandapower`
* `estimation` - used to choose recorded multi-start results from `main_dro.py` - indentify file needed based on maximal number of sapmles used
* `solution` - configures parameters of JCC problem and DD-DRO parameters
* `data` - scaling of fluctuations (1 is as in paper numerical section)
* `paths` - paths that define target location for results of computations

## Run

* Use `main_dro.py` to simulate experiments from paper
* Use `solution_estimation.py` to make plots that analyze results of simulations
