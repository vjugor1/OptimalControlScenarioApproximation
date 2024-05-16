echo grid6_005
python main_dro.py --config-name config_grid6_reduction_005 > grid6_005_stdout.txt 2> grid6_005_stderr.txt
python solution_estimation.py --config-name config_grid6_reduction_005 > grid6_005_est_stdout.txt 2> grid6_005_est_stderr.txt
echo grid6
python main_dro.py --config-name config_grid6_reduction > grid6_stdout.txt 2> grid6_stderr.txt
python solution_estimation.py --config-name config_grid6_reduction > grid6_005_est_stdout.txt 2> grid6_est_stderr.txt

echo grid14_005
python main_dro.py --config-name config_grid14_reduction_005 > grid14_005_stdout.txt 2> grid14_005_stderr.txt
python solution_estimation.py --config-name config_grid14_reduction_005 > grid14_005_est_stdout.txt 2> grid14_005_est_stderr.txt
echo grid14
python main_dro.py --config-name config_grid14_reduction > grid14_stdout.txt 2> grid14_stderr.txt
python solution_estimation.py --config-name config_grid14_reduction > grid14_005_est_stdout.txt 2> grid14_est_stderr.txt

echo grid30_005
python main_dro.py --config-name config_grid30_reduction_005 > grid30_005_stdout.txt 2> grid30_005_stderr.txt
python solution_estimation.py --config-name config_grid30_reduction_005 > grid30_005_est_stdout.txt 2> grid30_005_est_stderr.txt
echo grid30
python main_dro.py --config-name config_grid30_reduction > grid30_stdout.txt 2> grid30_stderr.txt
python solution_estimation.py --config-name config_grid30_reduction > grid30_005_est_stdout.txt 2> grid30_est_stderr.txt