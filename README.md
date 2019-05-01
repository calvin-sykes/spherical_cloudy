# spherical_cloudy
A spherically symmetric ionization balance code.

This code is primarily described in Cooke & Pettini (2016), MNRAS, 455, 1512. Additional modifications are detailed in Sykes et. al. (2019), MNRAS, in press.
If you find this code useful for your work, please cite these works for a description of the code.

To run this code, you will need Python 2 and the following modules:
- numpy
- scipy
- matplotlib
- cython
- astropy

You will need to compile the Cython code first. Go into the source directory and type: `source compile_cython.sh` (the warnings are expected).
The code is controlled using `.ini`-format configuration files. The available options and a description of each are found in options.py, and the code will use the `input/defaults.ini` file to load a default setting for each value not explicitly set in your configuration file.

To run the code, execute `python run_models.py <your_configuration_file.ini>`. Plane-parallel models can optionally be run in parallel, which requires MPI4py and a working MPI installation to be installed. The code can then be run in parallel by setting "pp_para = True" under [run], and executing the command `python -m mpi4py.futures run_models.py <your_configuration_file.ini>` using your normal MPI wrapper script (mpirun, mpiexec etc.). Note that the individual MPI ranks will still perform internal parallelisation via OpenMP unless configured otherwise.

WARNING: Use this code with caution. It was used and tested for the purposes of calculating the deuterium ionization correction, and to model fluorescent emission from gas bound to minihaloes. For any other use, you should check explicitly that the code is producing the output that you expect by checking your results for simple test cases with something more advanced like Cloudy.
