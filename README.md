# spherical_cloudy
A spherically symmetric ionization balance code

This code is described in Cooke & Pettini (2016), MNRAS, 455, 1512. If you find this code useful for your work, please cite that work for a description of the code.

As a pre-warning, I wrote this code with the intention that only I would find it useful. So I apologise for the lack of comments and documentation. You should also bear in mind that this is what I call "code" - this code is not a nice piece of well-written "software".

To run this code, you will need:
- numpy
- scipy
- matpotlib
- cython
- astropy

You will need to compile the code first. Go into the source directory and type: `source compile_Jnur.sh` which will compile the cython code (albeit, with a few warnings). To run the code, you will need to make changes to the `run_NFW_models.py` file to indicate the grid range of parameters that you would like to run. The code can be executed by typing `python run_NFW_models.py` on the command line. This code calls function_NFW.py which is the workhorse of the calculation. The code iterates to convergence (with two for loops), and this can take some time. The best practice is (probably) to run one model (preferably where the gas is mostly ionized), and use the output of this as the starting point of the next point in the grid. Unfortunately, you will need to do this manually. So, run the first calculation of the grid (often the one with the lowest halo mass), and wait until this completes. Then, set `loadprev=True` on line ~255 of `function_NFW.py` and make the appropriate changes to the next few lines to make sure that the previously calculated model is loaded (see the examples commented out in the code). This is very klunky code, but it works for me. Hopefully with a bit of fiddling, it will work for you too  :-)

WARNING: Use this code with caution. It was used and tested for the purposes of calculating the deuterium ionization correction. For all other uses, you should check explicitly that the code is giving you the output that you expect, possibly by checking your results in simple test cases with something slightly more advanced, like Cloudy.
