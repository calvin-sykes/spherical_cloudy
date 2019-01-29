# For GNU compiler
#python cython_fns_setup.py build_ext --inplace 

# For Intel compiler
CC=icc LINKCC=icc LDSHARED="icc -shared" python cython_fns_setup.py build_ext --inplace
