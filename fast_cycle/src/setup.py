from distutils.core import setup, Extension

module1 = Extension('fast_cycle_solver', 
                    sources = ['fast_cycle.cpp'],
					include_dirs = ['/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/'])

setup (name = 'fast_cycle_solver', 
       version = '1.0',
	   description = 'A Python module to use Wang and Koller\'s fast cycle solver.',
	   ext_modules = [module1])
