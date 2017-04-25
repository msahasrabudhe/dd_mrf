/*
  This file is part of CycleSolver.

  Converts the fast cycle solver into a Python module. 

  CycleSolver is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  CycleSolver is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with CycleSolver. If not, see <http://www.gnu.org/licenses/>.


  The following paper needed to be cited for any publication of research 
  work that uses this software.

  Huayan Wang and Daphne Koller: 
  A Fast and Exact Energy Minimization Algorithm for Cycle MRFs, 
  The 30th International Conference on Machine Learning (ICML 2013)

  Copyright 2013 Huayan Wang <huayanw@cs.stanford.edu>

*/

#ifndef STDIO
#define STDIO
#include <cstdio>
#endif

#include <cstdlib>
#include <vector>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "thirdparty.h"
#include "cycle.h"

#include <Python.h>
#include <arrayobject.h>        // to handle NumPy arrays. 

using namespace CycleSolver;

/* Returns pointer to the data array of a 1D NumPy double array. */
double *double_npy_1darray_to_Carray(PyArrayObject *arrayin)
{
	return (double *)arrayin->data;
}

/* Returns pointer to the data array of a 1D NumPy int array. */
int *int_npy_1darray_to_Carray(PyArrayObject *arrayin)
{
	return (int *)arrayin->data;
}

/* Returns a pointer which can be used to access elements of
   a 2D NumPy double array. 
   data[i][j] will return the element at the i,j-th position. */
double *double_npy_2darray_to_Carray(PyArrayObject *arrayin)
{
	int m, n;
	int i;
	double **data;
	double *a; 

	/* Get the dimensions of the array. */
	m    = arrayin->dimensions[0];
	n    = arrayin->dimensions[1];

	/* Create an array of double *s to refer to each row of the array. */
	data = (double **)malloc(m*sizeof(double *));

	/* Pointer to arrayin-> data as double *. */
	a    = (double *)arrayin->data;
	for(i = 0; i < m; i ++)
	{
		data[m] = a + i*n;
	}

	return data;
}

/* Returns a pointer which can be used to access elements of
   a 2D NumPy int array. 
   data[i][j] will return the element at the i,j-th position. */
int *int_npy_2darray_to_Carray(PyArrayObject *arrayin)
{
	int m, n;
	int i;
	int **data;
	int *a; 

	/* Get the dimensions of the array. */
	m    = arrayin->dimensions[0];
	n    = arrayin->dimensions[1];

	/* Create an array of int *s to refer to each row of the array. */
	data = (int **)malloc(m*sizeof(int *));

	/* Pointer to arrayin-> data as int *. */
	a    = (int *)arrayin->data;
	for(i = 0; i < m; i ++)
	{
		data[m] = a + i*n;
	}

	return data;
}

/* Checks whether the input array is 1-dimensional and of type NPY_DOUBLE. */
int double_1darray(PyArrayObject *vec)
{
	if(vec->descr->type_num != NPY_DOUBLE)
	{
		PyErr_SetString(PyExc_ValueError, 
				"Specified array is not double, when it needs to be.\n");
		return 0;
	}
	if(vec->nd != 1)
	{
		PyErr_SetString(PyExc_ValueError, 
				"Specified array is not 1-dimensional, when it needs to be.\n");
		return 0;
	}
	return 1;
}

/* Checks whether the input array is 2-dimensional and of type NPY_DOUBLE. */
int double_2darray(PyArrayObject *vec)
{
	if(vec->descr->type_num != NPY_DOUBLE)
	{
		PyErr_SetString(PyExc_ValueError, 
				"Specified array is not double, when it needs to be.\n");
		return 0;
	}
	if(vec->nd != 2)
	{
		PyErr_SetString(PyExc_ValueError, 
				"Specified array is not 2-dimensional, when it needs to be.\n");
		return 0;
	}
	return 1;
}

/* Checks whether the input array is 1-dimensional and of type NPY_LONG. */
int int_1darray(PyArrayObject *vec)
{
	if(vec->descr->type_num != NPY_LONG)
	{
		PyErr_SetString(PyExc_ValueError, 
				"Specified array is not int, when it needs to be.\n");
		return 0;
	}
	if(vec->nd != 1)
	{
		PyErr_SetString(PyExc_ValueError, 
				"Specified array is not 1-dimensional, when it needs to be.\n");
		return 0;
	}
	return 1;
}

/* Checks whether the input array is 2-dimensional and of type NPY_LONG. */
int int_2darray(PyArrayObject *vec)
{
	if(vec->descr->type_num != NPY_LONG)
	{
		PyErr_SetString(PyExc_ValueError, 
				"Specified array is not int, when it needs to be.\n");
		return 0;
	}
	if(vec->nd != 2)
	{
		PyErr_SetString(PyExc_ValueError, 
				"Specified array is not 2-dimensional, when it needs to be.\n");
		return 0;
	}
	return 1;
}
static PyObject *
solver(PyObject *self, PyObject *args)
{
	int i; 

	/* Create objects to handle the input lists */
	PyArrayObject * unary_ar;				// Input unary energies.
	PyArrayObject * pairwise_ar; 			// Input pairwise energies.
	PyArrayObject * n_labels_ar;			// The number of labels each variable can take.
	PyArrayObject * labels_; 				// The resulting labelling. 

	/* The arrays which will store the unary energies, pairwise energies, and the number of labels. */
	double *unaries;
	double *pairwise;
	double *n_labels;

	/* The number of elements in these lists. */
	int n_unaries;
	int n_pairwise;
	int n_n_labels;

	/* The data location of the output vector. */
	int *l_data_;

	/* Stores the dimensions of the output labels_ array. */
	int dims[0];

	/* Parse inputs. */
	if (!PyArg_ParseTuple(args, "OOO", &unary_ar, &pairwise_ar, &n_labels_ar))
		/* There was an error. */
		return NULL;

	if(unary_ar == NULL || pairwise_ar == NULL || n_labels_ar == NULL)
		return NULL;

	/* Check whether the input arrays are as needed. */
	if(!double_2darray(unary_ar))
		return NULL;
	if(!double_2darray(pairwise_ar))
		return NULL;
	if(!int_2darray(n_labels_ar))
		return NULL;

	/* Extract the sizes of the arrays. These also tell us the number of 
	   vertices and edges in the graph. 
	   */
	n_unaries  = unary_ar->dimensions[0];
	n_pairwise = pairwise_ar->dimensions[0];
	n_n_labels = n_labels_ar->dimensions[0];

	/* The number of nodes must be equal to the number of edges. */
	assert(n_unaries == n_pairwise);
	/* The number of nodes must equal the length of the n_labels array. */
	assert(n_unaries == n_n_labels);

	/* All of them should be greater than zero. */
	if(n_unaries < 0 || n_pairwise < 0 || n_n_labels < 0)
		/* Raise an error - they are not proper lists */
		return NULL;

	/* Set our required variables */
	unaries   = double_npy_2darray_to_Carray(unary_ar);
	pairwise  = double_npy_2darray_to_Carray(pairwise_ar);
	n_labels  = int_npy_1darray_to_Carray(n_labels_ar);
	dims[0]   = n_unaries;

	/* Create cycle. */
	Cycle c;

	/* Initialise the cycle. */
	c.initialiseCycle(unaries, pairwise, n_labels, n_unaries);

	/* Solve the cycle. Use fast solver. */
	c.runFastSolver(0);

	/* Create output array - assign memory. */
	labels_   = (PyArrayObject *)PyArray_FromDims(1, dims, NPY_LONG);
	/* Get the data loation of labels_. */
	l_data_   = int_npy_1darray_to_Carray(labels_);

	/* Assign the labelling to labels_. */
	for(i = 0; i < n_unaries; i ++)
	{
		l_data_[i] = c.assignment[i];
	}
	
	/* Clean up for cycle. */
	c.freeMemory();
	/* Clean up the memory allocated to the two 2D arrays. */
	free(unaries);
	free(pairwise);

	/* Return the result. */
	return PyArray_Return(labels_);
}

static PyMethodDef FastCycleSolverMethods[] = {
	{"solver", solver, METH_VARARGS, "The Fast Cycle Solver of Wang and Koller."},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initfast_cycle_solver(void)
{
	(void) Py_InitModule("fast_cycle_solver", FastCycleSolverMethods);
	import_array();
}

int 
main(int argc, char *argv[])
{
	/* Pass argv[0] to the Python interpreter. */
	Py_SetProgramName(argv[0]);

	/* Initialise the Python interpreter. Required. */
	Py_Initialize();

	/* Add a static module. */
	initmultarray();

	return 0;
}
