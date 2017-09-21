/* A small header defining functions to easily handle Python Numpy arrays. */

#include <Python.h>
#include <arrayobject.h>

#ifndef _PYNUMPY_H
#define _PYNUMPY_H

/* The int array used by NumPy. */
typedef long int __NPY_INT;

/* Returns pointer to the data array of a 1D NumPy double array. */
double *double_npy_1darray_to_Carray(PyArrayObject *arrayin)
{
	return (double *)arrayin->data;
}

/* Returns pointer to the data array of a 1D NumPy int array. */
__NPY_INT *int_npy_1darray_to_Carray(PyArrayObject *arrayin)
{
	return (__NPY_INT *)arrayin->data;
}

/* Returns a pointer which can be used to access elements of
   a 2D NumPy double array. 
   data[i][j] will return the element at the i,j-th position. */
double **double_npy_2darray_to_Carray(PyArrayObject *arrayin)
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
		data[i] = a + i*n;
	}

	return data;
}

/* Returns a pointer which can be used to access elements of
   a 2D NumPy int array. 
   data[i][j] will return the element at the i,j-th position. */
__NPY_INT **int_npy_2darray_to_Carray(PyArrayObject *arrayin)
{
	int m, n;
	int i;
	__NPY_INT **data;
	__NPY_INT *a; 

	/* Get the dimensions of the array. */
	m    = arrayin->dimensions[0];
	n    = arrayin->dimensions[1];

	/* Create an array of int *s to refer to each row of the array. */
	data = (__NPY_INT **)malloc(m*sizeof(__NPY_INT *));

	/* Pointer to arrayin-> data as int *. */
	a    = (__NPY_INT *)arrayin->data;
	for(i = 0; i < m; i ++)
	{
		data[i] = a + i*n;
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

#endif  				/* #ifndef _PYNUMPY_H */
