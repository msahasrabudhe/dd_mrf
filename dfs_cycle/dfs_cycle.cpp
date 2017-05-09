/* Detect a cycle in a graph no longer than a given length, 
   and containing a particular node, using depth-first search */

#include <stack>
#include <vector>
#include <cstdlib>
#include "../PyNumpy.h"

using namespace std;

/* Define a struct to hold elements of the stack. */
typedef struct it {
	int node;
	vector<int> path;
	vector<bool> visited;
} state;

vector<int> _dfs_cycle(__NPY_INT **adj_mat, int n_nodes, int root, int max_length)
{
	int i;

	/* Variables for intermediate processing. */
	int          cur_node;
	vector<int>  cur_path;
	vector<bool> cur_visited;
	state        cur_state;

	/* Current required path and its length. */
	vector<int> r_path;
	int         r_length = 0;

	/* Create a visited array. */
	vector<bool> visited(n_nodes, false);

	/* Create a path vector for root. */
	vector<int> path;
	path.push_back(root);

	/* Set the root to visited. */
	visited[root] = true;

	/* Create a stack */
	stack<state> s;

	/* Create a state instance for the root */
	state              root_state;
	root_state.node    = root;
	root_state.path    = path;
	root_state.visited = visited;
	
	/* Create a state instance to push neighbours. */
	state              n_state;

	/* Push root_state into stack. */
	s.push(root_state);	

	while(!s.empty())
	{
		/* Get the next node. */
		cur_state = s.top();
		s.pop();

		/* Extract elements from cur_state. */
		cur_node    = cur_state.node;
		cur_path    = cur_state.path;
		cur_visited = cur_state.visited;

		/* If current path length is more than max_length, don't bother 
		   adding further elements into the stack. We can purge our search
		   on this branch. */
		if(cur_path.size() > max_length)
			continue;
		
		/* Check if the root is a neighbour of cur_node. If so, 
		   we have found a cycle. */
		if(adj_mat[cur_node][root] && cur_path.size() > 2)
		{
			/* We have found a cycle. */
			if(cur_path.size() > r_length)
			{
				/* Update our longest cycle of length <= max_length. */
				r_length = cur_path.size();
				r_path   = cur_path;
				/* If we find a cycle of longest permissible length, 
				   return it, as there is no need to search more */
				if(r_length == max_length)
					return r_path;
			}
		}

		/* Find the neighbours of cur_node. */
		for(i = n_nodes - 1; i >= 0; i --)
		{
			if(adj_mat[cur_node][i] && !cur_visited[i])
			{
				/* We haven't yet found a cycle. */
				/* But we have found a node we haven't yet visited. */
				n_state.node    = i;
				n_state.path    = cur_path;
				n_state.visited = cur_visited;
				/* Update n_state.path and n_state.visited. */
				n_state.path.push_back(i);
				n_state.visited[i] = true;
					/* Push n_state into stack. */
				s.push(n_state);
			}
		}
	}
	return r_path;
}

static PyObject *
dfs_cycle(PyObject *self, PyObject *args)
{
	int i; 

	/* Inputs to this function. */
	PyArrayObject * adj_mat;
	int             n_nodes;
	int             root;
	int             max_length;

	/* The cycle output by _dfs_cycle. */
	vector<int>     l_cycle;
	/* An __NPY_INT * pointer pointing to the data array in the output vector. */
	__NPY_INT *     l_cycle_data; 
	/* Output variable. */
	PyArrayObject*  l_cycle_;

	/* Array to store the adjacency matrix. */
	__NPY_INT **    c_adj_mat;

	/* Stores the dimensions of the output l_cycle array. */
	int dims[2];

	/* Parse inputs. */
	if (!PyArg_ParseTuple(args, "Oii", &adj_mat, &root, &max_length))
		/* There was an error. */
		return NULL;

	if(adj_mat == NULL)
		return NULL;

	/* Make sure the array has the correct dimensions. */
	if(!int_2darray(adj_mat))
		return NULL;

	/* The number of nodes. */
	n_nodes   = adj_mat->dimensions[0];

	/* Get the pointer to the data array in adj_mat. */
	c_adj_mat = int_npy_2darray_to_Carray(adj_mat);
	
	/* Get a cycle from the graph. */
	l_cycle      = _dfs_cycle(c_adj_mat, n_nodes, root, max_length);
	/* Get the length of the cycle. */
	dims[0]      = l_cycle.size();
	/* Create a variable to output the cycle. */
	l_cycle_     = (PyArrayObject *)PyArray_FromDims(1, dims, NPY_LONG);
	/* Get the data loation of labels_. */
	l_cycle_data = int_npy_1darray_to_Carray(l_cycle_);

	/* Set the elemnts of the output array. */
	for(i = 0; i < dims[0]; i ++)
	{
		l_cycle_data[i] = l_cycle[i];
	}

	/* Clean up memory. */
	free(c_adj_mat);

	/* Return the result. */
	return PyArray_Return(l_cycle_);
}

static PyMethodDef DFSCycleMethods[] = {
	{"find_cycle", dfs_cycle, METH_VARARGS, "find_cycle(adj_mat, root, max_depth).\nUse depth-first search to find a cycle in a graph with a maximum given length."},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initdfs_cycle(void)
{
	(void) Py_InitModule("dfs_cycle", DFSCycleMethods);
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
	initdfs_cycle();

	return 0;
}
