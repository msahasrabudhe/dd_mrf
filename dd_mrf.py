#!/usr/bin/python

# This library calculates an approximation to the optimal of an artibrary energy on a
#   random graph, by splitting the graph into trees based on a greedy strategy, and
#   using the DD-MRF algorithm to solve the resulting dual problem. 
# The graph decomposition is equivalent to a standard LP relaxation of the problem. 

import numpy as np
import multiprocessing
from joblib import Parallel, delayed, cpu_count
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import threading

# To create shared numpy arrays using multiprocessing.
import ctypes

# Max product belief propagation on trees. 
import bp           

# Fast cycle solver. Wang and Koller, ICML 2013.
import fast_cycle_solver as fsc
        
# C++ implememtation of DFS cycle searching. 
from dfs_cycle import find_cycle

#  --- DTYPES ---
# The dtype to use to store energies. 
e_dtype = np.float64
# The dtype to use for labels. That is to say, labels will be stored as this dtype. 
l_dtype = np.int16  
# The dtype to use for node and edge indices
n_dtype = np.int32
# The dtype used when booleans are needed as ints. Should use only 1 byte.
m_dtype = np.int8
# The dtype used for updates to slave energies. These updates are usually small.
# --- Using float64 here anyway, because ctypes.c_float apparently results in 64 bit
#     floats, when it should actually result in 32 bit ones.
u_dtype = np.float64

# List of slave types handled by this module. 
slave_types         = ['free_node', 'free_edge', 'cell', 'tree', 'cycle']
# List of allowed graph decompositions.
decomposition_types = ['tree', 'mixed', 'custom', 'factor']

# Infinity energy. Deliberately introduced to skew primal and dual costs,
#    so that it is easier to debug when optimisation is buggy. 
inf_energy = 1e1000

# Create a multiprocessing.Manager() object for shared lists. 
# This shared list shall be used to record the slaves to be solved. 
#    and shall be used in the function _optimise_slave_mp()
#    to solve slaves in parallel. 
manager           = multiprocessing.Manager()
# Create a shared list which can be used to access the slave list across processes. 
shared_slave_list = manager.list()

class Slave:
    '''
    A class to store a slave. An instance of this class stores
    the list of nodes it contains, the list of edges it contains, 
    the energies corresponding to all of them, and the structure
    of the graph. 
    '''
    def __init__(self, node_list=None, edge_list=None, 
            node_energies=None, n_labels=None, edge_energies=None, graph_struct=None, struct='cell'):
        '''
        Slave.__init__(): Initialise parameters for this slave, if given. 
                          Parameters are None by default. 
        '''
        if struct not in slave_types:
            print 'Slave struct not recognised: %s.' %(struct)
            raise ValueError

        self.node_list      = node_list
        self.edge_list      = edge_list
        self.node_energies  = node_energies
        self.n_labels       = n_labels
        self.edge_energies  = edge_energies
        self.graph_struct   = graph_struct
        self.struct         = struct                # Whether a cell or a tree. 
                                
    def set_params(self, node_list, edge_list, 
            node_energies, n_labels, edge_energies, graph_struct, struct):
        '''
        Slave.set_params(): Set parameters for this slave.
                            Parameters must be specified.
        '''
        if struct not in slave_types:
            print 'Slave struct not recognised: %s.' %(struct)
            raise ValueError

        self.node_list      = node_list
        self.edge_list      = edge_list
        self.node_energies  = node_energies
        self.n_labels       = n_labels
        self.edge_energies  = edge_energies
        self.graph_struct   = graph_struct
        self.struct         = struct

        # The number of nodes and edges. Make code easier to understand. 
        self.n_nodes        = self.node_list.size
        self.n_edges        = self.edge_list.size

        # Generate all label permutations. Saves time by using more memory. 
#       self.all_labellings = _generate_label_permutations(self.n_labels)

        # The max label any node can have. 
        # TODO: Handle this so that sliced matrices are created in Graph._create_..._slaves()
        #    and passed to Slave.set_params. Otherwise, the reshape operation in 
        #    _optimise_cycle() will not work. 
        self.max_n_labels   = np.max(n_labels)

        # These dictionaries enable to determine easily at which 
        #    index in node_list or edge_list, a particular node
        #    or edge is. 
        self.node_map       = {}
        self.edge_map       = {}
        for i in range(self.n_nodes):
            self.node_map[node_list[i]] = i
        for i in range(self.n_edges):
            self.edge_map[edge_list[i]] = i

    def get_params(self):
        '''
        Slave.get_params(): Return parameters of this slave
        '''
        return self.struct, self.node_list, self.edge_list, self.node_energies, self.n_labels, self.edge_energies

    def set_labels(self, labels):
        '''
        Slave.set_labels(): Set the labelling for a slave
        '''
        self.labels = np.array(labels, dtype=l_dtype)

        # Also maintain a dictionary to easily fetch the label 
        #   given a node ID.
        self.label_from_node    = {}
        for i in range(self.n_labels.size):
            n_id = self.node_list[i]
            self.label_from_node[n_id] = self.labels[i]

    def get_node_label(self, n_id):
        '''
        Retrieve the label of a node in the current labelling
        '''
        if n_id not in self.node_list:
            print self.node_list,
            print 'Node %d is not in this slave.' %(n_id)
            raise ValueError
        return self.label_from_node[n_id]


    def optimise(self):
        '''
        Optimise this slave. 
        '''
        if self.struct == 'cell':
            return _optimise_4node_slave(self)
        elif self.struct == 'tree':
            return _optimise_tree(self)
        elif self.struct == 'cycle':
            return _optimise_cycle(self)
        elif self.struct == 'free_node':
            # Simply return the label with the least energy.
            nl = np.argmin(self.node_energies)
            return [nl], self.node_energies[0,nl]
        elif self.struct == 'free_edge':
            # Get all labellings of the two nodes involved. 
            _labels = _generate_label_permutations(self.n_labels)
            # First set of labels. 
            lx, ly = _labels[0]
            # Min energy so far ...
            min_energy = self.node_energies[0,lx] + self.node_energies[1,ly] + self.edge_energies[0,lx,ly]
            min_labels = [lx, ly]
            # Find overall minimum energy. 
            for [lx, ly] in _labels[1:]:
                _energy = self.node_energies[0,lx] + self.node_energies[1,ly] + self.edge_energies[0,lx,ly]
                if _energy < min_energy:
                    min_energy = _energy
                    min_labels = [lx, ly]
            # Return the minimum energy. 
            return min_labels, min_energy
                
        else:
            print 'Slave struct not recognised: %s.' %(self.struct)
            raise ValueError

    def _compute_energy(self):
        '''
        Slave._compute_energy(): Computes the energy of this slave. Assumes that
                                 we have already assigned labels to this slave. 
        '''
        if self.struct == 'cell':
            self._energy    = _compute_4node_slave_energy(self.node_energies, self.edge_energies, self.labels)
        elif self.struct == 'tree':
            self._energy    = _compute_tree_slave_energy(self.node_energies, self.edge_energies, self.labels, self.graph_struct)
        elif self.struct == 'cycle':
            self._energy    = _compute_cycle_slave_energy(self.node_energies, self.edge_energies, self.labels, self.node_list)
        elif self.struct == 'free_node':
            self._energy    = self.node_energies[0, self.labels[0]]
        elif self.struct == 'free_edge':
            self._energy    = self.node_energies[0,self.labels[0]] + \
                              self.node_energies[1,self.labels[1]] + \
                              self.edge_energies[0, self.labels[0], self.labels[1]]
        else:
            print 'Slave struct not recognised: %s.' %(self.struct)
            raise ValueError
        return self._energy

# ---------------------------------------------------------------------------------------


class Graph:
    '''
    A class which serves as an API to create a graph. 
    A Graph object can be created by specifying the number of nodes and the number of labels. 
    Node energies and edge energies can be added later. Only after the addition of 
    node and edge energies, can the user proceed to its optimisation. The optimisation
    is done by first breaking the graph into slaves (sub-graphs), and then iteratively solving
    them according to the DD-MRF algorithm. 
    '''

    def __init__(self, n_nodes=None, n_labels=None, n_edges=None, init_from_uai=None, _maximise=False):
        '''
        Graph.__init__(): Initialise the graph to be of shape (rows, cols), with 
                            a node taking a maximum of n_labels. 
        '''
        # If init_from_uai is set, call self._init_from_uai
        if init_from_uai is not None:
            self._init_from_uai(init_from_uai, _maximise)
            return

        # The rows, columns, and node indexing. 
        self.n_nodes    = n_nodes

        # Adjacency matrix. 
        self.adj_mat    = np.zeros((self.n_nodes, self.n_nodes), dtype=np.bool)

        # An array to store the final labels. 
        self.labels     = np.zeros(self.n_nodes)

        # If n_edges is None, we set the number of edges to zero, else we set it to 
        #   the maximum possible number, which is n_nodes*(n_nodes - 1)/2.
        if n_edges is None:
            self.n_edges = self.n_nodes*(self.n_nodes - 1)/2
        else:
            self.n_edges = n_edges

        # self._current_edge_count is a variable that keeps track of the index where
        #   every added edge is being inserted. 
        self._current_edge_count = 0

        # Create maps: V x V -> E and E -> V x V. 
        # These give us the edge ID given two end-points, and vice-versa, respectively. 
        self._edge_id_from_node_ids = np.zeros((self.n_nodes, self.n_nodes), dtype=n_dtype)
        self._node_ids_from_edge_id = np.zeros((self.n_edges, 2), dtype=n_dtype)

        # To set the maximum number of labels, we consider what kind of input is n_labels. 
        # If n_labels is an integer, we assume that all nodes should get the same max_n_labels.
        # Another option is to specify a list of max_n_labels. 
        if type(n_labels) == np.int:
            self.n_labels   = n_labels + np.zeros(self.n_nodes).astype(l_dtype)
        elif np.array(n_labels).size == self.n_nodes:
            self.n_labels   = np.array(n_labels).astype(l_dtype)
        # In any case, the max n labels for this Graph is np.max(self.n_lables)
        self.max_n_labels   = np.max(self.n_labels)
        
        # Initialise the node and edge energies. 
        self.node_energies  = np.zeros((self.n_nodes, self.max_n_labels), dtype=e_dtype)
        # Initialise edge energies to allow for maximum number of possible edges. We 
        #   can later trim this matrix. 
        self.edge_energies  = np.zeros((self.n_edges, self.max_n_labels, self.max_n_labels), dtype=e_dtype)

        # Flags set to ensure that node energies have been set. If any energies
        #   have not been set, we cannot proceed to optimisation as the graph is not complete. 
        self.node_flags     = np.zeros(self.n_nodes, dtype=np.bool)

        # Flag to determine whether the original model demands a maximisation. 
        self._maximise      = False
        self._max_pot       = None

        # Which strategy to use to generate primal solutions. 
        self._primal_strat  = 'vote'            # Use slaves to vote on nodes. 


    def _init_from_uai(self, uai_file, _maximise):
        # Read a .uai file and set parameters accordingly.
        fuai = open(uai_file, 'r')

        # Read the data into a variable
        fdata = fuai.readlines()
        # Close fuai. We don't need it anymore.
        fuai.close()

        # Remove whitespaces from each line.
        fdata = [t.strip() for t in fdata]

        # Remove empty lines. 
        fdata = [t for t in fdata if t !=  '']

        # The number of lines in the trimmed file. 
        n_lines = len(fdata)

        # --- This is the start of the preamble ---
        # The first line is type of network, which can be safely ignored. 

        # The second line is the number of nodes in the graph.
        n_nodes  = int(fdata[1])

        # The next line is specifies cardinalities of all variables, separated
        #   by spaces. 
        n_labels = [int(t) for t in fdata[2].split(' ')]
        
        # The next line contains one integer: the number of factors. This 
        #   number must be >= n_nodes, as all the nodes must have unaries. 
        n_factors = int(fdata[3])
        # DEBUG: We need to make sure that there are at least n_nodes factors.
        try:
            assert(n_factors >= n_nodes)
        except AssertionError:
            print 'The number of factors (%d) must be greater than or equal to',
            print 'the number of nodes (%d).' %(n_factors, n_nodes)
            return

        # We are currently at line ...
        c_line   = 4

        # Read the next n_factors lines, and store the order of factors. 
        factor_order = [[int(x) for x in l.split(' ')] for l in fdata[c_line:c_line+n_factors]]

        # We are currently at line ...
        c_line   = c_line + n_factors

        # Calculate the number of nodes and edges. 
        n_nodes  = reduce(lambda x, y: x + 1 if y[0] == 1 else x, factor_order, 0)
        n_edges  = n_factors - n_nodes

        # Call __init__ with n_nodes and n_labels.
        self.__init__(n_nodes=n_nodes, n_labels=n_labels, n_edges=n_edges, init_from_uai=None)

        # --- This is the end of the preamble ---
        # --- Next we move to function tables ---

        # Each factor is represented by giving the full table. 
        # We iterate over factor_order and depending on whether the factor is a node or an edge, 
        #    we record the potentials/energies. 
        for i in range(n_factors):
            this_factor = factor_order[i]
            
            # If it is a node ... 
            if this_factor[0] == 1:
                this_node = this_factor[1]
                # The first line is the number of labels for this factor. It must coincide with the 
                #   the number of labels for the factor it corresponds to in the model.
                n_labels_this_node = int(fdata[c_line])

                try:
                    assert(n_labels_this_node == n_labels[this_node])
                except:
                    print 'Conflicting inputs: number of labels in factor %d (%d) does not match number of labels' %(i, n_labels_this_node),
                    print 'in the preamble (%d).' %(n_labels[this_node])
                    return

                # Update c_line
                c_line = c_line + 1
    
                # Read this node's energies next. 
                n_elements    = 0
                node_energies = []
                while n_elements != n_labels[this_node]:
                    _next         =  [np.float64(x) for x in fdata[c_line].split(' ')]
                    n_elements    += len(_next)
                    node_energies += _next
                    c_line        =  c_line + 1
                # Set the node energies.
                self.set_node_energies(this_node, node_energies)

            # Else if it is an edge ...
            elif this_factor[0] == 2:
                # The edge ends for this factor. 
                eend0, eend1 = this_factor[1:]
                # The first line is the number of labels for this factor. It must coincide with the 
                #   the number of labels for the factor it corresponds to in the model.
                n_labels_this_edge = int(fdata[c_line])
                try:
                    assert(n_labels_this_edge == n_labels[eend0]*n_labels[eend1])
                except AssertionError:
                    print 'Conflicting inputs: number of labels in factor %d (%d) does not match number of labels' %(e+n_nodes, n_labels_this_edge),
                    print 'in the preamble (%d).' %(n_labels[eend0]*n_labels[eend1])
                    return

                # Update c_line
                c_line = c_line + 1
    
                # Read this edge's energies. It is the next n_labels[eend0]*n_labels[eend1] elements, as linebreaks
                #   don't count for anything. 
                n_elements    = 0
                edge_energies = []
                while n_elements != n_labels[eend0]*n_labels[eend1]:
                    _next         =  [np.float64(x) for x in fdata[c_line].split(' ')]
                    n_elements    += len(_next)
                    edge_energies += _next
                    c_line        =  c_line + 1
                # Reshape the array
                edge_energies = np.array(edge_energies, dtype=e_dtype).reshape(n_labels[eend0], n_labels[eend1])
    
    #           edge_energies = [[np.float32(x) for x in t.split(' ')] for t in fdata[c_line+1:c_line+1+n_labels[eend0]]]
                # Set the edge energies.
                self.set_edge_energies(eend0, eend1, edge_energies)

        # Make sure we have read the entire file. 
        try:
            assert(n_lines == c_line)
        except AssertionError: 
            print 'Some factors were left out. Reading did not reach end of line (%d/%d).' %(c_line/n_lines)
            return

        if _maximise:
            # If _maximise is set, it means the original problem asks for argmax E.
            # However, this code solves an argmin problem. Hence, we convert the potentials
            #    supplied in this file to energies by subtracting them from the 
            #    maximum potential found in the model.
            
            # First, find the maximum potential in the model. 
            _max_node_pot = np.max(self.node_energies)
            _max_edge_pot = np.max(self.edge_energies)
            _max_pot      = np.max([_max_node_pot, _max_edge_pot])
            # Next, normalize the potentials so that they lie in [0, 1]
            self.node_energies = self.node_energies/_max_pot
            self.edge_energies = self.edge_energies/_max_pot
            # Now, subtract these from 1. 
            self.node_energies = 1.0 - self.node_energies
            self.edge_energies = 1.0 - self.edge_energies

            # This flag determines whether a potential is to be maximised. 
            self._maximise     = True
            self._max_pot      = _max_pot
        else:
            self._maximise     = False
            self._max_pot      = None

    def save_uai(self, f_name):
        '''
        Graph.save_uai(): Save the current graph to a uai file, specified by f_name.
        '''
    
        fp = open(f_name, 'w')

        # PREAMBLE
        fp.write('MARKOV\n')
        fp.write('%d\n' %(self.n_nodes))
        n_labels_list = [str(t) for t in self.n_labels]
        fp.write(' '.join(n_labels_list) + '\n')
        fp.write('%d\n' %(self.n_nodes + self.n_edges))

        # List of cliques: nodes and edges. 
        for i in range(self.n_nodes):
            fp.write('1 %d\n' %(i))

        for i in range(self.n_edges):
            e_ends = self._node_ids_from_edge_id[i,:]
            fp.write('2 %d %d\n' %(e_ends[0], e_ends[1]))

        fp.write('\n')

        # Record nodes. 
        for i in range(self.n_nodes):
            fp.write('%d\n' %(self.n_labels[i]))
            e_array = [str(t) for t in self.node_energies[i,:].tolist()] 
            fp.write(' '.join(e_array) + '\n\n')

        # Record edges. 
        for i in range(self.n_edges):
            e_ends = self._node_ids_from_edge_id[i,:]
            fp.write('%d\n' %(self.n_labels[e_ends[0]]*self.n_labels[e_ends[1]]))
            e_array = [str(t) for t in self.edge_energies[i,:,:].flatten().tolist()]
            fp.write(' '.join(e_array) + '\n\n')

        fp.close()


    def set_node_energies(self, i, energies):
        '''
        Graph.set_node_energies(): Set the node energies for node i. 
        '''
        # Check if a valid index. 
        if i < 0 or i >= self.n_nodes:
            print 'Invalid index: %d.' %(i)
            raise IndexError

        # Convert the energy to a numpy array
        energies = np.array(energies, dtype=e_dtype)

        if energies.size != self.n_labels[i]:
            print 'Graph.set_node_energies(): The supplied node energies do not agree',
            print '(%d) on the number of labels required (%d).' %(energies.size, self.n_labels[i])
            print 'Node: %d; n_labels: %d' %(i, self.n_labels[i])
            print 'Supplied energies: ', energies
            raise ValueError

        # Make the assignment: set the node energies. 
        self.node_energies[i, 0:self.n_labels[i]] = energies
        # Set flag for this node to True.
        self.node_flags[i]      = True

    def set_edge_energies(self, i, j, energies):
        '''
        Graph.set_edge_energies(): Sets the edge energies for edge (i,j). The
        function first checks for the possibility of an edge between i and j, 
        and makes the assignment only if such an edge is possible.
        '''
        # Check that indices are not out of range. 
        if i >= self.n_nodes or j >= self.n_nodes or i < 0 or j < 0:
            print 'Graph.set_edge_energies(): At least one of the supplied edge indices is invalid (not in [0, n_nodes)).'
            raise IndexError

        # Convert the energy to a numpy array
        energies = np.array(energies, dtype=e_dtype)

        # Convention: one can only have edges from a lower index to a higher index. 
        # If specified energies do not conform, swap i and j, and transpose energies.
        if j < i:
            energies = energies.T
            i, j     = j, i
#       if j < i: 
#           print 'Graph.set_edge_energies(): Please specify indices from a lower index to higher index. %d > %d here.' %(i, j)
#           raise ValueError

        # Convert indices to int, just in case ...
        i = n_dtype(i)
        j = n_dtype(j)

        # Check that the supplied energy has the correct shape. 
        input_shape     = list(energies.shape)
        reqd_shape      = [self.n_labels[i], self.n_labels[j]]
        if input_shape != reqd_shape:
            print 'Graph.set_edge_energies(): The supplied energies have invalid shape:',
            print '(%d, %d). It must be (%d, %d).' \
                         %(energies.shape[0], energies.shape[1], self.n_labels[i], self.n_labels[j])
            raise ValueError

        # edge_id is the self._current_edge_count
        edge_id = self._current_edge_count

        # Make assignment: set the edge energies. 
        self.edge_energies[edge_id, 0:self.n_labels[i], 0:self.n_labels[j]]  = energies
        # Mark this edge in the adjacency matrix. 
        self.adj_mat[i,j] = self.adj_mat[j,i] = True

        # Update maps: V x V -> E and E -> V x V. 
        self._edge_id_from_node_ids[i,j]       = edge_id
        self._edge_id_from_node_ids[j,i]       = edge_id
        self._node_ids_from_edge_id[edge_id,:] = [i,j]

        # Increment the number of edges. 
        self._current_edge_count += 1


    def check_completeness(self):
        '''
        Graph.check_completeness(): Check whether all nodes have been assigned energies.
        '''
        # Check whether all nodes have been set. 
        if np.sum(self.node_flags) < self.n_nodes:
            return False
        # Everything is okay. 
        return True

    
    def create_slaves(self, decomposition='mixed', max_depth=5, slave_list=None):
        '''
        Graph.create_slaves(): Create slaves for this particular graph.
        The default decomposition is 'mixed'. Allowed values for decomposition are in
        ['mixed', 'tree', 'custom'].
        If decomposition is 'mixed', try to find as many small cycles as possible in the graph,
        and then decompose the rest of the graph into trees. 
        If decomposition is 'tree', create a set of trees
        instead by searching for trees in a greedy manner, starting at every node that
        still has edges which are not yet in any tree. 
        If decomposition is 'custom', the user can specify a custom decomposition. 
        A decomposition is entirely defined by the list of slaves. An option shall 
        be included later in which a decomposition can be specified using an adjacency
        matrix (which shall be easier for the user).
        '''
        # TODO: Add functionality to allow the user to set a decomposition by specifying
        #    the adjacency matrix. 

        # self._max_nodes_in_slave, and self._max_edges_in_slave are used
        #   to simplify node and edge updates. They shall be computed by the
        #   _create_*_slaves() functions, whichever is called. 
        self._max_nodes_in_slave = 0 
        self._max_edges_in_slave = 0

        # Create a closure to handle the required input of max_depth to self._create_tree_slaves(). 
        def _make_create_tree_slaves(md,sl):
            def h():
                return self._create_tree_slaves(max_depth=md, slave_list=sl)
            return h
        # Create a closure to handle the required input of max_depth to self._create_mixed_slaves().
        def _make_create_mixed_slaves(md,sl):
            def h():
                return self._create_mixed_slaves(max_length=md, slave_list=sl)
            return h
        # Create a closure to handle the required input of slave_list to self._create_custom_slaves().
        def _make_create_custom_slaves(sl):
            def h():
                return self._create_custom_slaves(sl)
            return h

        # Functions to call depending on which slave is chosen
        _slave_funcs = {
            'factor':  self._create_factor_slaves,
            'tree':    _make_create_tree_slaves(max_depth, slave_list),
            'mixed':   _make_create_mixed_slaves(max_depth, slave_list),
            'custom':  _make_create_custom_slaves(slave_list)
        }
        
        if decomposition not in _slave_funcs.keys():
            print 'decomposition must be one of', _slave_funcs.keys()
            raise ValueError

        # Create slaves depending on what decomposition is requested.
        _slave_funcs[decomposition]()

        # Two variables to hold how many slaves each node and edge is contained in (instead
        #   of computing the size of the corresponding vector each time. 
        self._n_slaves_nodes = np.array([self.nodes_in_slaves[n].size for n in range(self.n_nodes)], dtype=n_dtype)
        self._n_slaves_edges = np.array([self.edges_in_slaves[e].size for e in range(self.n_edges)], dtype=n_dtype)
        # Initially, we need only check those nodes and edges which associate with at least two slaves. 
        self._check_nodes    = np.where(self._n_slaves_nodes > 1)[0]
        self._check_edges    = np.where(self._n_slaves_edges > 1)[0]

        if decomposition != 'custom':
            # Finally, we must modify the energies for every edge or node depending on 
            #   how many slaves it is a part of. The energy for a node/edge is distributed
            #   equally among all slaves. 
            for n_id in np.where(self._n_slaves_nodes > 1)[0]:
                # Retrieve all the slaves this node is part of.
                s_ids   = self.nodes_in_slaves[n_id]
                # Distribute this node's energy equally between all slaves.
                for s in s_ids:
                    n_id_in_slave   = self.slave_list[s].node_map[n_id]
                    self.slave_list[s].node_energies[n_id_in_slave, :] /= 1.0*s_ids.size

            # Doing the same for edges ...
            for e_id in np.where(self._n_slaves_edges > 1)[0]:
                # Retrieve all slaves this edge is part of.
                s_ids   = self.edges_in_slaves[e_id]
                # Distribute this edge's energy equally between all slaves. 
                for s in s_ids:
                    e_id_in_slave   = self.slave_list[s].edge_map[e_id]
                    self.slave_list[s].edge_energies[e_id_in_slave, :] /= 1.0*s_ids.size

        # That is it. The slaves are ready. 

    def _create_factor_slaves(self):
        '''
        Graph._create_factor_slaves: Create a list of slaves in which each
        slave corresponds to a factor in the graph, i.e., either a node or
        an edge. 
        If the slave corresponds to an edge, its end-points (nodes) shall be shared
        with the corresponding 'node' slaves. 
        '''

        # A list to record in which slaves each vertex and edge occurs. 
        self.nodes_in_slaves = [[] for i in range(self.n_nodes)]
        self.edges_in_slaves = [[] for i in range(self.n_edges)]
        # The maximum number of slaves a node and an edge can appear in. 
        self._max_nodes_in_slave = 0
        self._max_edges_in_slave = 0

        # The number of slaves
        self.n_slaves   = self.n_nodes + self.n_edges

        # The slave list. 
        self.slave_list = np.array([Slave() for i in range(self.n_slaves)])

        # Create node slaves first. 
        for s_id in range(self.n_nodes):
            # The node this slave corresponds to. 
            n_id = s_id

            # The node list
            node_list = np.array([n_id])
            # The edge list is an empty array. 
            edge_list = np.array([])

            # Number of labels
            n_labels  = np.array([self.n_labels[n_id]])

            # Extract node energies. 
            node_energies    = np.zeros((1, n_labels[0]), dtype=e_dtype)
            node_energies[:] = self.node_energies[n_id,:n_labels[0]]

            # There are no edge energies. 
            edge_energies    = np.array([])

            # Assign parameters to this slave. 
            self.slave_list[s_id].set_params(node_list, edge_list, node_energies, n_labels, \
                    edge_energies, None, 'free_node')

            # Add this slave to the lists of nodes and edges in node_list and edge_list
            self.nodes_in_slaves[n_id] += [s_id]

        # Add edges now. 
        for s_id in range(self.n_nodes, self.n_slaves):
            # Get the edge ID. 
            e_id = s_id - self.n_nodes

            # The nodes corresponding to this edge. 
            i_id, j_id = self._node_ids_from_edge_id[e_id]

            # Node list
            node_list = np.array([i_id, j_id])
            # Edge list
            edge_list = np.array([e_id])

            # Number of labels
            n_labels       = np.array([self.n_labels[i_id], self.n_labels[j_id]])
            s_max_n_labels = np.max(n_labels)

            # Extract the node energies
            node_energies    = np.zeros((2, s_max_n_labels), dtype=e_dtype)
            node_energies[:] = self.node_energies[node_list, :s_max_n_labels]
            # Extract the edge energies
            edge_energies    = np.zeros((2, n_labels[0], n_labels[1]), dtype=e_dtype)
            edge_energies[:] = self.edge_energies[e_id, :n_labels[0], :n_labels[1]]

            # Assign these parameters to this slave. 
            self.slave_list[s_id].set_params(node_list, edge_list, node_energies, n_labels, \
                    edge_energies, None, 'free_edge')

            # Add this slave to the list of nodes and edges in node_list and edge_list
            self.nodes_in_slaves[i_id] += [s_id]
            self.nodes_in_slaves[j_id] += [s_id]
            self.edges_in_slaves[e_id] += [s_id]

        # For convenience, make elements of self.nodes_in_slaves, and self.edges_in_slaves into
        # Numpy arrays. 
        self.nodes_in_slaves = [np.array(t) for t in self.nodes_in_slaves]
        self.edges_in_slaves = [np.array(t) for t in self.edges_in_slaves]

        # Set self._max_nodes_in_slaves and self._max_edges_in_slaves. We can hard-code these values. 
        self._max_nodes_in_slave = 2
        self._max_edges_in_slave = 1
        # That's it. 


    def _create_tree_slaves(self, max_depth=5, slave_list=None):
        '''
        Graph._create_tree_slaves: Create a list of tree-structured sub-problems. 
        Trees is detected in a greedy manner starting at the first node. 
        Any further trees are started at a node if there are any edges
        incident on that node that are not already in a tree. 
        '''
        
        # A list to record in which slaves each vertex and edge occurs. 
        self.nodes_in_slaves = [[] for i in range(self.n_nodes)]
        self.edges_in_slaves = [[] for i in range(self.n_edges)]
        # The maximum number of slaves a node and an edge can appear in. 
        self._max_nodes_in_slave = 0
        self._max_edges_in_slave = 0

        if slave_list is None:
            # Create adjacency matrices. 
            subtree_data = self._generate_trees_greedy(max_depth=max_depth)
        else:
            subtree_data = slave_list

        # Create free_node slaves: these contain nodes that do not have
        #    any edges incident on them. 
        free_nodes    = np.where(np.sum(self.adj_mat,axis=1) == 0)[0]

        # The number of slaves
        n_trees       = len(subtree_data)
        n_free_nodes  = free_nodes.size
        self.n_slaves = n_trees + n_free_nodes
        # The list of slaves.
        self.slave_list = np.array([Slave() for i in range(self.n_slaves)])

        # Create each slave now. 
        for s_id in range(n_trees):
            # Extract the adjacency matrices for this slave. 
            tree_adj  = subtree_data[s_id][0]
            node_list = np.array(subtree_data[s_id][1], dtype=n_dtype)
            edge_list = np.array(subtree_data[s_id][2], dtype=n_dtype)

            # Number of nodes and edges in this tree. 
            n_nodes = node_list.size
            n_edges = edge_list.size

            # The number of labels for each node here. 
            n_labels         = np.zeros(n_nodes, dtype=l_dtype)
            n_labels[:]      = self.n_labels[node_list]

            # The maximum cardinality of a node in this slave. 
            s_max_n_labels   = np.max(n_labels)

            # Update self._max_nodes_in_slave, and self._max_edges_in_slave
            if self._max_nodes_in_slave < n_nodes:
                self._max_nodes_in_slave = n_nodes
            if self._max_edges_in_slave < n_edges:
                self._max_edges_in_slave = n_edges

            # Extract node energies. 
            node_energies    = np.zeros((n_nodes, s_max_n_labels), dtype=e_dtype)
            node_energies[:] = self.node_energies[node_list, 0:s_max_n_labels]

            # Extract edge energies.
            edge_energies    = np.zeros((n_edges, s_max_n_labels, s_max_n_labels), dtype=e_dtype)
            edge_energies[:] = self.edge_energies[edge_list, 0:s_max_n_labels, 0:s_max_n_labels]

            # Create graph structure. 
            gs = bp.make_graph_struct(tree_adj, n_labels)

            # Verify that everything is consistent. 
            for e in range(gs['n_edges']):
                e0, e1 = gs['edge_ends'][e,:]
                try:
                    assert(self._edge_id_from_node_ids[node_list[e0], node_list[e1]] == edge_list[e])
                except AssertionError:
                    print 'Conflicting edge IDs in Graph._create_tree_slaves for slave %d.' %(s_id)
                    print 'Edge ID %d in Graph does not agree with ID %d in slave.' %(self._edge_id_from_node_ids[node_list[e0], node_list[e1]], edge_list[e])
                    print 'Node ID in slave are (%d, %d), and in Graph are (%d, %d)' %(e0, e1, node_list[e0], node_list[e1])
                    return

            # Set slave parameters. 
            self.slave_list[s_id].set_params(node_list, edge_list, node_energies, n_labels, \
                    edge_energies, gs, 'tree')

            # Add this slave to the lists of nodes and edges in node_list and edge_list
            for n_id in node_list:
                self.nodes_in_slaves[n_id] += [s_id]
            for e_id in edge_list:
                self.edges_in_slaves[e_id] += [s_id]

        # Make slaves for free nodes now. 
        for s_id in range(n_trees, self.n_slaves):
            # ID of this free node. 
            fn_id = s_id - n_trees

            # Node id corresponding to this free node
            n_id  = free_nodes[fn_id]

            # Create its node list
            node_list = np.array([n_id])
            # Edge list is empty
            edge_list = np.array([])

            # Create n_labels
            n_labels  = np.array([self.n_labels[n_id]])

            # Create its node energies
            node_energies    = np.zeros([1, n_labels[0]])
            node_energies[:] = self.node_energies[n_id, :n_labels[0]]

            # There are no edge energies. 
            edge_energies    = np.array([])

            # There is no graph struct
            graph_struct     = None

            # Set slave parameters. 
            self.slave_list[s_id].set_params(node_list, edge_list, node_energies, n_labels, \
                    edge_energies, graph_struct, 'free_node')

            # Add this slave to the lists of nodes and edges in node_list and edge_list
            for n_id in node_list:
                self.nodes_in_slaves[n_id] += [s_id]
            for e_id in edge_list:
                self.edges_in_slaves[e_id] += [s_id]

        # For convenience, make elements of self.nodes_in_slaves, and self.edges_in_slaves into
        # Numpy arrays. 
        self.nodes_in_slaves = [np.array(t) for t in self.nodes_in_slaves]
        self.edges_in_slaves = [np.array(t) for t in self.edges_in_slaves]
        # C'est ca.


    def _create_mixed_slaves(self, max_length=4, slave_list=None):
        '''
        Create mixed slaves. 
        This algorithm tries to find a small cycle for every node, i.e., it 
        iterates over the list of nodes, and for each node, tries to locate a cycle
        that that node is a part of. If no such cycle is found, the algorithm moves on.
        After a list of cycles has been obtained, all those edges are removed from 
        the adjacency matrix, and the remaining adjacency matrix is broken into trees. 
        NOTE: This algorithm does NOT locate all cycles in a graph, merely that
        it locates AT LEAST ONE (if one exists) for every node.
        The set of cycles and the resulting set of trees gives a decomposition
        of the graph. 
        '''
        
        # A list to record in which slaves each vertex and edge occurs. 
        self.nodes_in_slaves = [[] for i in range(self.n_nodes)]
        self.edges_in_slaves = [[] for i in range(self.n_edges)]
        # The maximum number of slaves a node and an edge can appear in. 
        self._max_nodes_in_slave = 0
        self._max_edges_in_slave = 0

        # We work with the adjacency matrix of this graph. 
        # Create a copy of the adjacency matrix so that the original is not affected. 
        adj_mat    = np.zeros_like(self.adj_mat)
        adj_mat[:] = self.adj_mat[:]

        # Adjust max_length. It is set to 6 by default, because self.n_nodes + 1 is too high, 
        #    and computationally very expensive. 
        if max_length == -1:
            max_length = self.n_nodes

#       # The list of cycles. 
#       cycles = []
#
#       # Iterate over every node to find a cycle.
#       for n in range(self.n_nodes):
#           # If the degree of this vertex is 1, there cannot be any cycles here. 
#           if np.sum(adj_mat[n,:]) == 1:
#               continue
#
#           # Locate a cycle starting at this node. 
#           c_n = find_cycle_in_graph(adj_mat, n)
#
#           # If no cycle is found, move on
#           if c_n is None:
#               continue
#
#           # The cycle outputs both the starting node twice. Remove it. 
#           c_n = c_n[:-1]
#
#           # Add to our list of cycles. 
#           cycles += [c_n]
#           # Remove edges in the cycle containing node n. 
#           e0, e1          = c_n[0], c_n[1]
#           adj_mat[e0, e1] = adj_mat[e1, e0] = False
#           e0, e1          = c_n[-1], c_n[0]
#           adj_mat[e0, e1] = adj_mat[e1, e0] = False
##          for t in range(len(c_n)):
##              e0, e1          = c_n[t], c_n[(t+1)%len(c_n)]
##              adj_mat[e0, e1] = False
##              adj_mat[e1, e0] = False

        # The list of cycles. 
#   -   f  = plt.figure(1)
#   -   plt.subplot(121)
#   -   plt.imshow(1-adj_mat, cmap='Greys', interpolation='nearest')

        cycles = []

        # If slave_list is not None, use the cycles specified in slave_list. 
        if slave_list is not None:
            cycles = slave_list
            # Remove edges from the graph that are already in these cycles. 
            for c_n in cycles:
                for _n in range(len(c_n)):
                    i0, i1     = _n, (_n+1)%len(c_n)
                    adj_mat[c_n[i0], c_n[i1]] = False
                    adj_mat[c_n[i1], c_n[i0]] = False

        else:
            # Edit: Keep finding cycles till there are no more.         # Appended: 2017-08-24. 
            while True:
                cycles_i = dfs_unique_cycles(adj_mat, max_length=max_length)
                if len(cycles_i) == 0:
                    break
                else:
                    cycles += cycles_i
                    # Remove edges from the graph that are already in these cycles. 
                    for c_n in cycles_i:
                        for _n in range(len(c_n)):
                            i0, i1     = _n, (_n+1)%len(c_n)
                            adj_mat[c_n[i0], c_n[i1]] = False
                            adj_mat[c_n[i1], c_n[i0]] = False

#       cycles = dfs_unique_cycles(adj_mat, max_length=max_length)
#   -   plt.subplot(122)
#   -   plt.imshow(1-adj_mat, cmap='Greys', interpolation='nearest')
#   -   plt.show()

        # We would like each tree to have at most a quarter of the
        #    total nodes in the graph. The depth is hence set as 
        #    1 + log of self.n_nodes/4 to a base equal to the average node
        #    node degree of the graph, rounded to the nearest integer. 
        avg_degree   = np.mean(self.adj_mat)*self.n_nodes
        _max_depth_t = np.log(self.n_nodes/4.0)/avg_degree

        # Now find trees in the remaining adjacency matrix. 
        subtree_data = self._generate_trees_greedy(adjacency=adj_mat, max_depth=-1)     # Just using 2 for now. 

        # Finally, add any nodes that do not have any edges connected to them, 
        #    and place them in slaves of their own. 
        free_nodes    = np.where(np.sum(self.adj_mat,axis=1) == 0)[0]

        # The number of slaves is the length of cycles plus the length of subtree_data
        n_cycles      = len(cycles)
        n_trees       = len(subtree_data)
        n_free_nodes  = free_nodes.size
        self.n_slaves = n_cycles + n_trees + n_free_nodes

        # Create slave list. 
        self.slave_list = np.array([Slave() for s in range(self.n_slaves)])

        # Make cycle slaves first. 
        for s_id in range(n_cycles):
            # The current cycle. 
            c_n = cycles[s_id]

            # Make node and edge lists. 
            node_list = np.array(c_n, dtype=n_dtype)
            edge_list = np.zeros_like(node_list)

            # Number of nodes and edges. 
            n_nodes   = node_list.size
            n_edges   = edge_list.size

            for i in range(node_list.size):
                e0, e1       = node_list[i], node_list[(i+1)%n_nodes]
                e_id         = self._edge_id_from_node_ids[e0, e1]
                edge_list[i] = e_id

            # The number of labels for nodes in this slave. 
            n_labels       = np.zeros(n_nodes, dtype=l_dtype)
            n_labels[:]    = self.n_labels[node_list]

            # Max label for this slave. 
            s_max_n_labels = np.max(n_labels)

            # Update self._max_nodes_in_slave, and self._max_edges_in_slave
            if self._max_nodes_in_slave < n_nodes:
                self._max_nodes_in_slave = n_nodes
            if self._max_edges_in_slave < n_edges:
                self._max_edges_in_slave = n_edges

            # Node energies
            node_energies    = np.zeros((n_nodes, s_max_n_labels), dtype=e_dtype)
            for _n_id in range(n_nodes):
                node_energies[_n_id, 0:n_labels[_n_id]] = self.node_energies[node_list[_n_id], 0:n_labels[_n_id]]
#           node_energies[:] = self.node_energies[node_list, 0:s_max_n_labels]

            # Edge energies
            edge_energies    = np.zeros((n_edges, s_max_n_labels, s_max_n_labels), dtype=e_dtype)
            edge_energies[:] = self.edge_energies[edge_list, 0:s_max_n_labels, 0:s_max_n_labels]

            for e in range(n_edges):
                i0, i1 = e, (e+1)%n_nodes
                e0, e1 = node_list[i0], node_list[i1]
                try:
                    assert(np.array_equal(edge_energies[e, :n_labels[i0], :n_labels[i1]], self.edge_energies[edge_list[e], :self.n_labels[e0], :self.n_labels[e1]]))
                except AssertionError:
                    print 'In slave %d, ' %(s_id)
                    print node_list, edge_list
                    print 'Edge %d, %d -> %d, is not consistent with transposing energies.' %(e, i0, i1)
                    print 'edge_energies in slave:', edge_energies[e, :n_labels[i0], :n_labels[i1]].shape
                    print edge_energies[e, :n_labels[i0], :n_labels[i1]]
                    print 'edge_energies in Graph:', self.edge_energies[edge_list[e], :self.n_labels[e0], :self.n_labels[e1]].shape
                    print self.edge_energies[edge_list[e], :self.n_labels[e0], :self.n_labels[e1]]
                    return

            # Set slave parameters. 
            self.slave_list[s_id].set_params(node_list, edge_list, node_energies, n_labels, edge_energies, None, 'cycle')

            # Add this slave to the lists of nodes and edges in node_list and edge_list
            for n_id in node_list:
                self.nodes_in_slaves[n_id] += [s_id]
            for e_id in edge_list:
                self.edges_in_slaves[e_id] += [s_id]

        # Now make tree slaves. 
        for s_id in range(n_cycles, n_cycles + n_trees):
            # Tree ID is n_cycles less than s_id. 
            t_id      = s_id - n_cycles

            # Extract the adjacency matrices for this slave. 
            tree_adj  = subtree_data[t_id][0]
            node_list = np.array(subtree_data[t_id][1], dtype=n_dtype)
            edge_list = np.array(subtree_data[t_id][2], dtype=n_dtype)

            # Number of nodes and edges in this tree. 
            n_nodes = node_list.size
            n_edges = edge_list.size

            # The number of labels for each node here. 
            n_labels         = np.zeros(n_nodes, dtype=l_dtype)
            n_labels[:]      = self.n_labels[node_list]

            # Max label for this slave. 
            s_max_n_labels = np.max(n_labels)

            # Update self._max_nodes_in_slave, and self._max_edges_in_slave
            if self._max_nodes_in_slave < n_nodes:
                self._max_nodes_in_slave = n_nodes
            if self._max_edges_in_slave < n_edges:
                self._max_edges_in_slave = n_edges

            # Extract node energies. 
            node_energies    = np.zeros((n_nodes, s_max_n_labels), dtype=e_dtype)
            for _n_id in range(n_nodes):
                node_energies[_n_id, 0:n_labels[_n_id]] = self.node_energies[node_list[_n_id], 0:n_labels[_n_id]]
#           node_energies[:] = self.node_energies[node_list, 0:s_max_n_labels]

            # Create graph structure. 
            gs = bp.make_graph_struct(tree_adj, n_labels)

            # Extract edge energies.
            edge_energies    = np.zeros((n_edges, s_max_n_labels, s_max_n_labels), dtype=e_dtype)
            edge_energies[:] = self.edge_energies[edge_list, 0:s_max_n_labels, 0:s_max_n_labels]
            # Here, we must adjust self.edge_energies before transferring them to a slave. 
            # This is because self.edge_energies always has energies for an edge from a lower node index
            #    to a higher node index, but this convention might not be followed in the adjacency
            #    matrix returned for the tree. 
#           for _e in range(n_edges):
#               # Get the edge ID in the Graph. 
#               e_id     = edge_list[_e]
#               # Edge end indices in node_list
#               i0, i1   = gs['edge_ends'][_e]
#               # Get edge ends from the Graph. 
#               e0, e1   = node_list[i0], node_list[i1]
#
#               assert(n_labels[i0] == self.n_labels[e0])
#               assert(n_labels[i1] == self.n_labels[e1])
#               
#               # Now if e0 < e1, we can use the edge energies matrix for this edge, 
#               #    but in the other case, we must transpose it. 
#               if e0 < e1:
#                   edge_energies[_e, 0:n_labels[i0], 0:n_labels[i1]] = self.edge_energies[e_id, 0:self.n_labels[e0], 0:self.n_labels[e1]]
#               else:
#                   edge_energies[_e, 0:n_labels[i0], 0:n_labels[i1]] = self.edge_energies[e_id, 0:self.n_labels[e1], 0:self.n_labels[e0]].T

            # Set slave parameters. 
            self.slave_list[s_id].set_params(node_list, edge_list, node_energies, n_labels, \
                    edge_energies, gs, 'tree')

            # Verify that everything is consistent. 
            for e in range(gs['n_edges']):
                e0, e1 = gs['edge_ends'][e,:]
                try:
                    assert(self._edge_id_from_node_ids[node_list[e0], node_list[e1]] == edge_list[e])
                except AssertionError:
                    print 'In slave %d, ' %(s_id)
                    print 'Conflicting edge IDs in Graph._create_tree_slaves.'
                    print 'Edge ID %d in Graph does not agree with ID %d in slave.' %(self._edge_id_from_node_ids[node_list[e0], node_list[e1]], edge_list[e])
                    print 'Node ID in slave are (%d, %d), and in Graph are (%d, %d)' %(e0, e1, node_list[e0], node_list[e1])
                    return

            # Add this slave to the lists of nodes and edges in node_list and edge_list
            for n_id in node_list:
                self.nodes_in_slaves[n_id] += [s_id]
            for e_id in edge_list:
                self.edges_in_slaves[e_id] += [s_id]

        # Make slaves for free nodes now. 
        for s_id in range(n_cycles + n_trees, self.n_slaves):
            # ID of this free node. 
            fn_id = s_id - n_cycles - n_trees

            # Node id corresponding to this free node
            n_id  = free_nodes[fn_id]

            # Create its node list
            node_list = np.array([n_id])
            # Edge list is empty
            edge_list = np.array([])

            # Create n_labels
            n_labels  = np.array([self.n_labels[n_id]])

            # Create its node energies
            node_energies    = np.zeros([1, n_labels[0]])
            node_energies[:] = self.node_energies[n_id, :n_labels[0]]

            # There are no edge energies. 
            edge_energies    = np.array([])

            # There is no graph struct
            graph_struct     = None

            # Set slave parameters. 
            self.slave_list[s_id].set_params(node_list, edge_list, node_energies, n_labels, \
                    edge_energies, graph_struct, 'free_node')

            # Add this slave to the lists of nodes and edges in node_list and edge_list
            for n_id in node_list:
                self.nodes_in_slaves[n_id] += [s_id]
            for e_id in edge_list:
                self.edges_in_slaves[e_id] += [s_id]


        # For convenience, make elements of self.nodes_in_slaves, and self.edges_in_slaves into
        # Numpy arrays. 
        self.nodes_in_slaves = [np.array(t) for t in self.nodes_in_slaves]
        self.edges_in_slaves = [np.array(t) for t in self.edges_in_slaves]
        # C'est ca.


    def _create_custom_slaves(self, slave_list):
        '''
        Create a custom decomposition of the Graph. This function allows the user to 
        create a custom decomposition of the graph and apply this decomposition on 
        an instance of Graph. 

        Inputs
        ======
            slave_list: A series of instances of type Slave. Could 
                        be a list or a Numpy array. Each member of 'slave_list' 
                        must be of type Slave, and have all the required members
                        initialised. 
    
        '''

        # Convert to Numpy array. 
        slave_list = np.array(slave_list)

        # Assign to self.slave_list
        self.slave_list = slave_list

        # The number of slaves. 
        self.n_slaves = slave_list.size

        # Create empty lists for nodes_in_slaves and edges_in_slaves. 
        self.nodes_in_slaves = [[] for i in range(self.n_nodes)]
        self.edges_in_slaves = [[] for i in range(self.n_edges)]

        # Initialise _max_*_in_slave
        self._max_nodes_in_slave = 0
        self._max_edges_in_slave = 0

        for s_id in range(self.n_slaves):
            # Get node and edge lists. 
            node_list = slave_list[s_id].node_list 
            edge_list = slave_list[s_id].edge_list

            # Number of nodes and edges. 
            n_nodes   = node_list.size
            n_edges   = edge_list.size

            # Update self._max_nodes_in_slave, and self._max_edges_in_slave
            if self._max_nodes_in_slave < n_nodes:
                self._max_nodes_in_slave = n_nodes
            if self._max_edges_in_slave < n_edges:
                self._max_edges_in_slave = n_edges

            # Add them to nodes_in_slaves and edges_in_slaves. 
            for n_id in node_list:
                self.nodes_in_slaves[n_id] += [s_id]
            for e_id in edge_list:
                self.edges_in_slaves[e_id] += [s_id]

        # Convert lists in self.nodes_in_slaves and self.edges_in_slaves to 
        #    Numpy arrays for convenience. 
        self.nodes_in_slaves = [np.array(t) for t in self.nodes_in_slaves]
        self.edges_in_slaves = [np.array(t) for t in self.edges_in_slaves]



    def optimise(self, a_start=1.0, max_iter=1000, decomposition='tree', strategy='step', max_depth=2, \
            _momentum=0.0, slave_list=None, _verbose=True, _resume=False, _create_slaves=True):
        '''
        Graph.optimise(): Optimise the set energies over the graph and return a labelling. 

        Takes as input a_start, which is a float and denotes the starting value of \\alpha_t in
        the DD-MRF algorithm. 

        struct specifies the type of decomposition to use. struct must be in slave_types. 
        'cell' specifies a decomposition in which the graph in broken into 2x2 cells - each 
        being a slave. 'row_col' specifies a decomposition in which the graph is broken into
        rows and columns - each being a slave. 

        The strategy signifies what values of \\alpha to use at iteration t. Permissible 
        values are 'step' and 'adaptive'. The step strategy simply sets 

              \\alpha_t = a_start/sqrt(t).

        The adaptive strategy sets 
         
              \\alpha_t = a_start*\\frac{Approx_t - Dual_t}{norm(\\nabla g_t)**2},

        where \\nabla g_t is the subgradient of the dual at iteration t. 
        '''

        # If resume, resume previous optimisation. 
        if not _resume:
            # Check if a permissible decomposition is used. 
            if decomposition not in decomposition_types:
                print 'Permissible values for decomposition are \'tree\', and \'custom\' .'
                print 'Custom decomposition must be specified in the form of a list of slaves if \'custom\' is chosen.'
                raise ValueError
    
            # Check if a permissible strategy is being used. 
            if strategy not in ['step', 'step_ss', 'step_sg', 'adaptive', 'adaptive_d', 'adaptive_delta']:
                print 'Permissible values for strategy are \'step\', \'step_sg\', \'adaptive\', \'adaptive_d\', and \'adaptive_delta\''
                print '\'step\'           Use diminshing step-size rule: a_t = a_start/sqrt(it).'
                print '\'step_ss\'        Use a square summable but not summable sequence: a_t = a_start/(1.0 + t).'
                print '\'step_sg\'        Use subgradient in combination with diminishing step-size rule: a_t = a_start/(sqrt(it)*||dg||**2).'
                print '\'adaptive\'       Use adaptive rule given by the difference between the estimated PRIMAL cost and the current DUAL cost: a_t = a_start*(PRIMAL_t - DUAL_t)/||dg||**2.'
                print '\'adaptive_d\'     Use adaptive rule with diminishing step-size rule: a_t = a_start*(PRIMAL_t - DUAL_t)/(sqrt(it)*||dg||**2).'
                print '\'adaptive_delta\' Use adaptive rule which tries to get an improvement of delta at each iteration.'
                raise ValueError
            # If strategy is adaptive, we would like a_start to be in (0, 2).
            if strategy == 'adaptive' and (a_start <= 0 or a_start > 2):
                print 'Please use 0 < a_start < 2 for an adaptive strategy.'
                raise ValueError
    
            # Momentum must be in [0, 1)
            if _momentum < 0 or _momentum >= 1:
                print 'Momentum must be in [0, 1).'
                raise ValueError
    
            # First check if the graph is complete. 
            if not self.check_completeness():
                n_list = np.where(self.node_flags == False)
                print 'Graph.optimise(): The graph is not complete.'
                print 'The following nodes are not set:', n_list
                raise AssertionError
    
            # Trim edge energies and the E -> V x V map.
            self.edge_energies          = self.edge_energies[:self._current_edge_count,:,:]
            self._node_ids_from_edge_id = self._node_ids_from_edge_id[:self._current_edge_count,:]
            
            # Reset self.n_edges to _current_edge_count.
            self.n_edges = self._current_edge_count
    
            # Set the optimisation strategy. 
            self._optim_strategy = strategy
    
            # Create slaves. This creates a list of slaves and stores it in 
            #   self.slave_list. The numbering of the slaves starts from the top-left,
            #   and continues in row-major fashion. For example, there are 
            #   (self.rows-1)*(self.cols-1) slaves if the 'cell' decomposition is used. 
            if _create_slaves:
                self.decomposition = decomposition
                self.create_slaves(decomposition=self.decomposition, max_depth=max_depth, slave_list=slave_list)
                if _verbose:
                    print 'Checking decomposition ... ', 
                if self.check_decomposition():
                    if _verbose:
                        print 'OK!'
                else:
                    print 'Conflicts found (listed above). Please fix the decomposition before attempting to run Graph.optimise().'
                    return
    
            # Create update variables for slaves. Created once, reset to zero each time
            #   _compute_param_updates() and _apply_param_updates() are called. 
            self._slave_node_up = np.zeros((self.n_slaves, self._max_nodes_in_slave, self.max_n_labels), dtype=u_dtype)
            self._slave_edge_up = np.zeros((self.n_slaves, self._max_edges_in_slave, self.max_n_labels, self.max_n_labels), dtype=u_dtype)
            # Create a copy of these to hold the previous state update. Akin to momentum
            #   update used in NNs. 
            self._prv_node_sg   = np.zeros_like(self._slave_node_up)
            self._prv_edge_sg   = np.zeros_like(self._slave_edge_up)
            # Array to mark slaves for updates. 
            # The first row corresponds to node updates, while the second to edge updates. 
            self._mark_sl_up    = np.zeros((2,self.n_slaves), dtype=np.int8)
    
            # How much momentum to use. Must be in [0, 1)
            self._momentum = _momentum
        
            # Set all slaves to be solved at first. 
            self._slaves_to_solve   = np.arange(self.n_slaves)
    
            # Two lists to record the primal and dual cost progression
            self.dual_costs         = []
            self.primal_costs       = []
            self.subgradient_norms  = []

            # A list to record the number of disagreeing nodes at each iteration
            self._n_miss_history    = []
    
            # Best primal and dual costs
            self._best_primal_cost  = np.inf
            self._best_dual_cost    = -np.inf

            # Accumulators for computing intermediate primal and dual solutions. 
            self._wsg_accumulator   = np.zeros((self.n_nodes, self.max_n_labels))
            self._sg_accumulator    = np.zeros((self.n_nodes, self.max_n_labels))
            # Accumulator for alpha. 
            self._alpha_accumulator = 0.0
            # Stores the current subgradient, but not in terms of one-hot vectors. 
            self._sg_node           = np.zeros((self.n_nodes, self.max_n_labels), dtype=l_dtype)

            # The iteration in the optimisation process. This is stored as a member of the class so that
            #     continuing optimisation after it has stopped is easier. 
            self.it = 1

        # The iteration that measures how long to run this "batch" of optimisation. 
        it = 1

        # Whether converged or not. 
        converged = False
    
        _naive_search = False
    
        # Loop till not converged. 
        while not converged and it <= max_iter:
            if _verbose:
                print 'Iteration %5d. Solving %5d subproblems ...' %(self.it, self._slaves_to_solve.size),
            # Solve all the slaves. 
            # The following optimises the energy for each slave, and stores the 
            #    resulting labelling as a member in the slaves. 
            self._optimise_slaves()
            if _verbose:
                print 'done.',
            sys.stdout.flush()

            # Find the number of disagreeing points. 
            disagreements = self._find_conflicts()

            # Add to _n_miss_history
            self._n_miss_history += [disagreements.size]

            # Get the primal cost at this iteration
            primal_cost     = self._compute_primal_cost()
            if self._best_primal_cost > primal_cost:
                self._best_primal_cost     = primal_cost
                self._best_primal_solution = self.labels
            self.primal_costs += [primal_cost]

            # Get the dual cost at this iteration
            dual_cost       = self._compute_dual_cost()
            if self._best_dual_cost < dual_cost:
                self._best_dual_cost = dual_cost
            self.dual_costs += [dual_cost]

            # Compute parameter updates after this round of optimisation. But don't apply them yet!
            self._compute_param_updates(a_start)

            # Verify whether the algorithm has converged. If all slaves agree
            #    on the labelling of every node, we have convergence. 
            if disagreements.size == 0:
                print 'Converged after %d iterations!\n' %(self.it)
                print 'At convergence, PRIMAL = %.6f, DUAL = %.6f, Gap = %.6f.' %(primal_cost, dual_cost, primal_cost - dual_cost)
#               self._assign_labels()           # Use this if you want to infer a PRIMAL solution from the final state of the slaves instead.
                # Break from loop.
                converged = True
                break

            # Test: #TODO
            # If disagreements are less than or equal to 2, we do a brute force
            #    to search for the solution. 
            if _naive_search and primal_cost-dual_cost <= self._erg_step:
                print 'Forcing naive search as _naive_search is True, and difference in primal and dual costs <= _erg_step.'
                self.force_naive_search(disagreements, response='y')
                break

            # Apply updates to parameters of each slave. 
            self._apply_param_updates()

            # Print statistics. .
            if _verbose:
                print ' alpha = %10.6f. n_miss = %6d.' %(self.alpha, disagreements.size),
                print '||dg||**2 = %4.2f, PRIMAL = %6.6f. DUAL = %6.6f, P - D = %6.6f, min(P - D) = %6.6f '\
                %(self.subgradient_norms[-1], primal_cost, dual_cost, primal_cost-dual_cost, self._best_primal_cost - self._best_dual_cost)

            # Increase iteration.
            self.it += 1
            it      += 1

        print 'The best labelling is stored as a member \'labels\' in the object.'
        print 'Best PRIMAL = %.6f, Best DUAL = %.6f, Gap = %.6f' %(self._best_primal_cost, self._best_dual_cost, self._best_primal_cost - self._best_dual_cost)
        
    def _optimise_slaves(self):
        '''
        A function to optimise all slaves. This function distributes the job of
        optimising every slave to all but one cores on the machine. 
        '''
        # Extract the list of slaves to be optimised. This contains of all the slaves that
        #   disagree with at least one other slave on the labelling of at least one node. 
        _to_solve   = [self.slave_list[i] for i in self._slaves_to_solve]
        # The number of cores to use is the number of cores on the machine minus 1. 
        # Use only as many cores as needed. 
        n_cores     = np.min([cpu_count() - 1, self._slaves_to_solve.size])

        # Optimise the slaves. 
# =================== Using Joblib =====================
#        optima      = Parallel(n_jobs=n_cores)(delayed(_optimise_slave)(s) for s in _to_solve)
# ======================================================

# =============== Using multiprocessing ================
#        # Refer to the global shared_slave_list
#        global shared_slave_list
#        # Populate the shared list with slaves to solve. 
#        for i in self._slaves_to_solve:
#            shared_slave_list.append(self.slave_list[i])
#        
##        optima = []
##        for s in shared_slave_list:
##            optima.append(_optimise_slave(s))
#       
#        # Create a pool of workers. 
#        multp_pool  = multiprocessing.Pool(processes=8)
#        # Distribute the evaluation of _optimise_slave_mp over n_cores cores. 
#        optima      = multp_pool.map(_optimise_slave_mp, range(self._slaves_to_solve.size))
#        # Finish the closure. 
#        multp_pool.terminate()
#        # Reset shared_slave_list to zero. We still want shared_slave_list to be 
#        #    the of the type that it is. 
#        del shared_slave_list[:]
# ======================================================

# ===================== Serially =======================
        optima = []
        for s in _to_solve:
            optima.append(_optimise_slave(s))
# ======================================================

        # Reflect the result in slave list for our Graph. 
        for i in range(self._slaves_to_solve.size):
            s_id = self._slaves_to_solve[i]
            self.slave_list[s_id].set_labels(optima[i][0])
            self.slave_list[s_id]._energy = optima[i][1]
            if self.slave_list[s_id].struct == 'tree':
                self.slave_list[s_id]._messages    = optima[i][2]
                self.slave_list[s_id]._messages_in = optima[i][3]

    # End of Graph._optimise_slaves()


    def check_decomposition(self):
        '''
        Check the correctness of a decomposition. 
        To optimise, the dual and the primal must be in agreement. 
        '''
        # Iterate over nodes to check that node energies are split
        #    correctly among slaves. 

        flag = True
        numel = self.n_nodes + self.n_edges

        def _array_equal(a, b, epsilon=1e-6):
            '''
            Check whether two arrays are equal up to a precision.
            '''
            for i, t in np.ndenumerate(a):
                if np.abs(a[i] - b[i]) > epsilon:
                    return False
            return True

        for n_id in range(self.n_nodes):
            if n_id%(numel/10) == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            _n_energy = 0.0
            for s_id in self.nodes_in_slaves[n_id]:
                n_id_in_s = self.slave_list[s_id].node_map[n_id]
                n_lbl     = self.n_labels[n_id]

                _n_energy += self.slave_list[s_id].node_energies[n_id_in_s, :n_lbl]
            if not _array_equal(_n_energy, self.node_energies[n_id, :n_lbl]):
                print '\nGraph.check_decomposition: Dual decomposition disagreement for node %d.' %(n_id)
                print 'Graph.check_decomposition: Node energies in PRIMAL are '
                print self.node_energies[n_id, :n_lbl]
                print 'Graph.check_decomposition: Sum of node energies in the decomposition is '  
                print _n_energy.tolist()
                flag = False

        for e_id in range(self.n_edges):
            if (self.n_nodes + e_id)%(numel/10) == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            x, y             = self._node_ids_from_edge_id[e_id,:]
            n_lbl_x, n_lbl_y = self.n_labels[x], self.n_labels[y]
            _e_energy        = np.zeros((n_lbl_x, n_lbl_y))
            for s_id in self.edges_in_slaves[e_id]:
                e_id_in_s        = self.slave_list[s_id].edge_map[e_id]
                _e_energy    += self.slave_list[s_id].edge_energies[e_id_in_s, :n_lbl_x, :n_lbl_y]
            if not _array_equal(_e_energy, self.edge_energies[e_id, :n_lbl_x, :n_lbl_y]):
                print '\nGraph.check_decomposition: Dual decomposition disagreement for edge %d, with edge_ends %d and %d.' %(e_id, x, y)
                print 'Graph.check_decomposition: Edge energies in PRIMAL are '
                print self.edge_energies[e_id, :n_lbl_x, :n_lbl_y]
                print 'Graph.check_decomposition: Sum of edge energies in the decomposition is '
                print _e_energy
                flag = False

        return flag

    def _compute_param_updates(self, a_start):
        '''
        Compute parameter updates for slaves. 
        Updates are computed after every iteration. 
        This function calls _compute_param_updates_parallel, or 
        _compute_param_updates_sequential, based on the number
        of slaves to be solved. In case they are few in number, 
        the overhead of starting multiple threads to compute
        updates to all of them dominates the execution time, and
        hence it is simpler to compute them sequentially. 
        '''

        # Flags to determine whether to solve a slave.
        slave_flags = np.zeros(self.n_slaves, dtype=bool)

        # The L2-norm of the subgradient. This is calculated incrementally.
        norm_gt = 0.0

        # Change of strategy here: Instead of iterating over all labels, we create 
        #   vectors so that updates can be very easily calculated using array operations!
        # A huge speed up is expected. 
        # Con: Memory usage is increased. 
        # Reset update variables for slaves. 
        self._slave_node_up[:] = 0.0
        self._slave_edge_up[:] = 0.0

        # Set self._sg_node also to zero. 
        #self._sg_node[:,:]     = 0

        # Mark which slaves need updates. A slave s_id needs update only if self._slave_node_up[s_id] 
        #   has non-zero values in the end.
        self._mark_sl_up[:] = 0

        # How many slaves are to be solved?
        n_slaves_to_solve = self._slaves_to_solve.size
        # A simple heuristic. 
        if n_slaves_to_solve > 500000:
            norm_gt = self._compute_param_updates_parallel(a_start)
        else:
            norm_gt = self._compute_param_updates_sequential(a_start)
            
        # Reset the slaves to solve. 
        self._slaves_to_solve = np.where(np.sum(self._mark_sl_up, axis=0)!=0)[0].astype(n_dtype)

        # Record the norm of the subgradient. 
        self.subgradient_norms += [norm_gt]

        # Add momentum.
        if self.it > 1:
            self._slave_node_up = (1.0 - self._momentum)*self._slave_node_up + self._momentum*self._prv_node_sg
            self._slave_edge_up = (1.0 - self._momentum)*self._slave_edge_up + self._momentum*self._prv_edge_sg

        # Compute the alpha for this step. 
        if self._optim_strategy == 'step':
            alpha   = a_start/np.sqrt(self.it)
        elif self._optim_strategy == 'step_ss':
            alpha   = a_start/(1.0 + self.it)
        elif self._optim_strategy == 'step_sg':
            alpha   = a_start/np.sqrt(self.it)
            alpha   = alpha*1.0/norm_gt
        elif self._optim_strategy in ['adaptive', 'adaptive_d', 'adaptive_delta']:
            approx_t    = self._best_primal_cost
            dual_t      = self.dual_costs[-1]
            alpha       = a_start*(approx_t - dual_t)/norm_gt
            if self._optim_strategy == 'adaptive_d':
                alpha   = alpha*1.0/np.sqrt(self.it)
        # Set alpha. 
        self.alpha = alpha


    def _compute_param_updates_parallel(self, a_start):
        ''' 
        Compute parameter updates for slaves, in a parallel manner. 
        '''

        # How many nodes and edges to check for updates. 
        _n_check_nodes = self._check_nodes.size
        _n_check_edges = self._check_edges.size

        # An array to store the contributions to the norm of the 
        #   subgradient from each node and edge. 
        arr_norm_gt    = np.zeros(_n_check_nodes + _n_check_edges)
        # Concatenate indices for check nodes and check edges. 
        concat_indices = np.concatenate((self._check_nodes, self._check_edges))

        # Define a function to compute updates for nodes. 
        def node_updates(n_con, n_id):
            # Retrieve the list of slaves that use this node. 
            s_ids           = self.nodes_in_slaves[n_id]
            n_slaves_nid    = s_ids.size
    
            # Retrieve labels assigned to this point by each slave ...
            ls_int_     = [self.slave_list[s].get_node_label(n_id) for s in s_ids]
            nl_n        = self.n_labels[n_id]
            # ... and make them into one-hot vectors. The previous is needed by self._sg_node. 
            ls_         = np.array([make_one_hot(l_int, nl_n) for l_int in ls_int_])
            ls_avg_     = np.mean(ls_, axis=0)

            # Check if all labellings for this node agree. 
            if np.max(ls_avg_) == 1:
                # As all vectors are one-hot, this condition being true implies that 
                #   all slaves assigned the same label to this node (otherwise, the maximum
                #   number in ls_avg_ would be less than 1).
                return
    
            # The next step was to iterate over all slaves. We calculate the subgradient here
            #   given by 
            #   
            #    \delta_\lambda^s_p = x^s_p - ls_avg_
            #
            #   for all s. s here signifies slaves. 
            # This can be very easily done with array operations!
            _node_up    = ls_ - ls_avg_

            # Add to the subgradient, self._sg_node
            #self._sg_node[n_id, :self.n_labels[n_id]] = np.sum(ls_, axis=0)
    
            # Find the node ID for n_id in each slave in s_ids. 
            sl_nids     = [self.slave_list[s].node_map[n_id] for s in s_ids]
    
            # Mark this update to be done later. 
            self._slave_node_up[s_ids, sl_nids, :nl_n]  = _node_up #:self.n_labels[n_id]] = _node_up
            # Add this value to the subgradient. 
            arr_norm_gt[n_con] = np.sum(_node_up**2)
            # Mark this slave for node updates. 
            self._mark_sl_up[0, s_ids] = 1
    
        # That completes the updates for node energies. Now we move to edge energies. 
        def edge_updates(e_con, e_id):
            # Retrieve the list of slaves that use this edge. 
            s_ids           = self.edges_in_slaves[e_id]
            n_slaves_eid    = s_ids.size

            # Retrieve labellings of this edge, assigned by each slave.
            x, y          = self._node_ids_from_edge_id[e_id,:]
            # n_lables for x and y
            nl_x, nl_y    = self.n_labels[x], self.n_labels[y]
            ls_int_       = [(self.slave_list[s].get_node_label(x), self.slave_list[s].get_node_label(y)) for s in s_ids]
            ls_           = np.array([make_one_hot(l_int, nl_x, nl_y) for l_int in ls_int_])
            ls_avg_       = np.mean(ls_, axis=0, keepdims=True)

            # Check if all labellings for this node agree. 
            if np.max(ls_avg_) == 1:
                # As all vectors are one-hot, this condition being true implies that 
                #   all slaves assigned the same label to this node (otherwise, the maximum
                #   number in ls_avg_ would be less than 1).
                return 

            # The next step was to iterate over all slaves. We calculate the subgradient here
            #   given by 
            #   
            #    \delta_\lambda^s_p = x^s_p - ls_avg_
            #
            #   for all s. s here signifies slaves. 
            # This can be very easily done with array operations!
            _edge_up    = ls_ - ls_avg_
    
            # Find the edge ID for e_id in each slave in s_ids. 
            sl_eids = [self.slave_list[s].edge_map[e_id] for s in s_ids]
    
            # Mark this update to be done later. 
            self._slave_edge_up[s_ids, sl_eids, :nl_x, :nl_y] = _edge_up #:self.n_labels[x]*self.n_labels[y]] = _edge_up
            
            # Add this value to the subgradient. 
            arr_norm_gt[e_con + _n_check_nodes] = np.sum(_edge_up**2)
            # Mark this slave for edge updates. 
            self._mark_sl_up[1, s_ids] = 1

        # We iterate over nodes and edges which associate with at least two slaves
        #    and calculate updates to parameters of all slaves. 
        # Create threads to solve this problem. 
        threads = []
        for n in range(_n_check_nodes):
            t = threading.Thread(target=node_updates, args=(n, self._check_nodes[n],))
            threads.append(t)
        for e in range(_n_check_edges):
            t = threading.Thread(target=edge_updates, args=(e, self._check_edges[e],))
            threads.append(t)

        # Launch these threads ...
        for t in threads:
            t.start()
        #  ... and synchronise. 
        for t in threads:
            t.join()
        
        # The total subgradient norm.
        norm_gt = np.sum(arr_norm_gt)
        # Return the norm of the subgradient. 
        return norm_gt
    
    def _compute_param_updates_sequential(self, a_start):
        '''
        Compute parameter updates for slaves, in a sequential manner. 
        '''

        # Subgradient norm. 
        norm_gt = 0.0

        # We iterate over nodes and edges which associate with at least two slaves
        #    and calculate updates to parameters of all slaves. 
        for n_id in self._check_nodes:
            # Retrieve the list of slaves that use this node. 
            s_ids           = self.nodes_in_slaves[n_id]
            n_slaves_nid    = s_ids.size
    
            # Retrieve labels assigned to this point by each slave ...
            ls_int_     = [self.slave_list[s].get_node_label(n_id) for s in s_ids]
            nl_n        = self.n_labels[n_id]
            # ... and make them into one-hot vectors. The previous is needed by self._sg_node. 
            ls_         = np.array([make_one_hot(l_int, nl_n) for l_int in ls_int_])
            ls_avg_     = np.mean(ls_, axis=0)

            # Check if all labellings for this node agree. 
            if np.max(ls_avg_) == 1:
                # As all vectors are one-hot, this condition being true implies that 
                #   all slaves assigned the same label to this node (otherwise, the maximum
                #   number in ls_avg_ would be less than 1).
                continue
    
            # The next step was to iterate over all slaves. We calculate the subgradient here
            #   given by 
            #   
            #    \delta_\lambda^s_p = x^s_p - ls_avg_
            #
            #   for all s. s here signifies slaves. 
            # This can be very easily done with array operations!
            _node_up    = ls_ - ls_avg_

            # Add to the subgradient, self._sg_node
            #self._sg_node[n_id, :self.n_labels[n_id]] = np.sum(ls_, axis=0)
    
            # Find the node ID for n_id in each slave in s_ids. 
            sl_nids     = [self.slave_list[s].node_map[n_id] for s in s_ids]
    
            # Mark this update to be done later. 
            self._slave_node_up[s_ids, sl_nids, :nl_n]  = _node_up #:self.n_labels[n_id]] = _node_up
            # Add this value to the subgradient. 
            norm_gt += np.sum(_node_up**2)
            # Mark this slave for node updates. 
            self._mark_sl_up[0, s_ids] = 1
    
        # That completes the updates for node energies. Now we move to edge energies. 
        for e_id in self._check_edges:
            # Retrieve the list of slaves that use this edge. 
            s_ids           = self.edges_in_slaves[e_id]
            n_slaves_eid    = s_ids.size
    
            # Retrieve labellings of this edge, assigned by each slave.
            x, y          = self._node_ids_from_edge_id[e_id,:]
            nl_x, nl_y    = self.n_labels[x], self.n_labels[y]
            ls_int_       = [(self.slave_list[s].get_node_label(x), self.slave_list[s].get_node_label(y)) for s in s_ids]
            ls_           = np.array([make_one_hot(l_int, nl_x, nl_y) for l_int in ls_int_])
            ls_avg_       = np.mean(ls_, axis=0, keepdims=True)

            # Check if all labellings for this node agree. 
            if np.max(ls_avg_) == 1:
                # As all vectors are one-hot, this condition being true implies that 
                #   all slaves assigned the same label to this node (otherwise, the maximum
                #   number in ls_avg_ would be less than 1).
                continue    

            # The next step was to iterate over all slaves. We calculate the subgradient here
            #   given by 
            #   
            #    \delta_\lambda^s_p = x^s_p - ls_avg_
            #
            #   for all s. s here signifies slaves. 
            # This can be very easily done with array operations!
            _edge_up    = ls_ - ls_avg_
    
            # Find the edge ID for e_id in each slave in s_ids. 
            sl_eids = [self.slave_list[s].edge_map[e_id] for s in s_ids]
    
            # Mark this update to be done later. 
            self._slave_edge_up[s_ids, sl_eids, :nl_x, :nl_y] = _edge_up #:self.n_labels[x]*self.n_labels[y]] = _edge_up
            
            # Add this value to the subgradient. 
            norm_gt += np.sum(_edge_up**2)
            # Mark this slave for edge updates. 
            self._mark_sl_up[1, s_ids] = 1

        # Return the norm of the subgradient. 
        return norm_gt


    def _apply_param_updates_parallel(self):
        # Apply parameter updates in parallel. Use multiprocessing to create
        #    processes which share memory, and hence do not incur an overhead
        #    of memory coping. 
        
        # Retrieve alpha first. 
        alpha = self.alpha

        # Create a manager to share 
        #   * self.slave_list
        #   * self._slave_node_up
        #   * self._slave_edge_up
        #   # self._mark_sl_up
        manager_sl  = multiprocessing.Manager()

        # Create objects to handle memory sharing. 
        shared_sl   = manager_sl.list()
        # Add slaves to shared list. 
        for i in range(self.n_slaves):
            shared_sl.append(self.slave_list[i])

        # Create shared array for self._slave_node_up
        _sh_sne     = multiprocessing.Array(ctypes.c_float, self.n_slaves*self._max_nodes_in_slave, self.max_n_labels)

        # Create shared array for self._slave_edge_up
        _sh_see     = multiprocessing.Array(ctypes.c_float, self.n_slaves*self._max_nodes_in_slave, self.max_n_labels*self.max_n_labels)

        # Create shared array for self._mark_sl_up
        _sh_msl     = multiprocessing.Array(ctypes.c_byte,  2*self.n_slaves)

        # Make numpy arrays and ask them to share memory with these multiprocessing arrays.
#        _sh_sne_np  = np.frombuffer(_sh_sne.get_obj())
#        shared_sne  = _sh_sne_np.reshape((self.n_slaves, self._max_nodes_in_slave
#            iself._slave_node_up = np.zeros((self.n_slaves, self._max_nodes_in_slave, self.max_n_labels), dtype=u_dtype)
        

    def _apply_param_updates(self):
        # Retrieve alpha
        alpha = self.alpha
        # Perform the marked updates. The slaves to be updates are also the slaves
        #   to be solved!
        for s_id in self._slaves_to_solve:
            s_max_n_labels     = self.slave_list[s_id].max_n_labels
            if self._mark_sl_up[0, s_id]:
                # Node updates have been marked. 
                n_nodes_this_slave = self.slave_list[s_id].n_nodes
                self.slave_list[s_id].node_energies += alpha*self._slave_node_up[s_id, :n_nodes_this_slave, 0:s_max_n_labels]

            if self._mark_sl_up[1, s_id]:
                # Edge updates have been marked. 
                n_edges_this_slave = self.slave_list[s_id].n_edges
                self.slave_list[s_id].edge_energies += alpha*self._slave_edge_up[s_id, :n_edges_this_slave, 0:s_max_n_labels, 0:s_max_n_labels]

        # Copy the subgradient for the next iteration. 
        self._prv_node_sg[:] = self._slave_node_up[:]
        self._prv_edge_sg[:] = self._slave_edge_up[:]


    def force_naive_search(self, disagreements, response='t'):
        ''' 
        Force a naive search for the best solution by varying 
        the labelling of nodes in `disagreements`. Please use
        cautiously, specifying at most three nodes in `disagreements`.
        '''
        while response not in ['y', 'n']:
            print 'Naive search takes exponential time. Supplied disagreeing nodes',
            print 'are %d in number. Proceed? (y/n) ' %(disagreements.size),
            response = raw_input()
            print 'You said: ' + response + '.'
        if response == 'n':
            return
    
        # Get part of the primal solution. 
        labels_ = self._get_primal_solution()

        # Generate all possible labellings. 
        n_labels = self.n_labels[disagreements]
        labellings = _generate_label_permutations(n_labels) 
        
        # Find the minimum. 
        min_energy = np.inf
        min_labels = None
        for l_ in labellings:
            labels_[disagreements] = l_
            _energy = self._compute_primal_cost(labels=labels_)
            if _energy < min_energy:
                min_energy = _energy
                min_labels = l_

        # Set the best labels. 
        print 'Setting the best labels for disagreeing nodes ...'
        self.labels                = labels_
        self.labels[disagreements] = min_labels


    def plot_costs(self):
        f = plt.figure()
        pc,  = plt.plot(self.primal_costs, 'r--', label='PRIMAL') 
        bpc, = plt.plot(np.minimum.accumulate(self.primal_costs), 'r-', label='best PRIMAL')
        dc,  = plt.plot(self.dual_costs, 'b-', label='DUAL')
        plt.legend([pc, bpc, dc], ['PRIMAL', 'best PRIMAL', 'DUAL'])
        plt.show()


    def _generate_trees(self, adj_mat, max_depth=2):
        '''
        Generate a set of trees from a given adjacency matrix. The number of trees is
        equal to the number of nodes in the graph. Each tree has diameter at most 2*max_depth.
        '''
    
        n_nodes = adj_mat.shape[0]
        n_trees = n_nodes
    
        sliced_adjmats = []
        node_lists     = []
        edge_lists     = []

        n_cores = cpu_count() - 1
        
        _inputs  = [[adj_mat, i, max_depth, self._edge_id_from_node_ids] for i in range(n_nodes)]
        _outputs = Parallel(n_jobs=n_cores)(delayed(_generate_tree_with_root)(i) for i in _inputs)

        return _outputs

    def _generate_trees_greedy(self, adjacency=None, max_depth=-1):
        '''
        Generate trees in a greedy manner. The aim is to greedily generate trees starting at the first
        node, so that each tree is as large as possible, and we can skip as many nodes for root as possible. 
        '''
        n_nodes = self.adj_mat.shape[0]

        # Make a copy of the adjacency matrix so that the original is not affected. 
        if adjacency is None: 
            adj_mat_copy = np.zeros_like(self.adj_mat)
            adj_mat_copy[:] = self.adj_mat[:]
        else:
            adj_mat_copy = np.zeros_like(adjacency)
            adj_mat_copy[:] = adjacency[:]

        # Set the max_depth to n_nodes if it is -1, so that the maximum possible tree is discovered every time. 
        if max_depth == -1:
            max_depth = n_nodes

        subtrees_data = []
        for i in range(n_nodes):
            if np.sum(adj_mat_copy[i,:]) == 0:
                continue
            _res = _generate_tree_with_root([adj_mat_copy, i, max_depth, self._edge_id_from_node_ids])
            el = _res[2]    
            # Remove already selected edges from adj_mat_copy.
            for e in el:
                i, j = self._node_ids_from_edge_id[e,:]
                adj_mat_copy[i,j] = adj_mat_copy[j,i] = False
            subtrees_data += [_res]

        return subtrees_data


    def _check_consistency(self):
        '''
        A function to check convergence of the DD-MRF algorithm by checking if all slaves
        agree in their labels of the shared nodes. 
        It works by iterating over the list of subproblems over each node to make sure they 
        agree. If a disagreement is found, we do not have consistency
        '''
        for n_id in range(self.n_nodes):
            s_ids   = self.nodes_in_slaves[n_id]
            ls_     = [self.slave_list[s].get_node_label(n_id) for s in s_ids]
            ret_    = reduce(lambda x,y: x and (y == ls_[0]), ls_[1:], True)
            if not ret_:
                return False
        return True


    def _find_conflicts(self):
        '''
        A function to find disagreeing nodes at a step of the algorithm. 
        '''
        node_conflicts = np.zeros(self.n_nodes, dtype=bool)
        edge_conflicts = np.zeros(self.n_edges, dtype=bool)

        for n_id in range(self.n_nodes):
            if self._n_slaves_nodes[n_id] == 1:
                continue
            s_ids   = self.nodes_in_slaves[n_id]
            ls_     = [self.slave_list[s].get_node_label(n_id) for s in s_ids]
            ret_    = map(lambda x: x == ls_[0], ls_[1:])
            if False in ret_:
                node_conflicts[n_id] = True

        # Update self._check_nodes to find only those nodes where a disagreement exists. 
        self._check_nodes = np.where(node_conflicts == True)[0].astype(n_dtype)
        # Find disagreeing edges. We iterate over self._check_nodes, and add all 
        #    neighbours of a node in _check_nodes. 
        for i in range(self._check_nodes.size):
            n_id = self._check_nodes[i]
            neighs = np.where(self.adj_mat[n_id,:] == True)[0]
            e_neighs = [self._edge_id_from_node_ids[n_id, _n] for _n in neighs]
            edge_conflicts[e_neighs] = True

        # Update self._check_edges to reflect to be only these edges. 
        self._check_edges = np.where(edge_conflicts == True)[0].astype(n_dtype)
        # Return disagreeing nodes. 
        return self._check_nodes


    def _assign_labels(self):
        '''
        Assign the final labels to all points. This function must be called if Graph._check_consistency() returns 
        True. This function simply assigns to every node, the label assigned to it by the first
        slave in its own slave list. Thus, if called without checking consistency first, or even if
        Graph._check_consistency() returned False, it is not guaranteed that this function
        will return the correct labels. 
        Also computes the primal cost for the final labelling. 
        '''
        # Assign labels now. 
#       for n_id in range(self.n_nodes):
#           s_id                = self.nodes_in_slaves[n_id][0]
#           self.labels[n_id]   = self.slave_list[s_id].get_node_label(n_id)
        self.labels      = self._get_primal_solution()
        self.primal_cost = self._compute_primal_cost()

        return self.labels


    def _compute_dual_cost(self):
        '''
        Returns the dual cost at a given stage of the optimisation. 
        The dual cost is simply the sum of all energies of the slaves. 
        '''
        return reduce(lambda x, y: x + y, [s._energy for s in self.slave_list], 0)


    def _get_primal_solution(self):
        '''
        Estimate a primal solution from the obtained dual solutions. 
        This strategy uses the most voted label for every node. 
        '''
        labels = np.zeros(self.n_nodes, dtype=l_dtype)

        # Iterate over every node. 
        if self.decomposition == 'tree' and self._primal_strat == 'bp':
#       if self._primal_strat:          # Adding a method to estimate primals based on ergodic sequences. Commented previous if. 
            # Use Max product messages to compute the best solution. 
        
            # Conflicts are in self._check_nodes. 
            # Assign non-conflicting labels first. 
            for n_id in np.setdiff1d(np.arange(self.n_nodes), self._check_nodes):
                s_id = self.nodes_in_slaves[n_id][0]
                labels[n_id] = self.slave_list[s_id].get_node_label(n_id)

            # Now traverse conflicting labels.  
            node_order = self._check_nodes
            for i in range(node_order.size):
                n_id  = node_order[i]
                n_lbl = self.n_labels[n_id]
                    
                # Check that an edge exists between n_id and (n_id + offset)
                # No top (bottom) edges for vertices in the top (bottom) row.
                # No left edges for vertices in the left-most column. 
                # No right edges for vertices in the right-most column. 
                neighs = np.where(self.adj_mat[n_id,:] == True)[0]
                neighs = [_n for _n in neighs if _n in node_order[:i]]
                
                node_bel = np.zeros(n_lbl)
                if len(neighs) == 0:
                # If there are no previous neighbours, take the maximum of the node belief. 
                    for s_id in self.nodes_in_slaves[n_id]:
                        n_id_in_s = self.slave_list[s_id].node_map[n_id]
                        node_bel += self.slave_list[s_id]._messages_in[n_id_in_s, :n_lbl]
                    labels[n_id] = np.argmax(node_bel)
                else:
                # Else, take the argmax decided by the sum of messages from its neighbours that
                #   have already appeared in node_order. 
                    for _n in neighs:
                        e_id = self._edge_id_from_node_ids[_n, n_id]
                        for s_id in self.edges_in_slaves[e_id]:
                            n_edges_in_s = self.slave_list[s_id].graph_struct['n_edges']
                            _e_id = self.slave_list[s_id].edge_map[e_id]
                            _e_id += n_edges_in_s if _n > n_id else 0
                            node_bel += self.slave_list[s_id]._messages[_e_id, :n_lbl]

                labels[n_id] = np.argmax(node_bel)
        elif self._primal_strat == 'vote':          # Adding a method to estimate primals based on ergodic sequences. Changed else to elif False. 
            for n_id in range(self.n_nodes):
                # Retrieve the labels assigned by every slave to this node. 
                s_ids    = self.nodes_in_slaves[n_id]
                s_labels = [self.slave_list[s].get_node_label(n_id) for s in s_ids]
                # Find the most voted label. 
                labels[n_id] = n_dtype(stats.mode(s_labels)[0][0])
        elif self._primal_strat == 'wsg':
            # Get previous solution. self._wsg_accumulator accumulates weighted subgradients for each iteration. 
            # The weight for a subgradient is just the alpha corresponding to that iteration. 
            self._wsg_accumulator   += self.alpha*self._sg_node

            # Add the current alpha to this accumulator
            self._alpha_accumulator += self.alpha
            # Using the first strategy, compute next solution based on the method of weighted averaging. 
            # x_k = \frac{\sum_{t=1}^k alpha_t*subgrad_t}{\sum_{t=1}^k alpha_t}
            self._wsg_nxt_solution = self._wsg_accumulator/self._alpha_accumulator

            # The resulting labelling is obtained by rounding the solution.
            labels = np.argmax(self._wsg_nxt_solution, axis=1)

        elif self._primal_strat == 'sg':
            # Similarly, we have self._sg_accumulator, which does not weigh the subgradients. 
            self._sg_accumulator    += self._sg_node
            # The second strategy is to keep accumulating the subgradients and dividing by the
            # number of iterations so far. 
            self._sg_nxt_solution  = self._sg_accumulator/self.it

            # The resulting labelling is obtained by rounding the solution.
            labels = np.argmax(self._sg_nxt_solution, axis=1)

        # Return this labelling. 
        return labels

    def _compute_primal_cost(self, labels=None):
        '''
        Returns the primal cost given a labelling. 
        '''
        cost    = 0

        # Generate a labelling first, if not specified. 
        if labels is None:
            labels      = self._get_primal_solution()
            self.labels = labels

        # Compute node contributions.
        for n_id in range(self.n_nodes):
            cost += self.node_energies[n_id][labels[n_id]]

        # Compute edge contributions. 
        for e_id in range(self.n_edges):
            x, y = self._node_ids_from_edge_id[e_id,:]
            cost += self.edge_energies[e_id, labels[x], labels[y]]

        # This is the primal cost corresponding to either the input labels, or the generated ones. 
        return cost


    def _primal_dual_gap(self):
        '''
        Return the primal dual gap at the current stage of the optimisation.
        '''
        return self._compute_primal_cost() - self._compute_dual_cost()


    def _find_empty_attributes(self):
        '''
        Graph._find_empty_attributes(): Returns the list of attributes not set. 
        '''
        # Retrieve the indices for nodes and edges not set. 
        n   = np.where(self.node_flags == False)[0]
        e   = np.where(self.edge_flags == False)[0]

        # Compute the edge list
        edge_list   = [self._node_ids_from_edge_id[e_id,:] for e_id in e]

        # Compute the node_list
        node_list   = n.tolist()
        return node_list, edge_list
# ---------------------------------------------------------------------------------------


def _compute_node_updates(n_id, s_ids, slave_list, n_labels_nid):
    '''
    A function to handle parallel computation of node updates. 
    The entire graph cannot be passed as a parameter to this function, 
    and so we must create a function that is not a member of the class Graph.
    '''
    # The number of slaves.
    n_slaves_nid    = s_ids.size

    # If there is only one slave, we have nothing to do. However, to avoid the overhead
    #   of calling a function that does nothing, we will simply not call this function
    #   for those nodes that belong to only one slave. 

    # Retrieve labels assigned to this point by each slave, and make it into a one-hot vector. 
    ls_     = np.array([make_one_hot(slave_list[s].get_node_label(n_id), n_labels_nid) for s in range(n_slaves_nid)])
    ls_avg_ = np.mean(ls_, axis=0)

    # Check if all labellings for this node agree. 
    if np.max(ls_avg_) == 1:
        # As all vectors are one-hot, this condition being true implies that 
        #   all slaves assigned the same label to this node (otherwise, the maximum
        #   number in ls_avg_ would be less than 1).
        return False, None, None, 0.0

    # The next step was to iterate over all slaves. We calculate the subgradient here
    #   given by 
    #   
    #    \delta_\lambda^s_p = x^s_p - ls_avg_
    #
    #   for all s. s here signifies slaves. 
    # This can be very easily done with array operations!
    _node_up    = ls_ - np.tile(ls_avg_, [n_slaves_nid, 1])

    # Find the node ID for n_id in each slave in s_ids. 
    sl_nids = [slave_list[s].node_map[n_id] for s in range(n_slaves_nid)]

    # Add this value to the subgradient. 
    norm_gt = np.sum(_node_up**2)

    return True, _node_up, sl_nids, norm_gt
# ---------------------------------------------------------------------------------------


def _compute_edge_updates(e_id, s_ids, slave_list, pt_coords, n_labels):
    '''
    A function to handle parallel computation of edge updates. 
    The entire graph cannot be passed as a parameter to this function, 
    and so we must create a function that is not a member of the class Graph.
    '''
    # The number of slaves that this edge belongs to. 
    n_slaves_eid    = s_ids.size

    # If there is only one slave, we have nothing to do. However, to avoid the overhead
    #   of calling a function that does nothing, we will simply not call this function
    #   for those edges that belong to only one slave. 

    # Retrieve labellings of this edge, assigned by each slave.
    x, y    = pt_coords
    ls_     = np.array([
                make_one_hot([slave_list[s].get_node_label(x), slave_list[s].get_node_label(y)], n_labels[0], n_labels[1]) 
                for s in s_ids])
    ls_avg_ = np.mean(ls_, axis=0)

    # Check if all labellings for this node agree. 
    if np.max(ls_avg_) == 1:
        # As all vectors are one-hot, this condition being true implies that 
        #   all slaves assigned the same label to this node (otherwise, the maximum
        #   number in ls_avg_ would be less than 1).
        return False, None, None, 0.0

    # The next step was to iterate over all slaves. We calculate the subgradient here
    #   given by 
    #   
    #    \delta_\lambda^s_p = x^s_p - ls_avg_
    #
    #   for all s. s here signifies slaves. 
    # This can be very easily done with array operations!
    _edge_up    = ls_ - np.tile(ls_avg_, [n_slaves_eid, 1])

    # Find the node ID for n_id in each slave in s_ids. 
    sl_eids = [slave_list[s].edge_map[e_id] for s in range(s_ids)]

    # Add this value to the subgradient. 
    norm_gt = np.sum(_edge_up**2)

    # Mark this slave for edge updates, and return.
    return True, _edge_up, sl_eids, norm_gt

    
def _compute_4node_slave_energy(node_energies, edge_energies, labels):
    '''
    Compute the energy of a slave corresponding to the labels. 
    '''
    [i,j,k,l]   = labels
    
    # Add node energies. 
    total_e     = node_energies[0][i] + node_energies[1][j] + \
                    node_energies[2][k] + node_energies[3][l]

    # Add edge energies. 
    total_e     += edge_energies[0][i,j] + edge_energies[1][i,k] \
                    + edge_energies[2][j,l] + edge_energies[3][k,l]

    return total_e
# ---------------------------------------------------------------------------------------


def _compute_tree_slave_energy(node_energies, edge_energies, labels, graph_struct):
    ''' 
    Compute the energy corresponding to a given labelling for a tree slave. 
    The edges are specified in graph_struct. 
    '''
    
    energy = 0
    for n_id in range(graph_struct['n_nodes']):
        energy += node_energies[n_id][labels[n_id]]
    for edge in range(graph_struct['n_edges']):
        e0, e1 = graph_struct['edge_ends'][edge]
        energy += edge_energies[edge][labels[e0]][labels[e1]]

    return energy
# ---------------------------------------------------------------------------------------


def _compute_cycle_slave_energy(node_energies, edge_energies, labels, node_list):
    '''
    Compute the energy of a cycle. It is assumed that the edges are ordered as 
        x1 - x2
        x2 - x3
        ...
        xN - x1
    where N is the total number of nodes in the cycle.  
    '''
    energy = 0
    n_nodes = node_list.size

    for i in range(n_nodes):
        energy += node_energies[i, labels[i]]

    for e in range(n_nodes):
        end0 = e
        end1 = (e+1)%n_nodes
        l1, l2 = labels[end0], labels[end1]
        if node_list[end0] > node_list[end1]:
            l1, l2 = l2, l1
        energy += edge_energies[e, l1, l2]

    return energy
# ---------------------------------------------------------------------------------------
    

def _optimise_4node_slave(slave):
    '''
    Optimise the smallest possible slave consisting of four vertices. 
    This is a brute force optimisation done by enumerating all possible
    states of the four points and finding the minimum energy. 
    The nodes are arranged as 

            0 ----- 1
            |       |
            |       |
            2 ----- 3,

    where these indices are the same as their indices in node_energies. 

    Input:
        An instance of class Slave which has the following members:
            node_energies:      The node energies for every label,
                                in shape (4, max_n_labels)
            n_labels:           The number of labels for each node. 
                                shape: (4,)
            edge_energies:      Energies for each edge arranged in the order
                                0-1, 0-2, 1-3, 2-3, obeying the vertex order 
                                as well. 
                                shape: (max_num_nodes, max_num_nodes, 4)

    Outputs:
        labels:             The labelling for vertices in the order [0, 1, 2, 3]
        min_energy:         The total energy corresponding to the labelling. 
    '''

    # Extract parameters from the slave. 
    node_energies       = slave.node_energies
    n_labels            = slave.n_labels
    edge_energies       = slave.edge_energies

    # Use already generated all labellings. 
    all_labellings      = slave.all_labellings
    
    # Minimum energy. We set the minimum energy to four times the maximum node energy plus
    #   four times the maximum edge energy. 
    min_energy      = 4*np.max(node_energies) + 4*np.max(edge_energies)

    # The optimal labelling. 
    labels          = np.zeros(4)

    # Record energies for every labelling. 
    for l_ in range(all_labellings.shape[0]):
        total_e     = _compute_4node_slave_energy(node_energies, edge_energies, all_labellings[l_,:])
        # Check if best. 
        if total_e < min_energy:
            min_energy  = total_e
            labels[:]   = all_labellings[l_,:]

    return labels, min_energy
# ---------------------------------------------------------------------------------------


def _optimise_tree(slave):
    ''' 
    Optimise a tree-structured slave. We use max-product belief propagation for this optimisation. 
    The package bp provides a function max_prod_bp, which optimises a given tree based on supplied
    node and edge potentials. However, this function maximises the total potential on a tree. To 
    use it to minimise our energy, we apply exp(-x) on all node and edge energies before passing
    it to max_prod_bp
    '''
    node_pot    = np.array([np.exp(-1*ne) for ne in slave.node_energies])
    edge_pot    = np.array([np.exp(-1*ee) for ee in slave.edge_energies])
    gs          = slave.graph_struct
    # Call bp.max_prod_bp
    labels, messages, messages_in = bp.max_prod_bp(node_pot, edge_pot, gs)
    # We return the energy. 
    energy = _compute_tree_slave_energy(slave.node_energies, slave.edge_energies, labels, slave.graph_struct)
    return labels, energy, messages, messages_in
# ---------------------------------------------------------------------------------------

def _optimise_cycle2(slave):
    ''' Proxy: optimise four node cycle '''
    all_labellings = _generate_label_permutations(slave.n_labels)
    min_energy = np.inf
    min_labels = []

    # Minimum energy. We set the minimum energy to four times the maximum node energy plus
    #   four times the maximum edge energy. 
    min_energy      = 4*np.max(slave.node_energies) + 4*np.max(slave.edge_energies)

    for l_ in all_labellings:
        i, j, k, l = l_
        energy  = slave.node_energies[0][i] + slave.node_energies[1][j] + \
                  slave.node_energies[2][k] + slave.node_energies[3][l]

        node_list = slave.node_list
        for i in range(slave.n_edges):
            n0, n1 = node_list[i], node_list[(i+1)%4]
            l0, l1 = l_[i], l_[(i+1)%4]
            if n0 < n1:
                energy += slave.edge_energies[i, l0, l1]
            else:
                energy += slave.edge_energies[i, l1, l0]
#energy += slave.edge_energies[0, i, j] + slave.edge_energies[1, j, k] + \
#                 slave.edge_energies[2, k, l] + slave.edge_energies[3, l, i]

        if energy < min_energy:
            min_energy = energy
            min_labels = l_
    
    return min_labels, min_energy

def _optimise_cycle(slave):
    '''
    Optimise a cycle-slave. We use the fast cycle solver of Wang and Koller (ICML 2013).
    At the core, the code is the one released by Wang. However, a Python-wrapped
    version of that code is used here. 
    '''

    # Node energies must be an (N, max_n_labels) Numpy np.float32 array. 
    node_energies    = np.zeros_like(slave.node_energies)
    node_energies[:] = slave.node_energies[:]
    # Edge energies must be an (n, max_n_labels*max_n_labels) Numpy np.float32 array. 
    # It thus needs to be reshaped here. 
    edge_energies = np.zeros((slave.edge_energies.shape[0], slave.max_n_labels*slave.max_n_labels))
    for n in range(slave.n_nodes):
        e0, e1                       = n, (n+1)%slave.n_nodes
        nl_e0                        = slave.n_labels[e0]
        nl_e1                        = slave.n_labels[e1]
        numel_pw                     = nl_e0*nl_e1
        if slave.node_list[e0] < slave.node_list[e1]:
            slave_edge_energies      = slave.edge_energies[n, :nl_e0, :nl_e1]
        else:
            slave_edge_energies      = np.transpose(slave.edge_energies[n, :nl_e1, :nl_e0])

        # [0, 1, 0] needed in np.reshape because Wang's code assumes column-first ordering!
        edge_energies[n, 0:numel_pw] = np.reshape(slave_edge_energies, (numel_pw,), [0, 1, 0])      
    # The list of the number of labels. 
    n_labels      = np.zeros_like(slave.n_labels).astype(np.int)        # Again, the Python wrapper requires np.int!
    n_labels[:]   = slave.n_labels[:]

    # Adjust energies so that the least energy is zero. 
    n_min = np.min(node_energies)
    e_min = np.min(edge_energies)
    o_min = min(n_min, e_min)
    node_energies -= o_min
    edge_energies -= o_min

    # Solve the cycle. 
    labels        = fsc.solver(node_energies, edge_energies, n_labels)
    # Compute energy of labelling. 
    energy        = _compute_cycle_slave_energy(slave.node_energies, slave.edge_energies, labels, slave.node_list)
    # Return the labelling and the energy.
    return labels, energy

#   # Brute force solver here. 
#   all_labellings = slave.all_labellings
#   min_energy     = 1e10
#   min_labels     = []
#   for i in range(len(all_labellings)):
#       cur_labels = all_labellings[i]
#       cur_energy = _compute_cycle_slave_energy(slave.node_energies, slave.edge_energies, cur_labels, slave.node_list)
#       if cur_energy < min_energy:
#           min_energy = cur_energy
#           min_labels = cur_labels
#   return min_labels, min_energy

# ---------------------------------------------------------------------------------------


def _update_slave_states(c):
    '''
    Update the states for slave (given by c[1]) and set its labelling to the specified one
    (given by c[2][0]). Also performs a sanity check on c[1]._compute_energy and c[2][1], which
    is the optimal energy returned by _optimise_4node_slave().
    '''
    i                   = c[0]
    s                   = c[1]
    [s_labels, s_min]   = c[2]

    # Set the labels in s. 
    s.set_labels(s_labels)
    s._compute_energy()

    # Sanity check. The two energies (s_min and s._energy) must agree.
    if s._energy != s_min:
        print '_update_slave_states(): Consistency error. The minimum energy returned \
by _optimise_4node_slave() for slave %d is %g and does not match the one computed \
by Slave._compute_energy(), which is %g. The labels are [%d, %d, %d, %d]' \
                %(i, s_min, s._energy, s_labels[0], s_labels[1], s_labels[2], s_labels[3])
        return False, None

    # Everything is okay.
    return True, s

# ---------------------------------------------------------------------------------------
# Use shared list shared_slave_list to optimise slaves. 
def _optimise_slave_mp(i):
    s   = shared_slave_list[i]
    ret = _optimise_slave(s)
    return ret

# ---------------------------------------------------------------------------------------


def optimise_all_slaves(slaves):
    '''
    A function to optimise all slaves. This function distributes the job of
    optimising every slave to all but one cores on the machine. 
    Inputs:
        slaves:     A list of objects of type Slave. 
    '''
    # The number of cores to use is the number of cores on the machine minum 1. 
    n_cores     = cpu_count() - 1
    optima      = Parallel(n_jobs=n_cores)(delayed(_optimise_4node_slave)(s) for s in slaves)

    # Update the labelling in slaves. 
    success     = np.array(Parallel(n_jobs=n_cores)(delayed(_update_slave_states)(c) for c in zip(range(len(slaves)), slaves, optima)))
    if False in success:
        print 'Update of slave states failed for ',
        print np.where(success == False)[0]
        raise AssertionError

# ---------------------------------------------------------------------------------------


def _optimise_slave(s):
    '''
    Function to handle optimisation of any random slave. 
    '''
    if s.struct == 'cell':
        return _optimise_4node_slave(s)
    elif s.struct == 'tree':
        return _optimise_tree(s)
    elif s.struct == 'cycle':
        return _optimise_cycle(s)
    elif s.struct == 'free_node':
        nl = np.argmin(s.node_energies)
        return [nl], s.node_energies[0,nl]
    elif s.struct == 'free_edge':
        # Get all labellings of the two nodes involved. 
        _labels = _generate_label_permutations(s.n_labels)
        # First set of labels. 
        lx, ly = _labels[0]
        # Min energy so far ...
        min_energy = s.node_energies[0,lx] + s.node_energies[1,ly] + s.edge_energies[0,lx,ly]
        min_labels = [lx, ly]
        # Find overall minimum energy. 
        for [lx, ly] in _labels[1:]:
            _energy = s.node_energies[0,lx] + s.node_energies[1,ly] + s.edge_energies[0,lx,ly]
            if _energy < min_energy:
                min_energy = _energy
                min_labels = [lx, ly]
        # Return the minimum energy. 
        return min_labels, min_energy
    else:
        print 'Slave structure not recognised: %s.' %(s.struct)
        raise ValueError
# ---------------------------------------------------------------------------------------


def make_one_hot(label, s1, s2=None):
    '''
    Make one-hot vector for the given label depending on the number of labels
    specified by s1 and s2. 
    '''
    # Make sure the input label conforms with the input dimensions. 
    if s2 is None:
        if type(label) == list:
            print 'Please specify an int label for unary energies.'
            raise ValueError
        label = int(label)

    # Number of labels in the final vector. 
    size = s1 if s2 is None else (s1, s2)

    # Make final vector. 
    oh_vec = np.zeros(size, dtype=np.bool)
    
    # Set label.
    oh_vec[label] = True
    # Return 
    return oh_vec
# ---------------------------------------------------------------------------------------


def _generate_label_permutations(n_labels):
    if n_labels.size == 1:
        return [[i] for i in range(n_labels[0])]

    _t   = _generate_label_permutations(n_labels[1:])

    _ret = []
    for i in range(n_labels[0]):
        _ret += [[i] + _tt for _tt in _t]

    return _ret
# ---------------------------------------------------------------------------------------
    

def _generate_tree_with_root(_in):
    '''
    Generate a tree of max depth specified by max_depth, and with root specified by root. 
    '''

    adj_mat                = _in[0]
    root                   = _in[1]
    max_depth              = _in[2]
    _edge_id_from_node_ids = _in[3]

    # Create a queue to traverse the graph in a bredth-first manner
    # Each element of the queue is a pair, where the first of the 
    #    pair specifies the vertex, and the second specifies the depth. 
    queue = [[root, 0]]

    # The current depth of the tree. 
    c_depth = 0

    # The number of nodes. 
    n_nodes = adj_mat.shape[0]
    
    # Create the output adjacency matrix. 
    tree_adjmat = np.zeros((n_nodes, n_nodes), dtype=np.bool)

    # Record whether we already visited a node. 
    visited = np.zeros(n_nodes, dtype=np.bool)

    # We have alredy visited root. 
    visited[root] = True

    # The node list for this subgraph. 
    node_list = None
    # The edge list for this subgraph. 
    edge_list = None

    while len(queue) > 0:
        # The current root in the traversal. 
        _v, _d = queue[0]
        # Pop this vertex from the queue. 
        queue = queue[1:]

        # If we have reached the maximum allowed depth, stop.
        if _d == max_depth:
            continue
        
        # Neighbours of _v that we have not already visited. 
        neighbours = [i for i in np.where(adj_mat[_v, :] == True)[0] if not visited[i]]

        # If we have no more possible neighbours, stop.
        if len(neighbours) == 0:
            continue

        # Mark all neighbours as visited. 
        visited[neighbours] = True

        # Add these edges to the adjacency matrix. 
        tree_adjmat[_v, neighbours] = True
#       for _n in neighbours:
#           _from = np.min([_v, _n])        # Edges are always from low index to high index. 
#           _to   = _v + _n - _from         
#           tree_adjmat[_from, _to] = True
#       tree_adjmat[_v, neighbours] = True          # Old. Possibly wrong. 

        # Insert these in the queue. 
        _next_nodes = [[_n, _d + 1] for _n in neighbours]
        queue += _next_nodes    

    # The node list can be obtained directly from visited. 
    node_list = np.where(visited == True)[0].astype(n_dtype)


    # Make adjacency matrix symmetric. 
    tree_adjmat = tree_adjmat + tree_adjmat.T
    # These are the edge ends. 
    edge_ends = np.where(np.tril(tree_adjmat).T == True)
    edge_list = [_edge_id_from_node_ids[e0,e1] for e0, e1 in zip(edge_ends[0], edge_ends[1])]

    # Also create a sliced matrix, only from the visited nodes. 
    _sliced = tree_adjmat[node_list,:][:,node_list]

    # Return this adjcency matrix. 
    return _sliced, node_list, edge_list

# ---------------------------------------------------------------------------------------


def find_cycle_in_graph(adj, root):
    # Given adjacency matrix adj, and an arbitrary vertex a root, 
    #    return a cycle in the graph containing root. 

    # The number of nodes in the graph
    n_nodes = adj.shape[0]

    # Initialise queue. 
    queue = [root]

    # No longest path initially. 
    longest_path = []

    # The list of shortest paths to a given node from root. 
    shortest_paths       = [[] for i in range(n_nodes)]
    shortest_paths[root] = [root]

    # Visited array: whether we have already visited a node. 
    visited    = np.zeros(n_nodes, dtype=np.bool)
    visited[root] = True

    # BFS:
    while len(queue) > 0:
        # The current node that we are analysing ...
        cur_node = queue[0]
        # Pop the root. 
        queue    = queue[1:]

        # Path to current node. 
        cur_path = shortest_paths[cur_node]

        # Neighbours of current node. 
        neighs   = np.where(adj[cur_node] == True)[0]

        # Check if neighbours have been visited. 
        neighs = np.array([_n for _n in neighs if _n not in cur_path])

        if neighs.size == 0:
            return None

        if np.sum(visited[neighs]) == 0:
            for n in neighs:
                shortest_paths[n] = cur_path + [n]
            # Set visited to True for these neighbours. 
            visited[neighs] = True

            # Push them into the queue. 
            queue += np.random.permutation(neighs).tolist()
        else:
            # Visited neighbours. 
            visited_neighs = neighs[np.where(visited[neighs] == True)[0].tolist()]
            # Remove visited neighbours that have a point in common with our current path. 
            visited_neighs = [v for v in visited_neighs if np.intersect1d(shortest_paths[v][1:], cur_path[1:]).size == 0]
            # If there are no such neighbours, i.e., if visited_neighs is empty, return None - there are no 
            #    cycles of the type desired. 
            if len(visited_neighs) == 0:
                return None

            # Path lengths for all visited neighbours. 
            visited_plengths = [len(shortest_paths[v]) + len(cur_path) for v in visited_neighs]

            # Longest path
            lid          = np.argmax(visited_plengths)
            longest_path = shortest_paths[visited_neighs[lid]] + cur_path[-1::-1]
            # No need to continue further. 
            break

    if longest_path == []:
        return None

    return longest_path
# ---------------------------------------------------------------------------------------


def find_longest_cycle_in_graph(adj_mat, root):
    '''
    Find the longest cycle in given graph, containing node root. Very expensive computation.
    '''
    # Number of nodes. 
    n_nodes       = adj_mat.shape[0]
    # Initialise queue. 
    queue         = []

    # All paths to all nodes. 
    paths_to_node = [[] for n in range(n_nodes)]

    # Whether we have visited a node. 
    visited       = np.zeros(n_nodes, dtype=np.bool)

    # Longest cycle is empty initially. 
    longest_cycle = []

    # Set the paths from root to root. 
    paths_to_node[root] = [[root]]  

    while len(queue) > 0:
        # Current node. 
        cur_node   = queue[0]
        # Pop current node. 
        queue      = queue[1:]

        # Mark this node as visited. 
        visited[cur_node] = True

        # Paths to current node. 
        cur_paths  = paths_to_node[cur_node]

        # Check if node is already visited. If yes, compare all pairings 
        #    of paths to get to this node from the previous node. 
        neighs     = np.where(adj_mat[cur_node,:] == True)[0]   

        # Filter neighbours that are in cur_paths: we cannot have a cycle with 
        #    node occurring twice. 
        neighs     = np.array([_n for _n in neighs if not np.sum([_n in c_path for c_path in cur_paths])])

        if neighs.size == 0:
            return None

        # Insert non-visited neighbours into queue. 
        non_visited = np.array([_n for _n in neighs if not visited[_n]])


# ---------------------------------------------------------------------------------------
def dfs_unique_cycles(adj_mat, max_length=-1):
    '''
    Start DFS at every node in the graph to find a cycle with maximum length 
    given by max_length. Very expensive computation.
    '''
    def _contained_in(l1, l2):
        ''' 
        Whether list l2 is contains list l1. Cyclic rotations and reversals allowed. 
        '''
        s1, s2 = len(l1), len(l2)
        if s1 > s2:
            return False
        
        stotal   = s2 + s2
        ltotal   = l2 + l2
        for i in range(stotal - s1):
            if ltotal[i:i+s1] == l1 or ltotal[i:i+s1] == l1[::-1]:
                return True
        return False

    n_nodes = adj_mat.shape[0]

    # Parallel processing: find longest cycle for every node independently
    #    as these operations do not interact with each other, for different nodes. 
    n_cores = np.min([n_nodes, cpu_count() - 1])

    # Create inputs: 
    adj_mat_int = adj_mat.astype(np.int)        # Can't skip np.int here! The C program requires np.int
    _inputs = [[adj_mat_int, _n, max_length] for _n in range(n_nodes)]

    # Run the algorithm. 
# ---------------------------------------------------
    # Using JOBLIB. 
#   list_cycles = Parallel(n_jobs=n_cores)(delayed(dfs_cycle_)(_in) for _in in _inputs)
    list_cycles = Parallel(n_jobs=n_cores)(delayed(find_cycle_)(_in) for _in in _inputs)

    # Remove double cycles - cycles that have the same nodes, but in some other order. 
    kept_cycles = []
    for c_n in list_cycles:
        # Do not include a failed search.
        if len(c_n) == 0:
            continue

        # Whether we have already seen a cycle. 
        _already_seen = False
        for k_cn in kept_cycles:
            if _contained_in(c_n.tolist(), k_cn.tolist()): #np.array_equal(np.sort(k_cn), np.sort(c_n)):
                # This cycle has already been included. No need to include it again.
                _already_seen = True
                break
        if not _already_seen:
            kept_cycles += [c_n]

    # Return the list of cycles. 
    return kept_cycles
# ---------------------------------------------------

    # Doing sequentially, as JOBLIB seems to be very expensive!
#    list_cycles  = []
#    while True:
#        _cycles_found = False
#        for _n in range(n_nodes):
#            c_n = find_cycle(adj_mat_int, _n, max_length)
#            _already_seen = False
#            for k_cn in list_cycles:
#                if _contained_in(c_n.tolist(), k_cn.tolist()):
#                    _already_seen = True
#                    break
#            if not _already_seen:
#                list_cycles += [c_n]
#                _cycles_found = True
#        if not _cycles_found:
#            break  
#
#    return list_cycles


# ---------------------------------------------------------------------------------------
def find_cycle_(_in):
    '''
    Handler function for dfs_cycle.find_cycle().
    '''
    return find_cycle(_in[0], _in[1], _in[2])

# ---------------------------------------------------------------------------------------
def dfs_cycle_(_in):
    '''
    Start DFS at root to find the longest cycle. Very expensive computation.
    '''
    # Extract inputs. 
    adj_mat    = _in[0]
    root       = _in[1]
    max_length = _in[2]
    
    # Number of nodes in the graph. 
    n_nodes = adj_mat.shape[0]

    # Whether we have a visited a node or not. 
    visited = np.zeros(n_nodes, dtype=np.bool)
    visited[root] = True

    # The stack used for depth-first search. 
    stack = [[root, [root], visited]]

    # Stores the current longest cycle. 
    longest_cycle = []

    # Depth first search. 
    while len(stack) > 0:
        # The current node we are working on. 
        cur_node    = stack[0][0]
        cur_path    = stack[0][1]
        cur_visited = stack[0][2]

        # Pop the top-most node in the stack. 
        stack = stack[1:]

        if len(cur_path) > max_length:
            continue
    
        # Retrieve all neighbours of cur_node. Exclude the ones that 
        #    have already been visite, save the root, because we are explicitly
        #    looking for cycles containing the root. 
        neighs = np.where(adj_mat[cur_node,:] == True)[0]
        neighs = filter(lambda x: not cur_visited[x] or x == root, neighs)

        for _n in neighs:
            # If one of the neighbours is a root, we have found a cycle. 
            if _n == root:
                # Update our longest cycle if this one is longer than longest_cycle. 
                if len(cur_path) > 2 and len(cur_path) > len(longest_cycle):
                    longest_cycle = cur_path
                    # If we already have a cycle of length max_length, there is no need
                    #    to keep looking for a new cycle. 
                    if len(cur_path) == max_length:
                        return longest_cycle
            elif _n not in cur_path:
                # Create path and visited variables for the next node. 
                _n_visited = np.zeros_like(cur_visited)
                _n_visited[:] = cur_visited[:]
                # Only _n is set to True in this version of visited, as the other neighbours of
                #    cur_node have not been visited yet. 
                _n_visited[_n] = True
                # Push this node into the stack, with its corresponding visited array and path. 
                stack = [[_n, cur_path + [_n], _n_visited]] + stack

    return longest_cycle

