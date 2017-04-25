/*
  This file is part of CycleSolver.

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

using namespace CycleSolver;

int main(int argc, char* argv[])
{
	// srand(0);
	srand(time(NULL));

	if(argc!=3){
		printf("Usage: ./cycle N K\n");
		printf("N = length of cycle\n");
		printf("K = variable cardinality\n");
		exit(0);
	}

	int i=1;
	int N = atoi(argv[i++]);
	int K = atoi(argv[i++]);

	IdxType e_end;

	// hardness of problem, 0.0:hardest; 1.0:easy; >1.0: trivial
	double unary_strength = 1.0; //atof(argv[i++]); 

	Cycle c;
	//  c.randomCycle(N, K, unary_strength);
	std::vector< std::vector<ValType> > unaries;
	std::vector< std::vector<ValType> > pairwise;
	std::vector<int>                    n_labels;

	n_labels.assign(N, K);

	for(IdxType i = 0; i < N; i ++)
	{
		n_labels[i] = K;
	}

	unaries.assign(N, std::vector<ValType>());
	pairwise.assign(N, std::vector<ValType>());
	for(IdxType i = 0; i < N; i ++)
	{
		unaries[i].assign(n_labels[i], 0);
		for(IdxType j = 0; j < n_labels[i]; j ++)
			unaries[i][j] = randn(0.0, 1.0);

		e_end = (i + 1)%N;
		pairwise[i].assign(n_labels[i]*n_labels[e_end], 0);
		for(IdxType j = 0; j < n_labels[i]*n_labels[e_end]; j ++)
			pairwise[i][j] = randn(0.0, 1.0);
	}

	c.initialiseCycle(unaries, pairwise, n_labels);
	printf("Created random problem with N=%d, K=%d\n", N, K);

	// Note: Here we use normal distribution in generating random cycle,
	// the resulted problem are ``harder'' than using a uniform distribution.

	clock_t t;
	double time;
	c.runFastSolver(0);
	time = ((double)clock()-t)/CLOCKS_PER_SEC;
	printf("Fast solver,   time=%f secs, obj=%f, assignment=", time, c.obj);
	print_vector(c.assignment);

	c.freeMemory();
}
