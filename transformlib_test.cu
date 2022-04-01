#include <stdio.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "fftx3.hpp"
#include "fftx_mddft_public.h"

static int M, N, K;

static void buildInputBuffer(double *host_X, double *X)
{
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				host_X[(k + n*K + m*N*K)*2 + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
				host_X[(k + n*K + m*N*K)*2 + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
			}
		}
	}

	cudaMemcpy(X, host_X, M*N*K*2*sizeof(double), cudaMemcpyHostToDevice);
	return;
}

static void checkOutputBuffers ( double *Y, double *cufft_Y )
{
	printf("cube = [ %d, %d, %d ]\t", M, N, K);
	cufftDoubleComplex *host_Y       = new cufftDoubleComplex[M*N*K];
	cufftDoubleComplex *host_cufft_Y = new cufftDoubleComplex[M*N*K];

	cudaMemcpy(host_Y      ,       Y, M*N*K*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_cufft_Y, cufft_Y, M*N*K*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	bool correct = true;
	int errCount = 0;
	double maxdelta = 0.0;

	for (int m = 0; m < 1; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				cufftDoubleComplex s = host_Y      [k + n*K + m*N*K];
				cufftDoubleComplex c = host_cufft_Y[k + n*K + m*N*K];
	    
				bool elem_correct =
					(abs(s.x - c.x) < 1e-7) &&
					(abs(s.y - c.y) < 1e-7);
				maxdelta = maxdelta < (double)(abs(s.x -c.x)) ? (double)(abs(s.x -c.x)) : maxdelta ;
				maxdelta = maxdelta < (double)(abs(s.y -c.y)) ? (double)(abs(s.y -c.y)) : maxdelta ;

				correct &= elem_correct;
				if (!elem_correct && errCount < 10) 
				{
					correct = false;
					errCount++;
					//  printf("error at (%d,%d,%d): %f+%fi instead of %f+%fi\n", k, n, m, s.x, s.y, c.x, c.y);
				}
			}
		}
	}
	
	printf ( "Correct: %s\tMax delta = %E\t\t##PICKME## \n", (correct ? "True" : "False"), maxdelta );
	fflush ( stdout );
	delete[] host_Y;
	delete[] host_cufft_Y;

	return;
}

int main() {

	fftx::point_t<3> *wcube, curr;
	int iloop = 0;
	double *X, *Y;
	double sym[100];  // dummy symbol
						  
	//  cudaEvent_t start, stop, custart, custop;

	wcube = fftx_mddft_QuerySizes ();
	if (wcube == NULL) {
		printf ( "Failed to get list of available sizes\n" );
		exit (-1);
	}

	transformTuple_t *tupl;
	for ( iloop = 0; ; iloop++ ) {
		curr = wcube[iloop];
		if ( curr.x[0] == 0 && curr.x[1] == 0 && curr.x[2] == 0 ) break;

		printf ( "Cube size { %d, %d, %d } is available\n", curr.x[0], curr.x[1], curr.x[2]);
		tupl = fftx_mddft_Tuple ( wcube[iloop] );
		if ( tupl == NULL ) {
			printf ( "Failed to get tuple for cube { %d, %d, %d }\n", curr.x[0], curr.x[1], curr.x[2]);
		}
		else {
			M = curr.x[0], N = curr.x[1], K = curr.x[2];
			printf ( "M = %d, N = %d, K = %d, malloc sizes = %d * sizeof(double)\n", M, N, K, M*N*K*2 );
		
			cudaMalloc(&X,M*N*K*2*sizeof(double));
			cudaMalloc(&Y,M*N*K*2*sizeof(double));

			double *host_X = new double[M*N*K*2];

			cufftDoubleComplex *cufft_Y; 
			cudaMalloc(&cufft_Y, M*N*K * sizeof(cufftDoubleComplex));

			cufftHandle plan;
			if (cufftPlan3d(&plan, M, N, K,  CUFFT_Z2Z) != CUFFT_SUCCESS) {
				exit(-1);
			}

			//  Call the init function
			( * tupl->initfp )();
			checkCudaErrors ( cudaGetLastError () );
 
			// set up data in input buffer and run the transform
			buildInputBuffer(host_X, X);

			for ( int kk = 0; kk < 100; kk++ ) {
				//  try the run function

				( * tupl->runfp ) ( Y, X, sym );
				checkCudaErrors ( cudaGetLastError () );
			}
			
			// Tear down / cleanup
			( * tupl->destroyfp ) ();				//  destroy_mddft3d();
			checkCudaErrors ( cudaGetLastError () );

			if (cufftExecZ2Z(
					plan,
					(cufftDoubleComplex *) X,
					(cufftDoubleComplex *) cufft_Y,
					CUFFT_FORWARD
					) != CUFFT_SUCCESS) {
				printf("cufftExecZ2Z launch failed\n");
				exit(-1);
			}

			cudaDeviceSynchronize();
			if (cudaGetLastError() != cudaSuccess) {
				printf("cufftExecZ2Z failed\n");
				exit(-1);
			}

			//  check cufft and CUDA got same results
			checkOutputBuffers ( Y, (double *)cufft_Y );
			
			cudaFree ( X );
			cudaFree ( Y );
			cudaFree ( cufft_Y );
			delete[] host_X;
		}
	}

}
