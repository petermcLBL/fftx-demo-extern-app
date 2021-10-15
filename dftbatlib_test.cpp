#include <stdio.h>
//  #include <hip/hip_runtime.h>
//  #include <hipfft.h>
//  #include "rocfft.h"
#include <stdlib.h>
#include <string.h>

#include "fftx_dftbat_public.h"

//  Size will be defined when compiling 
//  #define M		100
//  #define N		224
//  #define K		224
//  #define FUNCNAME		mddft3d

static int M, N, K;

static void buildInputBuffer(double *X)
{
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			X[(n + m*N) * 2 + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
			X[(n + m*N) * 2 + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
		}
	}

	return;
}

#if 0
static void checkOutputBuffers ( double *Y, double *hipfft_Y )
{
	printf("cube = [ %d, %d, %d ]\t", M, N, K);
	hipfftDoubleComplex *host_Y        = new hipfftDoubleComplex[M*N*K];
	hipfftDoubleComplex *host_hipfft_Y = new hipfftDoubleComplex[M*N*K];

	hipMemcpy(host_Y       ,        Y, M*N*K*sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
	hipMemcpy(host_hipfft_Y, hipfft_Y, M*N*K*sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);

	bool correct = true;
	int errCount = 0;
	double maxdelta = 0.0;

	for (int m = 0; m < 1; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				hipfftDoubleComplex s = host_Y       [k + n*K + m*N*K];
				hipfftDoubleComplex c = host_hipfft_Y[k + n*K + m*N*K];
	    
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
	delete[] host_Y;
	delete[] host_hipfft_Y;

	return;
}
#endif

int main() {

	fftx::point_t<2> *wcube, curr;
	int iloop = 0;
	double *X, *Y;

	wcube = fftx_dftbat_QuerySizes();
	if (wcube == NULL) {
		printf ( "Failed to get list of available sizes\n" );
		exit (-1);
	}

	transformTuple_t *tupl;
	for ( iloop = 0; ; iloop++ ) {
		//  loop thru all the sizes in the library...
		curr = wcube[iloop];
		if ( curr.x[0] == 0 && curr.x[1] == 0 ) break;

		printf ( "Size { %d, %d } is available\n", curr.x[0], curr.x[1] );
		tupl = fftx_dftbat_Tuple ( wcube[iloop] );
		if ( tupl == NULL ) {
			printf ( "Failed to get tuple for size { %d, %d }\n", curr.x[0], curr.x[1] );
		}
		else {
			M = curr.x[0], N = curr.x[1];
			printf ( "M (batches) = %d, N (size) = %d, malloc sizes = %d * sizeof(double)\n", M, N, M*N*2 );
		
			X = (double *) malloc ( M * N * 2 * sizeof(double) );
			Y = (double *) malloc ( M * N * 2 * sizeof(double) );

			//  Call the init function
			( * tupl->initfp )();
 
			// set up data in input buffer and run the transform
			buildInputBuffer(X);

			for ( int kk = 0; kk < 100; kk++ ) {
				( * tupl->runfp ) ( Y, X );
			}
			
			// Tear down / cleanup
			( * tupl->destroyfp ) ();

			free ( X );
			free ( Y );
		}
	}

	//  Find specific entries in the library based on the size parameters...
	fftx::point_t<2> szs[] = { {1, 32}, {1, 60}, {1, 256}, {4, 64},
							   {16, 256}, {16, 1024}, {16, 400}, {0, 0} };
	for ( iloop = 0; ; iloop++) {
		curr = szs[iloop];
		if ( curr.x[0] == 0 && curr.x[1] == 0 ) break;

		M = curr.x[0], N = curr.x[1];
		printf ( "Get function pointers for batch = %d, size = %d", M, N);
		tupl = fftx_dftbat_Tuple ( curr);
		if ( tupl != NULL ) {
			//  allocate memory
			printf ( "\nAllocate buffers of size %d for input/output", M * N * 2 * sizeof(double) );
			X = (double *) malloc ( M * N * 2 * sizeof(double) );
			Y = (double *) malloc ( M * N * 2 * sizeof(double) );

			//  Call the init function
			( * tupl->initfp )();
 
			// set up data in input buffer and run the transform
			buildInputBuffer(X);

			for ( int kk = 0; kk < 100; kk++ ) {
				( * tupl->runfp ) ( Y, X );
			}
			printf ( " -- Run the transform\n" );
			
			// Tear down / cleanup
			( * tupl->destroyfp ) ();

			free ( X );
			free ( Y );
		}
		else {
			printf ( " -- No entry in library for this size\n" );
		}
	}
		
}
