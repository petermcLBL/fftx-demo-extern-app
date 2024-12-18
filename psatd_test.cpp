#include <stdio.h>

#include "fftx.hpp"
#include "fftx_psatd_gpu_public.h"
//  #include "fftx_imddft_gpu_public.h"
#include "fftxdevice_macros.h"

static int M, N, K;

static bool debug_print = false;

static void buildInputBuffer(double *host_X, double *X)
{
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				host_X[(k + n*K + m*N*K)] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
			}
		}
	}

	FFTX_DEVICE_MEM_COPY ( X, host_X, M*N*K*sizeof(double), FFTX_MEM_COPY_HOST_TO_DEVICE);
	FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR () );
	if ( debug_print ) printf ( "buildInputBuffer: Randomized input buffer copied to device memory\n" );
	return;
}

static void checkOutputBuffers ( double *Y, double *cufft_Y )
{
	printf("cube = [ %d, %d, %d ]\t", M, N, K);
	FFTX_DEVICE_FFT_DOUBLECOMPLEX *host_Y       = new FFTX_DEVICE_FFT_DOUBLECOMPLEX[M*N*K];
	FFTX_DEVICE_FFT_DOUBLECOMPLEX *host_cufft_Y = new FFTX_DEVICE_FFT_DOUBLECOMPLEX[M*N*K];

	FFTX_DEVICE_MEM_COPY ( host_Y      ,       Y, M*N*K*sizeof(FFTX_DEVICE_FFT_DOUBLECOMPLEX), FFTX_MEM_COPY_DEVICE_TO_HOST );
	FFTX_DEVICE_MEM_COPY ( host_cufft_Y, cufft_Y, M*N*K*sizeof(FFTX_DEVICE_FFT_DOUBLECOMPLEX), FFTX_MEM_COPY_DEVICE_TO_HOST );

	bool correct = true;
	int errCount = 0;
	double maxdelta = 0.0;

	for (int m = 0; m < 1; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				FFTX_DEVICE_FFT_DOUBLECOMPLEX s = host_Y      [k + n*K + m*N*K];
				FFTX_DEVICE_FFT_DOUBLECOMPLEX c = host_cufft_Y[k + n*K + m*N*K];
	    
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


int main ( int argc, char* argv[] ) {

	fftx::point_t<3> *wcube, curr;
	int iloop = 0;
	bool oneshot = false;
	double *X, *Y;
	double sym[100];  // dummy symbol
	char *prog = argv[0];
						  
	debug_print = true;
	int baz = 0;
	while ( argc > 1 && argv[1][0] == '-' ) {
		switch ( argv[1][1] ) {
		case 's':        // size specified, must be of form MMxNNxKK
			argv++, argc--;
			M = atoi ( argv[1] );
			while ( argv[1][baz] != 'x' ) baz++;
			baz++ ;
			N = atoi ( & argv[1][baz] );
			while ( argv[1][baz] != 'x' ) baz++;
			baz++ ;
			K = atoi ( & argv[1][baz] );
			oneshot = true;
			break;
			//  debug_print = true;

		case 'h':
			printf ( "Usage: %s: [ -s MMxNNxKK ]\n", argv[0] );
			exit (0);

		default:
			printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
		}
		argv++, argc--;
	}

	wcube = fftx_psatd_QuerySizes ();
	if (wcube == NULL) {
		printf ( "Failed to get list of available sizes\n" );
		exit (-1);
	}

	if ( oneshot ) {
		for ( iloop = 0; ; iloop++ ) {
			if ( wcube[iloop].x[0] == M && wcube[iloop].x[1] == N && wcube[iloop].x[2] == K ) {
				break;
			}
		}
	}
	transformTuple_t *tupl;
	for ( /* iloop = 0 */ ; ; iloop++ ) {
		curr = wcube[iloop];
		if ( curr.x[0] == 0 && curr.x[1] == 0 && curr.x[2] == 0 ) break;

		printf ( "Cube size { %d, %d, %d } is available\n", curr.x[0], curr.x[1], curr.x[2]);
		tupl = fftx_psatd_Tuple ( wcube[iloop] );
		if ( tupl == NULL ) {
			printf ( "Failed to get tuple for cube { %d, %d, %d }\n", curr.x[0], curr.x[1], curr.x[2]);
		}
		else {
			M = curr.x[0], N = curr.x[1], K = curr.x[2];
			if ( debug_print ) printf ( "M = %d, N = %d, K = %d, malloc sizes = %d * sizeof(double)\n", M, N, K, M*N*K );
		
			FFTX_DEVICE_MALLOC ( &X, M*N*K * sizeof(double) );
			if ( debug_print ) printf ( "Allocated %lu bytes for device input array X\n", M*N*K*sizeof(double) );
			FFTX_DEVICE_MALLOC ( &Y, M*N*K * sizeof(double) );
			if ( debug_print ) printf ( "Allocated %lu bytes for device output array Y\n", M*N*K*sizeof(double) );

			double *host_X = new double[M*N*K];
			
			// FFTX_DEVICE_FFT_DOUBLECOMPLEX *cufft_Y; 
			// FFTX_DEVICE_MALLOC ( &cufft_Y, M*N*K * sizeof(FFTX_DEVICE_FFT_DOUBLECOMPLEX) );
			// if ( debug_print ) printf ( "Allocated %lu bytes for device cu/roc fft output array\n", M*N*K*sizeof(FFTX_DEVICE_FFT_DOUBLECOMPLEX) );

			// FFTX_DEVICE_FFT_HANDLE plan;
			// if ( debug_print ) printf ( "Create plan for cu/roc fft...\n" );
			// bool check_buff = true;
			// FFTX_DEVICE_FFT_RESULT res;
			// //  FFTX_DEVICE_FFT_CHECK ( res, "*** Exiting ... Create FFTX_DEVICE_FFT_PLAN3D" );
			// res = FFTX_DEVICE_FFT_PLAN3D ( &plan, M, N, K,  FFTX_DEVICE_FFT_Z2Z );
			// if (res != FFTX_DEVICE_FFT_SUCCESS ) {
			// 	printf ( "Create FFTX_DEVICE_FFT_PLAN3D failed with error code %d ... skip buffer check\n", res );
			// 	check_buff = false;
			// }

			//  Call the init function
			( * tupl->initfp )();
			FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR () );
			if ( debug_print ) printf ( "Spiral init function called: %s\n", FFTX_DEVICE_GET_ERROR_STRING ( FFTX_DEVICE_GET_LAST_ERROR () ) );
 
			// set up data in input buffer and run the transform
			buildInputBuffer(host_X, X);

			if ( debug_print ) printf ( "Loop the Spiral run function 100 times...\n" );
			for ( int kk = 0; kk < 100; kk++ ) {
				//  try the run function

				( * tupl->runfp ) ( Y, X, sym );
				FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR () );
			}
			
			// Tear down / cleanup
			( * tupl->destroyfp ) ();
			FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR () );
			if ( debug_print ) printf ( "Spiral destroy function called: %s\n", FFTX_DEVICE_GET_ERROR_STRING ( FFTX_DEVICE_GET_LAST_ERROR () ) );

			// if ( check_buff ) {
			// 	if ( FFTX_DEVICE_FFT_EXECZ2Z (
			// 			 plan,
			// 			 (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) X,
			// 			 (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) cufft_Y,
			// 			 FFTX_DEVICE_FFT_FORWARD
			// 			 ) != FFTX_DEVICE_FFT_SUCCESS) {
			// 		printf("FFTX_DEVICE_FFT_EXECZ2Z launch failed\n");
			// 		exit(-1);
			// 	}
			// }

			// FFTX_DEVICE_SYNCHRONIZE();
			// if ( FFTX_DEVICE_GET_LAST_ERROR () != FFTX_DEVICE_SUCCESS ) {
			// 	printf("FFTX_DEVICE_FFT_EXECZ2Z failed\n");
			// 	exit(-1);
			// }

			// //  check cufft/rocfft and FFTX got same results
			// if ( check_buff ) checkOutputBuffers ( Y, (double *)cufft_Y );
			
			FFTX_DEVICE_FREE ( X );
			FFTX_DEVICE_FREE ( Y );
			//  FFTX_DEVICE_FREE ( cufft_Y );
			delete[] host_X;

			if ( oneshot ) break;
		}
	}

}
