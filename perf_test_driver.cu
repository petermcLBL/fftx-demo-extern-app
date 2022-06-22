#include <stdio.h>

#include "fftx3.hpp"
#include "fftx_mddft_gpu_public.h"
#include "fftx_imddft_gpu_public.h"
#include "device_macros.h"

#include <stdlib.h>
#include <string.h>

#if defined(FFTX_HIP)
#define GPU_STR "rocfft"
#else
#define GPU_STR "cufft"
#endif

static int M, N, K;
static bool writefiles = false;

//  generate file name

static char * generateFileName ( const char *type )
{
	// type is: input ==> random input data; output ==> spiral output data; roc ==> rocFFT output data
	static char fileNameBuff[100];
	sprintf ( fileNameBuff, "mddft3d-%s-%dx%dx%d.dat", type, M, N, K );
	return fileNameBuff;
}

//  write data to file(s) for test repeatability.

static void writeBufferToFile ( const char *type, double *datap )
{
	char *fname = generateFileName ( type );
	FILE *fhandle = fopen ( fname, "w" );
	fprintf ( fhandle, "[ \n" );
	for ( int mm = 0; mm < M; mm++ ) {
		for ( int nn = 0; nn < N; nn++ ) {
			for ( int kk = 0; kk < K; kk++ ) {
				fprintf ( fhandle, "FloatString(\"%.12g\"), FloatString(\"%.12g\"), ", 
						  datap[(kk + nn*K + mm*N*K)*2 + 0], datap[(kk + nn*K + mm*N*K)*2 + 1] );
				if ( kk > 0 && kk % 8 == 0 )
					fprintf ( fhandle, "\n" );
			}
			fprintf ( fhandle, "\n" );
		}
	}
	fprintf ( fhandle, "];\n" );
	
	//  fwrite ( datap, sizeof(double) * 2, M * N * K, fhandle );
	fclose ( fhandle );
	return;
}

static void buildInputBuffer ( double *host_X, double *X, int genData )
{
	if ( genData ) {					// generate a new data input buffer
		for (int m = 0; m < M; m++) {
			for (int n = 0; n < N; n++) {
				for (int k = 0; k < K; k++) {
					host_X[(k + n*K + m*N*K)*2 + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
					host_X[(k + n*K + m*N*K)*2 + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
				}
			}
		}
	}

	DEVICE_MEM_COPY ( X, host_X, (M * N * K * 2 * sizeof(double)), MEM_COPY_HOST_TO_DEVICE);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
	return;
}

static void checkOutputBuffers ( double *Y, double *cufft_Y )
{
	printf("cube = [ %d, %d, %d ]\t", M, N, K);
	DEVICE_FFT_DOUBLECOMPLEX *host_Y       = new DEVICE_FFT_DOUBLECOMPLEX[M*N*K];
	DEVICE_FFT_DOUBLECOMPLEX *host_cufft_Y = new DEVICE_FFT_DOUBLECOMPLEX[M*N*K];

	DEVICE_MEM_COPY ( host_Y,             Y, (M * N * K * sizeof(DEVICE_FFT_DOUBLECOMPLEX)), MEM_COPY_DEVICE_TO_HOST );
	DEVICE_MEM_COPY ( host_cufft_Y, cufft_Y, (M * N * K * sizeof(DEVICE_FFT_DOUBLECOMPLEX)), MEM_COPY_DEVICE_TO_HOST );

	bool correct = true;
	//  int errCount = 0;
	double maxdelta = 0.0;

	for ( int m = 0; m < M; m++ ) {
		for ( int n = 0; n < N; n++ ) {
			for ( int k = 0; k < K; k++ ) {
				DEVICE_FFT_DOUBLECOMPLEX s = host_Y      [k + n*K + m*N*K];
				DEVICE_FFT_DOUBLECOMPLEX c = host_cufft_Y[k + n*K + m*N*K];
	    
				bool elem_correct = ( abs(s.x - c.x) < 1e-7 ) && ( abs(s.y - c.y) < 1e-7 );
				maxdelta = maxdelta < (double)(abs(s.x -c.x)) ? (double)(abs(s.x -c.x)) : maxdelta ;
				maxdelta = maxdelta < (double)(abs(s.y -c.y)) ? (double)(abs(s.y -c.y)) : maxdelta ;

				correct &= elem_correct;
				//  if (!elem_correct && errCount < 10) {
				//  	correct = false;
				//  	errCount++;
				//      printf("error at (%d,%d,%d): %f+%fi instead of %f+%fi\n", k, n, m, s.x, s.y, c.x, c.y);
				//  }
			}
		}
	}
	
	printf ( "Correct: %s\tMax delta = %E\t\t##PICKME## \n", (correct ? "True" : "False"), maxdelta );
	fflush ( stdout );

	if ( writefiles ) {
		writeBufferToFile ( (const char *)"spiral-out", (double *)host_Y );
		writeBufferToFile ( (const char *)GPU_STR,      (double *)host_cufft_Y );
	}
	delete[] host_Y;
	delete[] host_cufft_Y;

	return;
}


static int NUM_ITERS = 100;

int main( int argc, char** argv) {

	int iloop = 0;
	bool oneshot = false;
	int iters = NUM_ITERS + 10;

	//  Test is to time on a GPU [CUDA or HIP]
	printf ( "Usage: %s: [ iterations ] [ size: MMxNNxKK ] [ writefiles ]\n", argv[0] );
	if ( argc > 1 ) {
		NUM_ITERS = atoi ( argv[1] );
		iters = NUM_ITERS + 10;
		printf ( "%s: Measure %d iterations, ", argv[0], iters );
		
		if ( argc > 2 ) {
			char * foo = argv[2];
			M = atoi ( foo );
			while ( * foo != 'x' ) foo++;
			foo++ ;
			N = atoi ( foo );
			while ( * foo != 'x' ) foo++;
			foo++ ;
			K = atoi ( foo );
			oneshot = true;
			printf ( "Run size: %dx%dx%d, ", M, N, K );
			
			if ( argc > 3 ) {
				//  Only write files when a specified [single] size is used.  Write data to
				//  files -- spiral input data, spiral output data, and rocFFT/cuFFT output
				writefiles = true;
			}
			printf ( "%s data files\n", (writefiles) ? "WRITE" : "DO NOT write" );
		}
		else {
			printf ( "Run all sizes found in library, " );
			printf ( "%s data files\n", (writefiles) ? "WRITE" : "DO NOT write" );
		}
	}
	else {
		printf ( "%s: Measure %d iterations for all sizes found in the library\n", argv[0], iters );
	}
						  
	fftx::point_t<3> *wcube, curr;

	wcube = fftx_mddft_QuerySizes ();
	if (wcube == NULL) {
		printf ( "%s: Failed to get list of available sizes\n", argv[0] );
		exit (-1);
	}

	if ( oneshot ) {
		for ( iloop = 0; ; iloop++ ) {
			if ( wcube[iloop].x[0] == 0 && wcube[iloop].x[1] == 0 && wcube[iloop].x[2] == 0 ) {
				//  requested size is not in library, print message & exit
				printf ( "%s: Cube { %d, %d, %d } not found in library ... exiting\n", argv[0], M, N, K );
				exit (-1);
			}
			if ( wcube[iloop].x[0] == M && wcube[iloop].x[1] == N && wcube[iloop].x[2] == K ) {
				break;
			}
		}
	}

#if defined(FFTX_HIP)
    //  setup the library
	rocfft_setup();
#endif
	
	DEVICE_EVENT_T start, stop, custart, custop;
	DEVICE_EVENT_CREATE ( &start );
	DEVICE_EVENT_CREATE ( &stop );
	DEVICE_EVENT_CREATE ( &custart );
	DEVICE_EVENT_CREATE ( &custop );

	double *X, *Y;
	double sym[100];  // dummy symbol
	transformTuple_t *tupl;

	for ( /* iloop is initialized */ ; ; iloop++ ) {
		curr = wcube[iloop];
		if ( curr.x[0] == 0 && curr.x[1] == 0 && curr.x[2] == 0 ) break;

		printf ( "Cube size { %d, %d, %d } is available\n", curr.x[0], curr.x[1], curr.x[2]);
		tupl = fftx_mddft_Tuple ( wcube[iloop] );
		if ( tupl == NULL ) {
			printf ( "Failed to get tuple for cube { %d, %d, %d }\n", curr.x[0], curr.x[1], curr.x[2]);
		}
		else {
			M = curr.x[0], N = curr.x[1], K = curr.x[2];
			DEVICE_MALLOC ( &X, ( M * N * K * 2 * sizeof(double) ) );
			DEVICE_MALLOC ( &Y, ( M * N * K * 2 * sizeof(double) ) );

			double *host_X = new double[ M * N * K * 2 ];
			DEVICE_FFT_DOUBLECOMPLEX *cufft_Y; 
			DEVICE_MALLOC ( &cufft_Y, ( M * N * K * sizeof(DEVICE_FFT_DOUBLECOMPLEX) ) );

			//  want to run and time: 1st iteration; 2nd iteration; then N iterations
			//  Report 1st time, 2nd time, and average of N further iterations
			float *milliseconds   = new float[iters];
			float *cumilliseconds = new float[iters];
			bool check_buff = true;

			DEVICE_FFT_HANDLE plan;
			DEVICE_FFT_RESULT res;
			res = DEVICE_FFT_PLAN3D ( &plan, M, N, K, DEVICE_FFT_Z2Z );
			if ( res != DEVICE_FFT_SUCCESS ) {
				printf ( "Create DEVICE_FFT_PLAN3D failed with error code %d ... skip buffer check\n", res );
				check_buff = false;
			}

			//  Call the transform init function
			( * tupl->initfp )();
			DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );

			// set up data in input buffer
			buildInputBuffer(host_X, X, 1);
			if ( writefiles ) {
				printf ( "Write input buffer to a file..." );
				writeBufferToFile ( (const char *)"input", host_X );
				printf ( "done\n" );
			}

			for ( int ii = 0; ii < iters; ii++ ) {
				//  Call the main transform function
				DEVICE_EVENT_RECORD ( start );
				( * tupl->runfp ) ( Y, X, sym );
				DEVICE_EVENT_RECORD ( stop );
				DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
		
				DEVICE_EVENT_SYNCHRONIZE ( stop );
				DEVICE_EVENT_ELAPSED_TIME ( &milliseconds[ii], start, stop );

/* #ifdef USE_DIFF_DATA */
/* 				buildInputBuffer(host_X, X, 1); */
/* #else */
/* 				buildInputBuffer(host_X, X, 0); */
/* #endif */
			}

			//  Call the destroy function
			( * tupl->destroyfp )();
			DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );

			if ( check_buff ) {
				for ( int ii = 0; ii < iters; ii++ ) {
					DEVICE_EVENT_RECORD ( custart );
					res = DEVICE_FFT_EXECZ2Z ( plan,
											   (DEVICE_FFT_DOUBLECOMPLEX *) X,
											   (DEVICE_FFT_DOUBLECOMPLEX *) cufft_Y,
											   DEVICE_FFT_FORWARD );
					if ( res != DEVICE_FFT_SUCCESS) {
						printf ( "Launch DEVICE_FFT_EXECZ2Z failed with error code %d ... skip buffer check\n", res );
						check_buff = false;
						break;
					}
					DEVICE_EVENT_RECORD ( custop );
					DEVICE_EVENT_SYNCHRONIZE ( custop );
					DEVICE_EVENT_ELAPSED_TIME ( &cumilliseconds[ii], custart, custop );

/* #ifdef USE_DIFF_DATA */
/* 				buildInputBuffer(host_X, X, 1); */
/* #else */
/* 				buildInputBuffer(host_X, X, 0); */
/* #endif */
				}
			}
			DEVICE_SYNCHRONIZE ();

			//  check cufft/rocfft and FFTX got same results
			if ( check_buff ) checkOutputBuffers ( Y, (double *)cufft_Y );

			//  printf("cube = [ %d, %d, %d ]\t\t ##PICKME## \n", M, N, K);
			printf("%f\tms (SPIRAL) vs\t%f\tms (%s),\t\tFIRST iteration\t##PICKME## \n",
				   milliseconds[0], cumilliseconds[0], GPU_STR);
			printf("%f\tms (SPIRAL) vs\t%f\tms (%s),\t\tSECOND iteration\t##PICKME## \n",
				   milliseconds[1], cumilliseconds[1], GPU_STR);

			float cumulSpiral = 0.0, cumulHip = 0.0;
			for ( int ii = 10; ii < iters; ii++ ) {
				cumulSpiral += milliseconds[ii];
				cumulHip    += cumilliseconds[ii];
			} 
			printf("%f\tms (SPIRAL) vs\t%f\tms (%s), AVERAGE over %d iterations (range: 11 - %d) ##PICKME## \n",
				   cumulSpiral / NUM_ITERS, cumulHip / NUM_ITERS, GPU_STR, NUM_ITERS, (10 + NUM_ITERS) );

			DEVICE_FREE ( X );
			DEVICE_FREE ( Y );
			DEVICE_FREE ( cufft_Y );
			delete[] host_X;
			delete[] milliseconds;
			delete[] cumilliseconds;

			if ( oneshot ) break;
		}
	}

#if defined(FFTX_HIP)
	//  cleanup the library
	rocfft_cleanup();
#endif

}
