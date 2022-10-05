#include <stdio.h>

#include "fftx3.hpp"
#include "fftx_mddft_gpu_public.h"
#include "fftx_imddft_gpu_public.h"
#include "fftx_mdprdft_gpu_public.h"
#include "fftx_imdprdft_gpu_public.h"
#include "device_macros.h"

#include <stdlib.h>
#include <string.h>
#include <strings.h>

#if defined(FFTX_HIP)
#define GPU_STR "rocfft"
#else
#define GPU_STR "cufft"
#endif

static int M, N, K, K_adj;

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

static void buildInputBuffer ( double *host_X, double *X, bool genData, bool genComplex, bool useFullK )
{
	int KK = ( useFullK ) ? K : K_adj;
	
	if ( genData ) {					// generate a new data input buffer
		for (int m = 0; m < M; m++) {
			for (int n = 0; n < N; n++) {
				for (int k = 0; k < KK; k++) {
					if ( genComplex ) {
						host_X[(k + n*KK + m*N*KK)*2 + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
						host_X[(k + n*KK + m*N*KK)*2 + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
					}
					else {
						host_X[(k + n*KK + m*N*KK)] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
					}
				}
			}
		}
	}

	unsigned int nbytes = M * N * KK * sizeof(double);
	if ( genComplex ) nbytes *= 2;
	DEVICE_MEM_COPY ( X, host_X, nbytes, MEM_COPY_HOST_TO_DEVICE);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
	return;
}


static bool writefiles = false;

static void checkOutputBuffers ( double *Y, double *cufft_Y, bool isR2C, bool xfmdir )
{
	int datasz, KK = K;
	bool compCplx = true;				//  compare complex buffers

	if ( isR2C ) {
		//  real (double) to complex
		if ( xfmdir ) {
			//  output buffers are complex, dims M * N * K_adj
			datasz = M * N * K_adj * 2;
			KK = K_adj;
		}
		else {
			//  out buffers are real (double), dims M * N * K
			datasz = M * N * K;
			compCplx = false;
		}
	}
	else {
		//  complex to complex, dims M * N * K
		datasz = M * N * K * 2;
	}

	printf ( "cube = [ %d, %d, %d ]\t%s\t%s\t", M, N, K,
			 ( isR2C ) ? "MDPRDFT" : "MDDFT",
			 ( xfmdir ) ? "(Forward)" : "(Inverse)" );

	double *tmp_Y       = new double [ datasz ];
	double *tmp_cufft_Y = new double [ datasz ];
	DEVICE_MEM_COPY ( tmp_Y,             Y, datasz * sizeof(double), MEM_COPY_DEVICE_TO_HOST );
	DEVICE_MEM_COPY ( tmp_cufft_Y, cufft_Y, datasz * sizeof(double), MEM_COPY_DEVICE_TO_HOST );

	bool correct = true;
	double maxdelta = 0.0;

	for ( int m = 0; m < M; m++ ) {
		for ( int n = 0; n < N; n++ ) {
			for ( int k = 0; k < KK; k++ ) {
				if ( compCplx ) {
					DEVICE_FFT_DOUBLECOMPLEX *host_Y       = (DEVICE_FFT_DOUBLECOMPLEX *) tmp_Y;
					DEVICE_FFT_DOUBLECOMPLEX *host_cufft_Y = (DEVICE_FFT_DOUBLECOMPLEX *) tmp_cufft_Y;
					DEVICE_FFT_DOUBLECOMPLEX s = host_Y      [k + n*KK + m*N*KK];
					DEVICE_FFT_DOUBLECOMPLEX c = host_cufft_Y[k + n*KK + m*N*KK];

					bool elem_correct = ( (abs(s.x - c.x) < 1e-7) && (abs(s.y - c.y) < 1e-7) );
					maxdelta = maxdelta < (double)(abs(s.x -c.x)) ? (double)(abs(s.x -c.x)) : maxdelta ;
					maxdelta = maxdelta < (double)(abs(s.y -c.y)) ? (double)(abs(s.y -c.y)) : maxdelta ;
					correct &= elem_correct;
				}
				else {
					double *host_Y = tmp_Y, *host_cufft_Y = tmp_cufft_Y;
					double deltar = abs ( host_Y[(k + n*KK + m*N*KK)] - host_cufft_Y[(k + n*KK + m*N*KK)] );
					bool   elem_correct = ( deltar < 1e-7 );
					maxdelta = maxdelta < deltar ? deltar : maxdelta ;
					correct &= elem_correct;
				}
			}
		}
	}
	
	printf ( "Correct: %s\tMax delta = %E\t\t##PICKME##\n", (correct ? "True" : "False"), maxdelta );
	fflush ( stdout );

	if ( writefiles ) {
		writeBufferToFile ( (const char *)"spiral-out", (double *)tmp_Y );
		writeBufferToFile ( (const char *)GPU_STR,      (double *)tmp_cufft_Y );
	}
	delete[] tmp_Y;
	delete[] tmp_cufft_Y;

	return;
}


static int NUM_ITERS = 100;
static transformTuple_t *mdprdft_fwd;

static void	run_transform ( fftx::point_t<3> curr, transformTuple_t *tupl, bool isR2C, bool xfmdir )
{
	DEVICE_EVENT_T start, stop, custart, custop;
	DEVICE_EVENT_CREATE ( &start );
	DEVICE_EVENT_CREATE ( &stop );
	DEVICE_EVENT_CREATE ( &custart );
	DEVICE_EVENT_CREATE ( &custop );

	double *X, *Y;
	double sym[100];  // dummy symbol
	int iters = NUM_ITERS + 10;
	
	M = curr.x[0], N = curr.x[1], K = curr.x[2];
	K_adj = (int) ( K / 2 ) + 1;
	double *host_X;
	DEVICE_FFT_DOUBLEREAL *cufft_Y; 

	if ( isR2C && xfmdir ) {
		//  When is real-2-complex and xfmdir (i.e., forward) input is real (double) of dims M * N * K
		//  and the output array is (complex) of dims M * N * (K/2) + 1)
		DEVICE_MALLOC ( &X,       ( M * N * K     * sizeof(DEVICE_FFT_DOUBLEREAL) ) );
		DEVICE_MALLOC ( &Y,       ( M * N * K_adj * sizeof(DEVICE_FFT_DOUBLECOMPLEX) ) );
		DEVICE_MALLOC ( &cufft_Y, ( M * N * K_adj * sizeof(DEVICE_FFT_DOUBLECOMPLEX) ) );
		host_X = new double[ M * N * K ];
	}
	else if ( isR2C && !xfmdir ) {
		//  When is real-2-complex and !xfmdir (i.e., inverse) input is complex of dims M * N * (K/2) + 1)
		//  and the output array is (double) of dims M * N * K
		DEVICE_MALLOC ( &X,       ( M * N * K_adj * sizeof(DEVICE_FFT_DOUBLECOMPLEX) ) );
		DEVICE_MALLOC ( &Y,       ( M * N * K     * sizeof(DEVICE_FFT_DOUBLEREAL) ) );
		DEVICE_MALLOC ( &cufft_Y, ( M * N * K     * sizeof(DEVICE_FFT_DOUBLEREAL) ) );
		host_X = new double[ M * N * K_adj * 2];
	}
	else {
		// complex-2-complex: input and output are complex of dims M * N * K
		DEVICE_MALLOC ( &X,       ( M * N * K * sizeof(DEVICE_FFT_DOUBLECOMPLEX) ) );
		DEVICE_MALLOC ( &Y,       ( M * N * K * sizeof(DEVICE_FFT_DOUBLECOMPLEX) ) );
		DEVICE_MALLOC ( &cufft_Y, ( M * N * K * sizeof(DEVICE_FFT_DOUBLECOMPLEX) ) );
		host_X = new double[ M * N * K * 2];
	}

	//  want to run and time: 1st iteration; 2nd iteration; then N iterations
	//  Report 1st time, 2nd time, and average of N further iterations
	float *milliseconds   = new float[iters];
	float *cumilliseconds = new float[iters];
	bool check_buff = true;

	DEVICE_FFT_HANDLE plan;
	DEVICE_FFT_RESULT res;
	DEVICE_FFT_TYPE   xfmtype = ( !isR2C ) ? DEVICE_FFT_Z2Z : ( xfmdir ) ? DEVICE_FFT_D2Z : DEVICE_FFT_Z2D ;
	
	res = DEVICE_FFT_PLAN3D ( &plan, M, N, K, xfmtype );
	if ( res != DEVICE_FFT_SUCCESS ) {
		printf ( "Create DEVICE_FFT_PLAN3D failed with error code %d ... skip buffer check\n", res );
		check_buff = false;
	}

	// set up data in input buffer: gen data = true,
	// gen complex = true if !isR2C or (isR2C and inverse direction); false otherwise
	// use full K dim = false when (isR2C and inverse direction); true otherwise

	if ( isR2C && !xfmdir ) {
		//  real FFT, inverse direction.  To guarantee hermitian symmetry (required for
		//  rocfft output to be valid) we want to get the output from a forward real FFT
		//  and use that as input.  For inverse fft we've allocated the output buffer Y at
		//  the size needed for a real input array which can be used to generate the
		//  complex output we'll use as input, so flip X & Y in the run transform call below.
		//  The tupl pointing to the forward trnsform is stored in mdprdft_fwd
		
		double *herm_X;
		herm_X = new double[ M * N * K ];
		( * mdprdft_fwd->initfp )();
		buildInputBuffer ( herm_X, Y, true /* generate data */, false /* gen complex data */, true /* full K dim */ );

		( * mdprdft_fwd->runfp ) ( X, Y, sym );
		DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
		DEVICE_MEM_COPY ( host_X, X, (  M * N * K_adj * 2 ) * sizeof(double), MEM_COPY_DEVICE_TO_HOST );
		( * mdprdft_fwd->destroyfp )();
		DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );

		// normalize the data returned 
		for ( int ii = 0; ii < M * N * K_adj * 2; ii++ )
			host_X[ii] /= ( M * N * K_adj );

		( * tupl->initfp )();
		DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
		buildInputBuffer ( host_X, X, false, ( !isR2C || ( isR2C && !xfmdir ) ), ! ( isR2C && !xfmdir ) );
		delete[] herm_X;
	}
	else {
		( * tupl->initfp )();
		DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
		buildInputBuffer ( host_X, X, true,
						   ( !isR2C || ( isR2C && !xfmdir ) ),
						   ! ( isR2C && !xfmdir ) );
	}
	/* if ( writefiles ) { */
	/* 	//  Currently, only for MDDFT, i.e., complex to complex */
	/* 	printf ( "Write input buffer to a file..." ); */
	/* 	writeBufferToFile ( (const char *)"input", host_X ); */
	/* 	printf ( "done\n" ); */
	/* } */

	for ( int ii = 0; ii < iters; ii++ ) {
		//  Call the main transform function
		DEVICE_EVENT_RECORD ( start );
		( * tupl->runfp ) ( Y, X, sym );
		DEVICE_EVENT_RECORD ( stop );
		DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );

		DEVICE_EVENT_SYNCHRONIZE ( stop );
		DEVICE_EVENT_ELAPSED_TIME ( &milliseconds[ii], start, stop );

/* #ifdef USE_DIFF_DATA */
/* 		buildInputBuffer ( host_X, X, true, */
/* 						   ( !isR2C || ( isR2C && !xfmdir ) ), */
/* 						   ! ( isR2C && !xfmdir ) ); */
/* #else */
/* 		buildInputBuffer ( host_X, X, false, */
/* 						   ( !isR2C || ( isR2C && !xfmdir ) ), */
/* 						   ! ( isR2C && !xfmdir ) ); */
/* #endif */
	}

	//  Call the destroy function
	( * tupl->destroyfp )();
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );

	if ( check_buff ) {
		for ( int ii = 0; ii < iters; ii++ ) {
			DEVICE_EVENT_RECORD ( custart );
			if ( !isR2C ) {
				res = DEVICE_FFT_EXECZ2Z ( plan,
										   (DEVICE_FFT_DOUBLECOMPLEX *) X,
										   (DEVICE_FFT_DOUBLECOMPLEX *) cufft_Y,
										   ( xfmdir ) ? DEVICE_FFT_FORWARD : DEVICE_FFT_INVERSE );
			}
			else {
				if ( xfmdir )
					res = DEVICE_FFT_EXECD2Z ( plan,
											   (DEVICE_FFT_DOUBLEREAL *) X,
											   (DEVICE_FFT_DOUBLECOMPLEX *) cufft_Y );
				else
					res = DEVICE_FFT_EXECZ2D ( plan,
											   (DEVICE_FFT_DOUBLECOMPLEX *) X,
											   (DEVICE_FFT_DOUBLEREAL *) cufft_Y );
			}
			if ( res != DEVICE_FFT_SUCCESS) {
				printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
				check_buff = false;
				break;
			}
			DEVICE_EVENT_RECORD ( custop );
			DEVICE_EVENT_SYNCHRONIZE ( custop );
			DEVICE_EVENT_ELAPSED_TIME ( &cumilliseconds[ii], custart, custop );

			//  if ( isR2C && !xfmdir ) {
			if ( isR2C  ) {
				//  Input buffer is over-written / corrupted when doing IMDPRDFT (CUDA); both fwd & inv (HIP) 
#ifdef USE_DIFF_DATA
				buildInputBuffer ( host_X, X, true,
								   ( !isR2C || ( isR2C && !xfmdir ) ),
								   ! ( isR2C && !xfmdir ) );
#else
				buildInputBuffer ( host_X, X, false,
								   ( !isR2C || ( isR2C && !xfmdir ) ),
								   ! ( isR2C && !xfmdir ) );
#endif
			}
		}
	}
	DEVICE_SYNCHRONIZE ();

	//  check cufft/rocfft and FFTX got same results
	if ( check_buff ) checkOutputBuffers ( Y, (double *)cufft_Y, isR2C, xfmdir );
	
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

	return;
}


int main( int argc, char** argv)
{
	//  Test is to time on a GPU [CUDA or HIP]
	bool oneshot = false;
	bool runmddft = true, runimddft = true, runmdprdft = true, runimdprdft = true;
	char *prog = argv[0];

	int baz = 0;
	while ( argc > 1 && argv[1][0] == '-' ) {
		switch ( argv[1][1] ) {
		case 'i':
			argv++, argc--;
			NUM_ITERS = atoi ( argv[1] );
			break;

		case 'w': 					//  write files option
			writefiles = true;
			break;

		case 's':
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

		case 't':
			argv++, argc--;
			runmddft = false, runimddft = false, runmdprdft = false, runimdprdft = false;
			if ( strcasecmp ( argv[1], "mddft" ) == 0 )    runmddft    = true;
			if ( strcasecmp ( argv[1], "imddft" ) == 0 )   runimddft   = true;
			if ( strcasecmp ( argv[1], "mdprdft" ) == 0 )  runmdprdft  = true;
			if ( strcasecmp ( argv[1], "imdprdft" ) == 0 ) runimdprdft = true;
			break;

		case 'h':
			printf ( "Usage: %s: [ -i iterations ] [ -s MMxNNxKK ] [ -w[ritefiles] ] [ -t transform ]\n", argv[0] );
			printf ( "One -t option is permitted (name is case insensitive): -t { mddft | imddft | mdprdft | imdprdft\n" );
			printf ( "Writefiles is only permitted when a single size is specified\n" ); 
			exit (0);

		default:
			printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
		}
		argv++, argc--;
	}

	int iloop = 0;
	int iters = NUM_ITERS + 10;

	char buf[100];
	if ( oneshot ) sprintf ( buf, "for size: %dx%dx%d, ", M, N, K );
	if ( !oneshot ) writefiles = false;
	printf ( "%s: Measure %d iterations, %s", prog, iters, (oneshot) ? buf : "for all sizes, " );
	if ( runmddft && runimddft && runmdprdft && runimdprdft ) {
		sprintf ( buf, "for all transforms, " );
	}
	else {
		if ( runmddft ) sprintf ( buf, "for transform: MDDFT, " );
		if ( runimddft ) sprintf ( buf, "for transform: IMDDFT, " );
		if ( runmdprdft ) sprintf ( buf, "for transform: MDPRDFT, " );
		if ( runimdprdft ) sprintf ( buf, "for transform: IMDPRDFT, " );
	}
	printf ( "%s%s data files\n", buf, (writefiles) ? "WRITE" : "DO NOT write" );
	
	fftx::point_t<3> *wcube, curr;

	wcube = fftx_mddft_QuerySizes ();
	if (wcube == NULL) {
		printf ( "%s: Failed to get list of available sizes from MDDFT library\n", prog );
		exit (-1);
	}

	if ( oneshot ) {
		for ( iloop = 0; ; iloop++ ) {
			if ( wcube[iloop].x[0] == 0 && wcube[iloop].x[1] == 0 && wcube[iloop].x[2] == 0 ) {
				//  requested size is not in library, print message & exit
				printf ( "%s: Cube { %d, %d, %d } not found in library ... exiting\n", prog, M, N, K );
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

	transformTuple_t *tupl, *tupli;
	bool isR2C;

	for ( /* iloop is initialized */ ; ; iloop++ ) {
		curr = wcube[iloop];
		if ( curr.x[0] == 0 && curr.x[1] == 0 && curr.x[2] == 0 ) break;

		printf ( "Cube size { %d, %d, %d } is available\n", curr.x[0], curr.x[1], curr.x[2]);
		tupl  = fftx_mddft_Tuple ( wcube[iloop] );
		tupli = fftx_imddft_Tuple ( wcube[iloop] );
		if ( tupl == NULL || tupli == NULL ) {
			printf ( "Failed to get tuples (MDDFT/IMDDFT) for cube { %d, %d, %d }\n", curr.x[0], curr.x[1], curr.x[2]);
			continue;
		}

		isR2C = false;			//  do complex-2-complex first
		if ( runmddft )  run_transform ( curr, tupl, isR2C, true );
		if ( runimddft ) run_transform ( curr, tupli, isR2C, false );
		
		tupl  = fftx_mdprdft_Tuple ( wcube[iloop] );
		tupli = fftx_imdprdft_Tuple ( wcube[iloop] );
		if ( tupl == NULL || tupli == NULL ) {
			printf ( "Failed to get tuples (MDPRDFT/IMDPRDFT) for cube { %d, %d, %d }\n", curr.x[0], curr.x[1], curr.x[2]);
			continue;
		}

		isR2C = true;			//  do R2C & C2R
		if ( runmdprdft )  run_transform ( curr, tupl, isR2C, true );
		mdprdft_fwd = tupl;
		if ( runimdprdft ) run_transform ( curr, tupli, isR2C, false );
		
		if ( oneshot ) break;
	}

#if defined(FFTX_HIP)
	//  cleanup the library
	rocfft_cleanup();
#endif

}
