#include <stdio.h>

#include "fftxdevice_macros.h"

#include "fftx_mddft_public.h"
#include "fftx_imddft_public.h"
#include "fftx_mdprdft_public.h"
#include "fftx_imdprdft_public.h"
// #include "fftx_rconv_public.h"

#include "fftx3utilities.h"

enum TransformType { MDDFT, IMDDFT, MDPRDFT, IMDPRDFT };

static bool writefiles = false;

template<typename T>
T avgSubarray(const T* arr, int lo, int hi)
{
  T tot = 0.;
  int len = 0;
  for (int i = lo; i <= hi; i++)
    {
      tot += arr[i];
      len++;
    }
  T avg = tot / (len * 1.);
  return avg;
}

void setRand(double& a_val)
{
  a_val = 1. - ((double) rand()) / (double) (RAND_MAX/2);
}

void setRand(std::complex<double>& a_val)
{
  double x, y;
  setRand(x);
  setRand(y);
  a_val = std::complex<double>(x, y);
}

double diffAbs(double a_x,
               double a_y)
{
  double diffNorm = a_x - a_y;
  if (diffNorm < 0.) diffNorm = -diffNorm;
  return diffNorm;
}

double diffAbs(std::complex<double>& a_x,
               std::complex<double>& a_y)
{
  double diffNorm = std::abs(a_x - a_y);
  return diffNorm;
}

FFTX_DEVICE_FFT_RESULT deviceExecD2Z(FFTX_DEVICE_FFT_HANDLE a_plan,
                                double* a_in,
                                std::complex<double>* a_out)
{
  return FFTX_DEVICE_FFT_EXECD2Z(a_plan,
                            (FFTX_DEVICE_FFT_DOUBLEREAL*) a_in,
                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX*) a_out);
}


template<typename T_IN, typename T_OUT>
struct deviceTransform
{
  deviceTransform(FFTX_DEVICE_FFT_TYPE a_tp,
                  int a_dir = 0)
  {
    m_tp = a_tp;
    m_dir = a_dir;
  }
                  
  FFTX_DEVICE_FFT_TYPE m_tp;

  int m_dir;

  FFTX_DEVICE_FFT_RESULT plan3d(FFTX_DEVICE_FFT_HANDLE& a_plan,
                           fftx::point_t<3> a_tfmSize)
  {
    return FFTX_DEVICE_FFT_PLAN3D(&a_plan,
                             a_tfmSize[0], a_tfmSize[1], a_tfmSize[2],
                             m_tp);
  }

  FFTX_DEVICE_FFT_RESULT exec(FFTX_DEVICE_FFT_HANDLE a_plan,
                         T_IN* a_in,
                         T_OUT* a_out)
  {
    if (m_tp == FFTX_DEVICE_FFT_Z2Z)
      {
        return FFTX_DEVICE_FFT_EXECZ2Z(a_plan,
                                  (FFTX_DEVICE_FFT_DOUBLECOMPLEX*) a_in,
                                  (FFTX_DEVICE_FFT_DOUBLECOMPLEX*) a_out,
                                  m_dir);
      }
    else if (m_tp == FFTX_DEVICE_FFT_D2Z)
      {
        return FFTX_DEVICE_FFT_EXECD2Z(a_plan,
                                  (FFTX_DEVICE_FFT_DOUBLEREAL*) a_in,
                                  (FFTX_DEVICE_FFT_DOUBLECOMPLEX*) a_out);
      }
    else if (m_tp == FFTX_DEVICE_FFT_Z2D)
      {
        return FFTX_DEVICE_FFT_EXECZ2D(a_plan,
                                  (FFTX_DEVICE_FFT_DOUBLECOMPLEX*) a_in,
                                  (FFTX_DEVICE_FFT_DOUBLEREAL*) a_out);
      }
    else
      {
        return (FFTX_DEVICE_FFT_RESULT) -1;
      }
  }
};
  

deviceTransform<std::complex<double>, std::complex<double> >
mddftDevice(FFTX_DEVICE_FFT_Z2Z, FFTX_DEVICE_FFT_FORWARD);

deviceTransform<std::complex<double>, std::complex<double> >
imddftDevice(FFTX_DEVICE_FFT_Z2Z, FFTX_DEVICE_FFT_INVERSE);

deviceTransform<double, std::complex<double> >
mdprdftDevice(FFTX_DEVICE_FFT_D2Z);

deviceTransform<std::complex<double>, double>
imdprdftDevice(FFTX_DEVICE_FFT_Z2D);

template<typename T_IN, typename T_OUT>
void inoutSizes(fftx::point_t<3>& a_inSize,
                fftx::point_t<3>& a_outSize,
                const fftx::point_t<3>& a_fullSize,
                T_IN* a_inPtr,
                T_OUT* a_outPtr);

void inoutSizes(fftx::point_t<3>& a_inSize,
                fftx::point_t<3>& a_outSize,
                const fftx::point_t<3>& a_fullSize,
                std::complex<double>* a_inPtr,
                std::complex<double>* a_outPtr)
{
  a_inSize = a_fullSize;
  a_outSize = a_fullSize;
}


void inoutSizes(fftx::point_t<3>& a_inSize,
                fftx::point_t<3>& a_outSize,
                const fftx::point_t<3>& a_fullSize,
                double* a_inPtr,
                std::complex<double>* a_outPtr)
{
  a_inSize = a_fullSize;
  a_outSize = a_fullSize;
  // Halve the domain of the complex array.
#if FFTX_COMPLEX_TRUNC_LAST
  a_outSize[2] = a_outSize[2]/2 + 1;
#else
  a_outSize[0] = a_outSize[0]/2 + 1;
#endif
}


void inoutSizes(fftx::point_t<3>& a_inSize,
                fftx::point_t<3>& a_outSize,
                const fftx::point_t<3>& a_fullSize,
                std::complex<double>* a_inPtr,
                double* a_outPtr)
{
  a_inSize = a_fullSize;
  a_outSize = a_fullSize;
  // Halve the domain of the complex array.
#if FFTX_COMPLEX_TRUNC_LAST
  a_inSize[2] = a_inSize[2]/2 + 1;
#else
  a_inSize[0] = a_inSize[0]/2 + 1;
#endif
}


static int NUM_ITERS = 100;
static int BASE_ITERS = 10;

template<typename T_IN, typename T_OUT>
void compareSize(fftx::point_t<3> a_size,
                 transformTuple_t *a_tupl,
                 deviceTransform<T_IN, T_OUT>& a_tfmDevice)
{
  bool doDevice = true;
  bool doSpiral = true;

  if (a_tupl == NULL)
    {
      doSpiral = false;
      printf ( "Failed to get tuple for cube { %d, %d, %d }\n",
               a_size[0], a_size[1], a_size[2]);
    }

  /*
    Allocate space for arrays, and set input array.
  */
  fftx::point_t<3> inputSize, outputSize;
  T_IN* inPtr;
  T_OUT* outPtr;
  inoutSizes(inputSize, outputSize, a_size, inPtr, outPtr);
  

  // This doesn't work. :/
  // const fftx::point_t<3> unit = fftx::point_t<3>::Unit();
  //   fftx::box_t<3> inputDomain(unit, inputSize);
  //   fftx::box_t<3> outputDomain(unit, outputSize);

  fftx::box_t<3> inputDomain(fftx::point_t<3>({{1, 1, 1}}),
                             fftx::point_t<3>({{inputSize[0],
                                                inputSize[1],
                                                inputSize[2]}}));
  fftx::box_t<3> outputDomain(fftx::point_t<3>({{1, 1, 1}}),
                              fftx::point_t<3>({{outputSize[0],
                                                 outputSize[1],
                                                 outputSize[2]}}));
  
  fftx::array_t<3, T_IN> inputArrayHost(inputDomain);
  size_t nptsInput = inputDomain.size();
  size_t nptsOutput = outputDomain.size();
  size_t bytesInput = nptsInput * sizeof(T_IN);
  size_t bytesOutput = nptsOutput * sizeof(T_OUT);
  forall([](T_IN(&v), const fftx::point_t<3>& p)
         {
           setRand(v);
         }, inputArrayHost);
  // This symmetrizes only for complex input and real output,
  // in order to get a complex array that transforms to a real array.
  fftx::array_t<3, T_OUT> outputArrayHost(outputDomain);
  symmetrizeHermitian(inputArrayHost, outputArrayHost);

  T_IN* inputHostPtr = inputArrayHost.m_data.local();
  // additional code for GPU programs
  T_IN* inputDevicePtr;
  T_OUT* outputSpiralDevicePtr;
  T_OUT* outputDeviceFFTDevicePtr;
  FFTX_DEVICE_MALLOC(&inputDevicePtr, bytesInput);
  FFTX_DEVICE_MALLOC(&outputSpiralDevicePtr, bytesOutput);
  FFTX_DEVICE_MALLOC(&outputDeviceFFTDevicePtr, bytesOutput);
  // Do this at the beginning of each iteration instead of here.
  //  FFTX_DEVICE_MEM_COPY(inputDevicePtr, inputHostPtr, // dest, source
  //                  npts*sizeof(double), // bytes
  //                  FFTX_MEM_COPY_HOST_TO_DEVICE); // type
  
  /*
    Set up timers for deviceFFT.
   */
  FFTX_DEVICE_EVENT_T spiralFFT_start, spiralFFT_stop;
  FFTX_DEVICE_EVENT_T deviceFFT_start, deviceFFT_stop;
  FFTX_DEVICE_EVENT_CREATE ( &spiralFFT_start );
  FFTX_DEVICE_EVENT_CREATE ( &spiralFFT_stop );
  FFTX_DEVICE_EVENT_CREATE ( &deviceFFT_start );
  FFTX_DEVICE_EVENT_CREATE ( &deviceFFT_stop );

  int iters = NUM_ITERS + BASE_ITERS;

  /*
    Get plan for deviceFFT.
  */
  // printf("get deviceFFT plan\n");
  FFTX_DEVICE_FFT_HANDLE plan;
  {
    auto rc = a_tfmDevice.plan3d(plan, a_size);
    if (rc != FFTX_DEVICE_FFT_SUCCESS)
      {
        printf ( "Create FFTX_DEVICE_FFT_PLAN3D failed with error code %d ... skip buffer check\n",
                 rc );
        doDevice = false;
      }
  }

  /*
    Time iterations of real-to-complex deviceFFT calls using the plan.
   */
  // printf("call deviceExec %d times\n", a_iterations);

  float* deviceFFT_gpu = new float[iters];
  for (int i = 0; i < iters; i++)
    {
      deviceFFT_gpu[i] = 0.;
    }
  if (doDevice)
    {
      for (int itn = 0; itn < iters; itn++ )
        {
          FFTX_DEVICE_MEM_COPY(inputDevicePtr, // dest
                          inputHostPtr, // source
                          bytesInput, // bytes
                          FFTX_MEM_COPY_HOST_TO_DEVICE); // type
          FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR() );
          FFTX_DEVICE_EVENT_RECORD( deviceFFT_start );
          int rc = a_tfmDevice.exec(plan,
                                    inputDevicePtr,
                                    outputDeviceFFTDevicePtr);
          if (rc != FFTX_DEVICE_FFT_SUCCESS)
            {
              printf ( "Launch device exec failed with error code %d ... skip buffer check\n",
                       rc );
              doDevice = false;
              break;
            }
          FFTX_DEVICE_EVENT_RECORD( deviceFFT_stop );
          FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR() );
          FFTX_DEVICE_EVENT_SYNCHRONIZE( deviceFFT_stop );
          FFTX_DEVICE_EVENT_ELAPSED_TIME( &deviceFFT_gpu[itn],
                                     deviceFFT_start,
                                     deviceFFT_stop );
        }
    }
  FFTX_DEVICE_FFT_DESTROY(plan);

  FFTX_DEVICE_SYNCHRONIZE();

  // printf("call Spiral transform %d times\n", iters);

  /*
    Time iterations of transform with SPIRAL-generated code.
   */
  float* spiral_gpu = new float[iters];
  for (int i = 0; i < iters; i++)
    {
      spiral_gpu[i] = 0.;
    }

  FFTX_DEVICE_MEM_COPY(inputDevicePtr, // dest
                  inputHostPtr, // source
                  bytesInput, // bytes
                  FFTX_MEM_COPY_HOST_TO_DEVICE); // type
  FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR() );

  if (doSpiral)
    {
      double sym[100];  // dummy symbol
      ( * a_tupl->initfp )();
      FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR () );

      for (int itn = 0; itn < iters; itn++)
        {
          FFTX_DEVICE_EVENT_RECORD( spiralFFT_start );
          ( * a_tupl->runfp ) ( (double*) outputSpiralDevicePtr,
                                (double*) inputDevicePtr,
                                sym );
          FFTX_DEVICE_EVENT_RECORD( spiralFFT_stop );
          FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR () );
          FFTX_DEVICE_EVENT_SYNCHRONIZE ( spiralFFT_stop );
          FFTX_DEVICE_EVENT_ELAPSED_TIME ( &spiral_gpu[itn],
                                      spiralFFT_start,
                                      spiralFFT_stop );
        }

      //  Call the destroy function
      ( * a_tupl->destroyfp )();
      FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR () );
    }
  
  /*
    Check that deviceFFT and SPIRAL give the same results on last iteration.
  */
  T_OUT* outputSpiralHostPtr = new T_OUT[nptsOutput];
  T_OUT* outputDeviceFFTHostPtr = new T_OUT[nptsOutput];
  FFTX_DEVICE_MEM_COPY(outputSpiralHostPtr, // dest
                  outputSpiralDevicePtr, // source
                  bytesOutput, // bytes
                  FFTX_MEM_COPY_DEVICE_TO_HOST); // type
  FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR() );
  FFTX_DEVICE_MEM_COPY(outputDeviceFFTHostPtr, // dest
                  outputDeviceFFTDevicePtr, // source
                  bytesOutput, // bytes
                  FFTX_MEM_COPY_DEVICE_TO_HOST); // type
  FFTX_DEVICE_CHECK_ERROR ( FFTX_DEVICE_GET_LAST_ERROR() );

  FFTX_DEVICE_FREE(inputDevicePtr);
  FFTX_DEVICE_FREE(outputSpiralDevicePtr);
  FFTX_DEVICE_FREE(outputDeviceFFTDevicePtr);

  printf("cube = [ %d, %d, %d ]\t", a_size[0], a_size[1], a_size[2]);
  if (doSpiral && doDevice)
    {
      bool correct = true;
      const double tol = 1.e-7;
      double maxdelta = 0.;
      for (size_t ind = 0; ind < nptsOutput; ind++)
        {
          T_OUT outputSpiralPoint = outputSpiralHostPtr[ind];
          T_OUT outputDeviceFFTPoint = outputDeviceFFTHostPtr[ind];
          // auto diffPoint = outputSpiralPoint - outputDeviceFFTPoint;
          // double diffReal = outputSpiralPoint.x - outputDeviceFFTPoint.x;
          // double diffImag = outputSpiralPoint.y - outputDeviceFFTPoint.y;
          double diffAbsPoint = diffAbs(outputSpiralPoint, outputDeviceFFTPoint);
          updateMaxAbs(maxdelta, diffAbsPoint);
          bool correctPoint = (diffAbsPoint < tol);
          if (!correctPoint)
            {
              correct = false;
            }
        }
      printf( "Correct: %s\tMax delta = %E\t\t##PICKME## \n",
              (correct ? "True" : "False"), maxdelta );
    }
  else
    {
      printf( "Correct: Could not compare.\t\t##PICKME## \n" );
    }

  // FIXME: writeBufferToFile
  
  delete[] outputSpiralHostPtr;
  delete[] outputDeviceFFTHostPtr;

  printf("%f\tms (SPIRAL) vs\t%f\tms (hipfft),\t\tFIRST iteration\t##PICKME## \n",
         spiral_gpu[0], deviceFFT_gpu[0]);
  printf("%f\tms (SPIRAL) vs\t%f\tms (hipfft),\t\tSECOND iteration\t##PICKME## \n",
         spiral_gpu[1], deviceFFT_gpu[1]);

  float avgSpiral = avgSubarray(spiral_gpu, BASE_ITERS, iters-1);
  float avgDevice = avgSubarray(deviceFFT_gpu, BASE_ITERS, iters-1);
  delete[] spiral_gpu;
  delete[] deviceFFT_gpu;

  printf("%f\tms (SPIRAL) vs\t%f\tms (hipfft), AVERAGE over %d iterations (range: 11 - %d) ##PICKME## \n",
         avgSpiral, avgDevice, NUM_ITERS, BASE_ITERS + NUM_ITERS );
}


int main(int argc, char* argv[])
{
  int iloop = 0;
  bool oneshot = false;
  TransformType ttype;
  int M, N, K;

  printf("Usage:  %s mddft|imddft|mdprdft|imdprdft [iterations=20] [MxNxK] [writefiles]\n",
         argv[0]);

  if (argc > 1)
    {
      int libmode;
      if (std::string(argv[1]) == "mddft")
        {
          ttype = MDDFT;
          libmode = fftx_mddft_GetLibraryMode();
        }
      else if (std::string(argv[1]) == "imddft")
        {
          ttype = IMDDFT;
          libmode = fftx_imddft_GetLibraryMode();
        }
      else if (std::string(argv[1]) == "mdprdft")
        {
          ttype = MDPRDFT;
          libmode = fftx_mdprdft_GetLibraryMode();
        }
      else if (std::string(argv[1]) == "imdprdft")
        {
          ttype = IMDPRDFT;
          libmode = fftx_imdprdft_GetLibraryMode();
        }
      else
        {
          printf("%s: failed to specify one of mddft|imddft|mdprdft|imdprdft\n",
                 argv[0]);
          exit(-1);
        }
      if ( (libmode != LIB_MODE_CUDA ) &&
           (libmode != LIB_MODE_HIP ) )
        { // Test is to time on a GPU [CUDA or HIP];
          // check library support this mode
          printf ( "%s: fftx_%s library doesn't support GPU, exiting...\n",
                   argv[0], argv[1] );
          exit (-1);
        }

      if (argc > 2)
        {
          NUM_ITERS = atoi ( argv[2] );

          if (argc > 3)
            { // Run size specified in form MxNxK.
              char * foo = argv[3];
              M = atoi ( foo );
              while ( * foo != 'x' ) foo++;
              foo++ ;
              N = atoi ( foo );
              while ( * foo != 'x' ) foo++;
              foo++ ;
              K = atoi ( foo );
              oneshot = true;
              printf ( "Run size: %dx%dx%d, ", M, N, K );

              if (argc > 4)
                { // Only write files when a specified [single] size is used.
                  // Write data to files:
                  // spiral input data, spiral output data, rocFFT/cuFFT output.
                  writefiles = true;
                  printf("WRITE data files\n");
                }
              else
                {
                  printf("DO NOT write data files\n");
                }
            }
          else
            {
              printf ( "Run all sizes found in library, DO NOT write data files\n" );
            }
        }
    }
  else
    {
      printf("%s: failed to specify one of mddft|imddft|mdprdft|imdprdft",
             argv[0]);
      exit(-1);
    }

  int iters = NUM_ITERS + BASE_ITERS;
  printf ( "%s %s: Measure %d iterations\n",
           argv[0], argv[1], iters );

  fftx::point_t<3> *wcube, curr;

  // last entry is { 0, 0, 0 }
  if (ttype == MDDFT)
    {
      wcube = fftx_mddft_QuerySizes ();
    }
  else if (ttype == IMDDFT)
    {
      wcube = fftx_imddft_QuerySizes ();
    }
  else if (ttype == MDPRDFT)
    {
      wcube = fftx_mdprdft_QuerySizes ();
    }
  else if (ttype == IMDPRDFT)
    {
      wcube = fftx_imdprdft_QuerySizes ();
    }
  if (wcube == NULL)
    {
      printf ( "%s %s: Failed to get list of available sizes\n",
               argv[0], argv[1] );
      exit (-1);
    }

  if ( oneshot )
    {
      for ( iloop = 0; ; iloop++ )
        {
          if ( wcube[iloop].x[0] == 0 &&
               wcube[iloop].x[1] == 0 &&
               wcube[iloop].x[2] == 0 )
            {
              //  requested size is not in library, print message & exit
              printf ( "%s %s: Cube { %d, %d, %d } not found in library ... exiting\n",
                       argv[0], argv[1], M, N, K );
              exit (-1);
            }
          if ( wcube[iloop].x[0] == M &&
               wcube[iloop].x[1] == N &&
               wcube[iloop].x[2] == K )
            {
              break;
            }
        }
      // Now wcube[iloop] is set to [M, N, K].
    }

#if defined(FFTX_HIP)
  //  Set up the library.
  rocfft_setup();
#endif

  double *X, *Y;
  // double sym[100];  // dummy symbol
  // transformTuple_t *tupl;

  for ( /* iloop is initialized */ ; ; iloop++ )
    {
      curr = wcube[iloop];
      if ( curr[0] == 0 &&
           curr[1] == 0 &&
           curr[2] == 0 )
        { // This is the end.
          break;
        }

      // M = curr.x[0];
      // N = curr.x[1];
      // K = curr.x[2];
      printf ( "Cube size { %d, %d, %d } is available\n",
               curr[0], curr[1], curr[2] );
      // If tupl == NULL then compareSize function will catch it.
      if (ttype == MDDFT)
        {
          transformTuple_t *tupl = fftx_mddft_Tuple ( curr );
          compareSize(curr, tupl, mddftDevice);
        }
      else if (ttype == IMDDFT)
        {
          transformTuple_t *tupl = fftx_imddft_Tuple ( curr );
          compareSize(curr, tupl, imddftDevice);
        }
      else if (ttype == MDPRDFT)
        {
          transformTuple_t *tupl = fftx_mdprdft_Tuple ( curr );
          compareSize(curr, tupl, mdprdftDevice);
        }
      else if (ttype == IMDPRDFT)
        {
          transformTuple_t *tupl = fftx_imdprdft_Tuple ( curr );
          compareSize(curr, tupl, imdprdftDevice);
        }

      if (oneshot)
        {
          break;
        }
    }

#if defined(FFTX_HIP)
  //  cleanup the library
  rocfft_cleanup();
#endif
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
