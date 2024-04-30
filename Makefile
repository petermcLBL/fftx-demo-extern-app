#
# FFTX
#
FFTX_INCLUDE_DIR=$(FFTX_HOME)/include
FFTX_MPI_INCLUDE_DIR=$(FFTX_HOME)/src/library/lib_fftx_mpi
FFTX_INCLUDE=-I$(FFTX_INCLUDE_DIR) -I$(FFTX_MPI_INCLUDE_DIR)

FFTX_LIB_DIR=$(FFTX_HOME)/lib
# The FFTX libraries (for either GPU or CPU) go after $(FFTX_LINK).
# Apple needs a space after rpath, while other platforms need an equal sign.
ifeq ($(VENDOR),apple)
  FFTX_LINK=-Wl,-rpath $(FFTX_LIB_DIR) -L$(FFTX_LIB_DIR)
else
  ifndef ROCM_PATH
    # This flag gets rid of annoying "DSO missing from command line" error.
    LDFLAGS=-Wl,--copy-dt-needed-entries
  endif
  FFTX_LINK=-Wl,-rpath=$(FFTX_LIB_DIR) -L$(FFTX_LIB_DIR)
endif

ifdef CUDATOOLKIT_HOME
  default: CUDA
else ifdef ROCM_PATH
  default: HIP
else
  default: CPU
endif

#
## make CUDA: needs CUDATOOLKIT_HOME
#
CUDA: CC=nvcc
CUDA: CCFLAGS=-x cu
CUDA: PRESETS=-DFFTX_CUDA
# To get helper_cuda.h
CUDA: CC_INCLUDE=-I$(CUDATOOLKIT_HOME)/../../examples/OpenMP/SDK/include
CUDA: CC_LINK=-L$(CUDATOOLKIT_HOME)/lib64 -lcudart
# Need to link to ALL the libraries, even though we don't call them all.
CUDA: FFTX_LIBRARIES=-lfftx_mpi -lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu -lfftx_rconv_gpu
# Targets to build.
CUDA: batch1d_test_driver perf_test_driver transformlib_test

#
## make HIP: needs ROCM_PATH and CRAY_MPICH_PREFIX
#
HIP: CC=hipcc
HIP: PRESETS=-DFFTX_HIP
# To get mpi.h
HIP: CC_INCLUDE=-I$(CRAY_MPICH_PREFIX)/include
HIP: CC_LINK=-L$(ROCM_PATH)/lib -lamdhip64 -lhipfft -lrocfft -lstdc++
# Need to link to ALL the libraries, even though we don't call them all.
HIP: FFTX_LIBRARIES=-lfftx_mpi -lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu -lfftx_rconv_gpu
# Targets to build.
HIP: batch1d_test_driver perf_test_driver transformlib_test

#
## make CPU: default
#
CPU: CC=mpicxx
CPU: CCFLAGS=-std=c++11
# Need to link to ALL the libraries, even though we don't call them all.
CPU: FFTX_LIBRARIES=-lfftx_mddft_cpu -lfftx_imddft_cpu -lfftx_mdprdft_cpu -lfftx_imdprdft_cpu -lfftx_dftbat_cpu -lfftx_idftbat_cpu -lfftx_prdftbat_cpu -lfftx_iprdftbat_cpu -lfftx_rconv_cpu
# Targets to build.
CPU: poissonTest

#
# Compile source files to object files in temp directory.
#
%.o : %.cpp
	mkdir -p temp
	$(CC) $(CCFLAGS) $(PRESETS) $(CC_INCLUDE) $(FFTX_INCLUDE) $< -c -o temp/$@

#
# Link object files, put executables in bin directory.
#
LD=mpicc
poissonTest batch1d_test_driver perf_test_driver transformlib_test : % : %.o
	mkdir -p bin
	$(LD) $(LDFLAGS) $(CC_LINK) $(FFTX_LINK) $(FFTX_LIBRARIES) temp/$< -o bin/$@

#
## make clean: remove executables and other files
#
clean:
		@rm -f bin/poissonTest temp/poissonTest.o
		@rm -f bin/batch1d_test_driver temp/batch1d_test_driver.o
		@rm -f bin/perf_test_driver temp/perf_test_driver.o
		@rm -f bin/transformlib_test temp/transformlib_test.o

#
## make help: prints the help
#
.PHONY: help
help:
		@awk '/^##/ {$$1=""; print $$0}' Makefile
