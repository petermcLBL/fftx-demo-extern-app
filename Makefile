#
# FFTX
#
FFTX_INCLUDE=$(FFTX_HOME)/include
FFTX_MPI_INCLUDE=$(FFTX_HOME)/src/library/lib_fftx_mpi
FFTX_LIB_DIR=$(FFTX_HOME)/lib
FFTX_LINK=-Wl,-rpath=$(FFTX_LIB_DIR) -L$(FFTX_LIB_DIR)
# libraries after $(FFTX_LINK)

#
# either CUDA or HIP
#
ifndef_any_of = $(filter undefined,$(foreach v,$(1),$(origin $(v))))
ifdef_any_of = $(filter-out undefined,$(foreach v,$(1),$(origin $(v))))
ifneq ($(call ifdef_any_of,CUDATOOLKIT_HOME ROCM_PATH), )
  # Need to link to ALL the libraries, even though we don't call them all.
  FFTX_LIBRARIES=-lfftx_mpi -lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu -lfftx_rconv_gpu
#
# CPU only
#
else
  # Need to link to ALL the libraries, even though we don't call them all.
  FFTX_LIBRARIES=-lfftx_mddft_cpu -lfftx_imddft_cpu -lfftx_mdprdft_cpu -lfftx_imdprdft_cpu -lfftx_dftbat_cpu -lfftx_idftbat_cpu -lfftx_prdftbat_cpu -lfftx_iprdftbat_cpu -lfftx_rconv_cpu  
endif

#
# CUDA only
#
ifdef CUDATOOLKIT_HOME
  CC=nvcc
  CC_FLAGS=-x cu -DFFTX_CUDA
  # FFTX_GPU_INCLUDE=/usr/local/cuda/include
  ### To get helper_cuda.h
  CUDA_INCLUDE=$(CUDATOOLKIT_HOME)/../../examples/OpenMP/SDK/include
  CC_INCLUDE=-I$(CUDA_INCLUDE)
  # CUDA_LIB=/usr/local/cuda/lib64/
  CUDA_LIB=$(CUDATOOLKIT_HOME)/lib64
  ### To get libcudart.so.11.0 on perlmutter.
  # CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64
  CUDA_LINK=-L$(CUDA_LIB) -lcudart
  CC_LINK=$(CUDA_LINK)
  # Get rid of annoying "DSO missing from command line" error.
  LDFLAGS=-Wl,--copy-dt-needed-entries


#
# HIP only
#
else ifdef ROCM_PATH
  CC=hipcc
  CC_FLAGS=-DFFTX_HIP
  ### To get mpi.h
  HIP_INCLUDE=$(CRAY_MPICH_PREFIX)/include
  CC_INCLUDE=-I$(HIP_INCLUDE)
  # FFTX_GPU_INCLUDE=/opt/rocm-5.3.0/include
  HIP_LIB=$(ROCM_PATH)/lib
  # HIP_LIB=/opt/rocm-5.3.0/lib/
  HIP_LINK=-L$(HIP_LIB) -lamdhip64 -lhipfft -lrocfft -lstdc++
  CC_LINK=$(HIP_LINK)

#
# neither CUDA nor HIP, so CPU
#
else
  CC=mpicxx
  # Get rid of annoying "DSO missing from command line" error.
  LDFLAGS=-Wl,--copy-dt-needed-entries
endif

LINK_LINE=mpicc $(LDFLAGS) $(CC_LINK) $(FFTX_LINK) $(FFTX_LIBRARIES)

COMPILE_LINE=$(CC) $(CC_FLAGS) $(CC_INCLUDE) -I$(FFTX_MPI_INCLUDE) -I$(FFTX_INCLUDE)

ifneq ($(call ifdef_any_of,CUDATOOLKIT_HOME ROCM_PATH), )

default: bin/poissonTest bin/batch1d_test_driver bin/perf_test_driver bin/transformlib_test

else

default: bin/poissonTest

endif

bin/poissonTest: temp/poissonTest.o
	mkdir -p bin
	$(LINK_LINE) temp/poissonTest.o -o bin/poissonTest

temp/poissonTest.o: poissonTest.cpp
	mkdir -p temp
	$(COMPILE_LINE) poissonTest.cpp -c -o temp/poissonTest.o

#
# These ones require either CUDA or HIP.
#
ifneq ($(call ifdef_any_of,CUDATOOLKIT_HOME ROCM_PATH), )

bin/batch1d_test_driver: temp/batch1d_test_driver.o
	mkdir -p bin
	$(LINK_LINE) temp/batch1d_test_driver.o -o bin/batch1d_test_driver

temp/batch1d_test_driver.o: batch1d_test_driver.cpp
	mkdir -p temp
	$(COMPILE_LINE) batch1d_test_driver.cpp -c -o temp/batch1d_test_driver.o


bin/perf_test_driver: temp/perf_test_driver.o
	mkdir -p bin
	$(LINK_LINE) temp/perf_test_driver.o -o bin/perf_test_driver

temp/perf_test_driver.o: perf_test_driver.cpp
	mkdir -p temp
	$(COMPILE_LINE) perf_test_driver.cpp -c -o temp/perf_test_driver.o


bin/transformlib_test: temp/transformlib_test.o
	mkdir -p bin
	$(LINK_LINE) temp/transformlib_test.o -o bin/transformlib_test

temp/transformlib_test.o: transformlib_test.cpp
	mkdir -p temp
	$(COMPILE_LINE) transformlib_test.cpp -c -o temp/transformlib_test.o

endif

## make clean: remove executables and other files
clean:
		@rm -f bin/poissonTest temp/poissonTest.o
		@rm -f bin/batch1d_test_driver temp/batch1d_test_driver.o
		@rm -f bin/perf_test_driver temp/perf_test_driver.o
		@rm -f bin/transformlib_test temp/transformlib_test.o


## make help: prints the help
.PHONY: help
help:
		@awk '/^##/ {$$1=""; print $$0}' Makefile
