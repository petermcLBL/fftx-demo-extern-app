##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

##  Sample external application that wants to use FFTX
##  define normal items needed in app

cmake_minimum_required ( VERSION 3.18 )
set ( PROJECT transformlib_test )
project ( ${PROJECT} LANGUAGES C CXX CUDA )

set ( DFT_libtest dftbatlib_test )
set ( COMPARE_device testcompare_device )

##  Set C standard 11
set ( CMAKE_C_STANDARD 11 )
set ( CMAKE_CXX_STANDARD 11 )

set ( CUDA_COMPILE_FLAGS -m64 -rdc=false )
if ( WIN32 )
    set ( LIBS_FOR_CUDA cufft )
else ()
    set ( LIBS_FOR_CUDA cufft culibos )
endif ()

add_executable             ( ${PROJECT} ${PROJECT}.cu )
target_compile_options     ( ${PROJECT} PRIVATE ${CUDA_COMPILE_FLAGS} )
set_property        ( TARGET ${PROJECT} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
target_link_libraries      ( ${PROJECT} ${LIBS_FOR_CUDA} )

add_executable             ( ${COMPARE_device} ${COMPARE_device}.cu )
target_compile_options     ( ${COMPARE_device} PRIVATE ${CUDA_COMPILE_FLAGS} )
set_property        ( TARGET ${COMPARE_device} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
target_link_libraries      ( ${COMPARE_device} ${LIBS_FOR_CUDA} )

add_executable             ( ${DFT_libtest} ${DFT_libtest}.cpp )

##  Setup and access any SPIRAL related items...
##  Get SPIRAL home... only need SPIRAL for some include files

if ( DEFINED ENV{SPIRAL_HOME} )
    message ( STATUS "SPIRAL_HOME = $ENV{SPIRAL_HOME}" )
    set ( SPIRAL_SOURCE_DIR $ENV{SPIRAL_HOME} )
else ()
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
        message ( FATAL_ERROR "SPIRAL_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( SPIRAL_SOURCE_DIR ${SPIRAL_HOME} )
endif ()

include_directories ( ${SPIRAL_SOURCE_DIR}/profiler/targets/include )

##  FFTX_HOME must be defined in the environment
if ( DEFINED ENV{FFTX_HOME} )
    message ( STATUS "FFTX_HOME = $ENV{FFTX_HOME}" )
    set ( FFTX_SOURCE_DIR $ENV{FFTX_HOME} )
else ()
    if ( "x${FFTX_HOME}" STREQUAL "x" )
        message ( FATAL_ERROR "FFTX_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( FFTX_SOURCE_DIR ${FFTX_HOME} )
endif ()

##  Include FFTX CMake functions
include ( "${FFTX_SOURCE_DIR}/CMakeIncludes/FFTXCmakeFunctions.cmake" )

##  FFTX_find_libraries () finds the FFTX libraries, paths, etc. and
##  exposes the following variables:
##
##  FFTX_LIB_INCLUDE_PATHS -- include paths for FFTX include & library headers
##  FFTX_LIB_NAMES         -- list of FFTX libraries
##  FFTX_LIB_LIBRARY_PATH  -- path to libraries (for linker)
##  
##  You don't need to call FFTX_find_libraries() directly unless you
##  specifically want access to the variables listed above -- its
##  called as part of FFTX_add_includes_libs_to_target ( _target )
##

FFTX_find_libraries ()
##  message ( STATUS "Include paths:   ${FFTX_LIB_INCLUDE_PATHS}" )
##  message ( STATUS "Libraries found: ${FFTX_LIB_NAMES}" )
##  message ( STATUS "Library path is: ${FFTX_LIB_LIBRARY_PATH}" )

##  Setup CMAKE_INSTALL_RPATH to point to the FFTX library directory
set ( CMAKE_INSTALL_RPATH ${FFTX_LIB_LIBRARY_PATH} )
message ( STATUS "Defined: INSTALL_RPATH = ${CMAKE_INSTALL_RPATH}" )

FFTX_add_includes_libs_to_target ( ${PROJECT} )
FFTX_add_includes_libs_to_target ( ${DFT_libtest} )
FFTX_add_includes_libs_to_target ( ${COMPARE_device} )

install ( TARGETS
    ${PROJECT} ${DFT_libtest} ${COMPARE_device}
    DESTINATION ${CMAKE_BINARY_DIR}/bin )
