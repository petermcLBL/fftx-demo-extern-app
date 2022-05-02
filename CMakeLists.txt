##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

##  Sample external application that wants to use FFTX
##  define normal items needed in app

cmake_minimum_required ( VERSION 3.14 )
set ( PROJECT dftbatlib_test )

enable_language ( C )
enable_language ( CXX )

set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 )

include ( CheckLanguage )
check_language ( CUDA )
if ( CMAKE_CUDA_COMPILER )
    enable_language ( CUDA )
    find_package ( CUDAToolkit )
    if ( ${CUDAToolkit_FOUND} )
	message ( STATUS "CUDA Toolkit Found : Version = ${CUDAToolkit_VERSION}" )
	project ( ${PROJECT} LANGUAGES C CXX CUDA )
    else ()
	message ( STATUS "CUDA Toolkit NOT found -- only build CPU / serial example" )
	project ( ${PROJECT} LANGUAGES C CXX )
    endif ()
else ()
    message ( STATUS "No support for CUDA available" )
endif ()

check_language ( HIP )
if ( CMAKE_HIP_COMPILER )
    enable_language ( HIP )
else ()
    message ( STATUS "No support for HIP available" )
endif ()
    
if ( APPLE )
    ##  Check if we're on Apple/MAC
    ##  Check if arch should be x86_64 or arm64
    message ( STATUS "Running on Apple / MAC" )
    message ( STATUS "CMAKE_OSX_ARCHITECTURES = ${CMAKE_OSX_ARCHITECTURES}" )
    message ( STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}" )
    if ( ${CMAKE_OSX_ARCHITECTURES} MATCHES "arm64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64.*" )
	set ( ADDL_COMPILE_FLAGS -arch arm64 )
        message ( STATUS "Set addl compile flags: ${ADDL_COMPILE_FLAGS}" )
    elseif ( ${CMAKE_OSX_ARCHITECTURES} MATCHES "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64.*")
	set ( ADDL_COMPILE_FLAGS -arch x86_64 )
        message ( STATUS "Set addl compile flags: ${ADDL_COMPILE_FLAGS}" )
    endif ()
endif ()	

set ( XFORMLIB_test transformlib_test )
set ( POISSON1_TEST poissonTest )

##  Set C standard 11
set ( CMAKE_C_STANDARD 11 )
set ( CMAKE_CXX_STANDARD 11 )

if ( ${CUDAToolkit_FOUND} )
    set ( CUDA_COMPILE_FLAGS -m64 -rdc=false )
    if ( WIN32 )
	set ( LIBS_FOR_CUDA cufft )
    else ()
	set ( LIBS_FOR_CUDA cufft culibos )
    endif ()
endif ()

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

##  Setup CMAKE_INSTALL_PREFIX to point to the top level source directory
set ( CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR} )
message ( STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )

# ##  set CMAKE_INSTALL_RPATH to point to FFTX library directory
# ##  use, i.e. don't skip the full RPATH for the build tree
# set ( CMAKE_SKIP_BUILD_RPATH FALSE )

# ##  when building, don't use the install RPATH already (but later on when installing)
# set ( CMAKE_BUILD_WITH_INSTALL_RPATH FALSE )
# set ( CMAKE_INSTALL_RPATH ${FFTX_LIB_LIBRARY_PATH} )

# ##  Add the automatically determined parts of the RPATH that point to directories outside
# ##  the build tree to the install RPATH
# set ( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )

# ##  the RPATH to be used when installing, but only if it's not a system directory
# list ( FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES ${FFTX_LIB_LIBRARY_PATH} isSystemDir )
# if ( "${isSystemDir}" STREQUAL "-1" )
#     set ( CMAKE_INSTALL_RPATH ${FFTX_LIB_LIBRARY_PATH} )
# endif ()

set ( CMAKE_INSTALL_RPATH ${FFTX_LIB_LIBRARY_PATH} )
message ( STATUS "Defined: INSTALL_RPATH = ${CMAKE_INSTALL_RPATH}" )

add_executable             ( ${PROJECT} ${PROJECT}.cpp )
set ( _targets ${PROJECT} )

add_executable             ( ${POISSON1_TEST} ${POISSON1_TEST}.cpp )
list ( APPEND _targets ${POISSON1_TEST} )

if ( ${CUDAToolkit_FOUND} )
    add_executable             ( ${XFORMLIB_test} ${XFORMLIB_test}.cu )
    target_compile_options     ( ${XFORMLIB_test} PRIVATE ${CUDA_COMPILE_FLAGS} )
    set_property        ( TARGET ${XFORMLIB_test} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
    target_link_libraries      ( ${XFORMLIB_test} ${LIBS_FOR_CUDA} )
    list ( APPEND _targets ${XFORMLIB_test} )
endif ()

foreach ( _targ ${_targets} )
    FFTX_add_includes_libs_to_target ( ${_targ} )
endforeach ()

install ( TARGETS ${_targets} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin )
