Demo FFTX External Application
==============================

This is the public repository for the Demo FFTX External Application

### Building Demo FFTX External Application

To use and build this demo application you must have installed FFTX and
spiral-software. Ensure your environment sets **FFTX_HOME** and **SPIRAL_HOME** to point
to the locations of FFTX and Spiral.

### Installing Pre-requisites

Clone **spiral-software** to a location on your computer.  E.g., do:
```
cd ~/work
git clone https://www.github.com/spiral-software/spiral-software
```
This location is known as *SPIRAL HOME* and you must set an environment variable
**SPIRAL_HOME** to point to this location later.

You must also install two spiral packages, do the following:
```
cd ~/work/spiral-software/namespaces/packages
git clone https://www.github.com/spiral-software/spiral-package-fftx fftx
git clone https://www.github.com/spiral-software/spiral-package-simt simt
```
**NOTE:** The spiral packages must be installed under the directory
**$SPIRAL_HOME/namespaces/packages** and must be placed in folders with the
prefix *spiral-package* removed. 

Follow the build instructions for **spiral-software** (see the **README**
[**here**](https://github.com/spiral-software/spiral-software/blob/master/README.md) ).

### Installing FFTX

Clone **FFTX** to a location on your computer.  E.g., do:
```
cd ~/work
git clone https://www.github.com/spiral-software/fftx
```
Follow the build instructions for **FFTX** (see the **README**
[**here**](https://github.com/spiral-software/FFTX/blob/master/README.md) ).

### Install and Build the Demo Application

Ensure you have valid settings for **FFTX_HOME** and **SPIRAL_HOME**.  Clone the
demo application.  E.g., do:
```
cd ~/work
git clone https://www.github.com/spiral-software/fftx-demo-extern-app
cd fftx-demo-extern-app
mkdir build
cd build
cmake ..
make install
```
The demo application is installed at ~/work/fftx-demo-extern-app/build/bin
