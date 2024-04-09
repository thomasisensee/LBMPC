#include <iostream>

#include <cuda_runtime.h>

#include "cudaUtilities.h"

#include "cudaErrorHandler.cuh"

void DisplayDeviceProperties()
{
    cudaDeviceProp deviceProp;
    cudaMemset(&deviceProp, 0, sizeof(deviceProp));
    int device = getDevice();
    cudaErrorCheck(cudaGetDeviceProperties(&deviceProp, device))
    
    std::cout << "==========================================================================" << std::endl;
    std::cout << "== Device Name\t" << deviceProp.name << "\t\t\t\t\t==" << std::endl;
    std::cout << "== Device Index\t" << device << "\t\t\t\t\t\t\t==" << std::endl;
    std::cout << "==========================================================================" << std::endl;
    printf( "== Total Global Memory\t\t\t %ld KB\t\t\t==\n", (long int)(deviceProp.totalGlobalMem/1024) );
    printf( "== Shared memory available per block\t %ld KB\t\t\t\t==\n", (long int)(deviceProp.sharedMemPerBlock/1024) );
    printf( "== Number of registers per thread block  %d\t\t\t\t==\n", deviceProp.regsPerBlock );
    printf( "== Warp size in threads             \t %d\t\t\t\t==\n", deviceProp.warpSize );
    printf( "== Memory Pitch                     \t %ld bytes\t\t==\n", (long int)(deviceProp.memPitch) );
    printf( "== Maximum threads per block        \t %d\t\t\t\t==\n", deviceProp.maxThreadsPerBlock );
    printf( "== Maximum Thread Dimension (block) \t %d * %d * %d\t\t==\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2] );
    printf( "== Maximum Thread Dimension (grid)  \t %d * %d * %d\t==\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2] );
    printf( "== Total constant memory            \t %ld bytes\t\t\t==\n", (long int)(deviceProp.totalConstMem) );
    printf( "== Compute capability               \t %d.%d\t\t\t\t==\n", deviceProp.major, deviceProp.minor );
    printf( "== Clock rate                       \t %d KHz\t\t\t==\n", deviceProp.clockRate );
    printf( "== Texture Alignment                \t %ld bytes\t\t\t==\n", (long int)(deviceProp.textureAlignment) );
    printf( "== Device Overlap                   \t %s\t\t\t==\n", deviceProp. deviceOverlap?"Allowed":"Not Allowed" );
    printf( "== Number of Multi processors       \t %d\t\t\t\t==\n", deviceProp.multiProcessorCount );
    std::cout << "==========================================================================" << std::endl << std::endl;
}

void SetDevice()
{
    int cudaDeviceCount = getDeviceCount();
    if (cudaDeviceCount < 1) {
        std::cout << "No CUDA devices with compute capability greater or equal to 2.0 found." << std::endl;
        return;
    }
    unsigned int cudaDevice = 0;
    cudaSetDevice(cudaDevice);
    DisplayDeviceProperties();
}

int getDevice() {
  int device{};
  cudaErrorCheck(cudaGetDevice(&device));
  return device;
}

int getDeviceCount() {
  int devices{};
  cudaErrorCheck(cudaGetDeviceCount(&devices));
  return devices;
}

CUDATimer::CUDATimer() {
    cudaErrorCheck(cudaEventCreate(&start));
    cudaErrorCheck(cudaEventCreate(&stop));
}

CUDATimer::~CUDATimer() {
    cudaErrorCheck(cudaEventDestroy(start));
    cudaErrorCheck(cudaEventDestroy(stop));
}

void CUDATimer::startTimer() {
    cudaErrorCheck(cudaEventRecord(start, 0));
}

void CUDATimer::stopTimer() {
    cudaErrorCheck(cudaEventRecord(stop, 0));
    cudaErrorCheck(cudaEventSynchronize(stop));
}

float CUDATimer::getElapsedTime() {
        float milliseconds = 0;
        cudaErrorCheck(cudaEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
}