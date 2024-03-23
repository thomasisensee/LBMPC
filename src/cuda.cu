#include <stdio.h>
#include <cuda_runtime.h>

#include "constants.h"
#include "cuda.h"
#include "lbmModel.h"
#include "gridGeometry.h"

/**
 * Checks the returned cudaError_t and prints corresponding message in case of error.
 */
#define cudaErrorCheck(ans){ cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"CUDA Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template<typename T>
void allocateDeviceField(T** d_array, size_t ArraySize)
{
	cudaErrorCheck(cudaMalloc(d_array, ArraySize));
}
template void allocateDeviceField<float>(float**, size_t);
template void allocateDeviceField<double>(double**, size_t);

template<typename T>
void freeDeviceField(T* d_array)
{
	cudaErrorCheck(cudaFree(d_array));
}
template void freeDeviceField<float>(float*);
template void freeDeviceField<double>(double*);

template<typename T>
__global__ void initializeLBMDistributions(T* Collide, LBMModel<T>* lbmModel, GridGeometry2D<T>* gridGeometry)
{
#define pos(x,y)		(Nx*(y)+(x))
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    const unsigned int Q = lbmModel->getQ();
    const unsigned int Nx = gridGeometry->getGhostNx();
    
	T firstOrder, secondOrder, thirdOrder, fourthOrder;
    int cix,ciy;
    T cixcs2,ciycs2;
    T U=0.0,V=0.0;
    
	for(int l=0; l<Q; l++)
	{
	    //cix = lbmModel->getCX(i); //-->illegal memory access
        //ciy = lbmModel->getCY(i);
        cix = 0.0;
        ciy = 0.0;
   		printf("(cix,ciy) = (%d,%d)\n",cix,ciy);
        cixcs2 = cix*cix-C_S_POW2;
        ciycs2 = ciy*ciy-C_S_POW2;
        firstOrder = C_S_POW2_INV*(U*cix+V*ciy);
        secondOrder = 0.5*C_S_POW4_INV*(cixcs2*U*U + ciycs2*V*V + 2.0*cix*ciy*U*V);
        thirdOrder = 0.5*C_S_POW6_INV*(cixcs2*ciy*U*U*V + ciycs2*cix*U*V*V);
        fourthOrder = 0.25*C_S_POW8_INV*(cixcs2*ciycs2*U*U*V*V);
 

		Collide[Q*pos(i,j)+l] = 1.0;
	}
}
template __global__ void initializeLBMDistributions<float>(float*, LBMModel<float>*, GridGeometry2D<float>*);
template __global__ void initializeLBMDistributions<double>(double*, LBMModel<double>*, GridGeometry2D<double>*);

template<typename T>
void KERNEL_CALLER_initializeLBMDistributions(T* Collide, LBMModel<T>* lbmModel, GridGeometry2D<T>* gridGeometry)
{
    /// Create a D2Q9(LBMModel) device pointer and copy from Host to Device
    D2Q9<T>* d_lbmModel;
    cudaMalloc(&d_lbmModel, sizeof(D2Q9<T>));
    cudaMemcpy(d_lbmModel, lbmModel, sizeof(D2Q9<T>), cudaMemcpyHostToDevice);
 
    /// Create a D2Q9(LBMModel) device pointer and copy from Host to Device
    GridGeometry2D<T>* d_gridGeometry;
    cudaMalloc(&d_gridGeometry, sizeof(GridGeometry2D<T>));
    cudaMemcpy(d_gridGeometry, gridGeometry, sizeof(GridGeometry2D<T>), cudaMemcpyHostToDevice);

    /// Call CUDA kernel
    unsigned int blockDimX = 16;
    unsigned int blockDimY = 16;
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((gridGeometry->getGhostNx() + blockDimX - 1) / blockDimX, (gridGeometry->getGhostNy() + blockDimY - 1) / blockDimY);
    initializeLBMDistributions<T><<<gridDim,blockDim>>>(Collide, d_lbmModel, d_gridGeometry);

    /// Free allocated device memory
    cudaFree(d_lbmModel);
    cudaFree(d_gridGeometry);
}
template void KERNEL_CALLER_initializeLBMDistributions<float>(float*, LBMModel<float>*, GridGeometry2D<float>*);
template void KERNEL_CALLER_initializeLBMDistributions<double>(double*, LBMModel<double>*, GridGeometry2D<double>*);
