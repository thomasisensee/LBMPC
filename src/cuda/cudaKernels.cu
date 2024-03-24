#include <stdio.h>
#include <cuda_runtime.h>

#include "cudaKernels.h"
#include "cudaErrorHandler.h"
#include "core/constants.h"
#include "core/lbmModel.h"
#include "core/gridGeometry.h"

template<typename T>
__global__ void testKernel(T a)
{
    printf("Test kernel.\n");
};
template __global__ void testKernel<float>(float a);
template __global__ void testKernel<double>(double a);

template<typename T>
__global__ void useClass(LBMModel<T>* lbmModel)
{
    unsigned int Q = lbmModel->getQ();
    unsigned int D = lbmModel->getD();
    printf("Q = %d, D = %d\n",Q,D);
    int cix,ciy;
    for(unsigned int i=0; i<Q; i++)
    {
        //cix = lbmModel->LATTICE_VELOCITIES[i*2];
        //ciy = lbmModel->LATTICE_VELOCITIES[i*2+1];        
        cix = lbmModel->getCX(i);
        ciy = lbmModel->getCY(i);
        printf("cix = %d, ciy = %d\n",cix,ciy);
    }
}
template __global__ void useClass<float>(LBMModel<float>* lbmModel);
template __global__ void useClass<double>(LBMModel<double>* lbmModel);

template<typename T>
__global__ void testKernel(T* test)
{
    for(int i=0; i<9; i++)
    {
        printf("%g\n",test[i]);
    }
}
template __global__ void testKernel<float>(float* test);
template __global__ void testKernel<double>(double* lbtestmModel);

template<typename T>
void test1(LBMModel<T>* lbmModel)
{
    printf("before\n");
    useClass<T><<<1,1>>>(lbmModel);
    cudaDeviceSynchronize();
    T a = 1.0;
    //testKernel<T><<<1,1>>>(a);
    cudaDeviceSynchronize();
    printf("after\n");
}
template void test1<float>(LBMModel<float>*);
template void test1<double>(LBMModel<double>*);

template<typename T>
void test2(T* test)
{
    printf("before\n");
    testKernel<<<1,1>>>(test);
    cudaDeviceSynchronize();
    //testKernel<T><<<1,1>>>(a);
    printf("after\n");
}
template void test2<float>(float*);
template void test2<double>(double*);


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
void KERNEL_CALLER_initializeLBMDistributions(T* Collide, LBMModelWrapper<T>* lbmModel, GridGeometry2DWrapper<T>* gridGeometry)
{
    /// Call CUDA kernel
    unsigned int blockDimX = 16;
    unsigned int blockDimY = 16;
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((gridGeometry->getHostGridGeometry()->getGhostNx() + blockDimX - 1) / blockDimX, (gridGeometry->getHostGridGeometry()->getGhostNy() + blockDimY - 1) / blockDimY);

    initializeLBMDistributions<T><<<gridDim,blockDim>>>(Collide, lbmModel->getDeviceModel(), gridGeometry->getDeviceGridGeometry());
}
template void KERNEL_CALLER_initializeLBMDistributions<float>(float*, LBMModelWrapper<float>*, GridGeometry2DWrapper<float>*);
template void KERNEL_CALLER_initializeLBMDistributions<double>(double*, LBMModelWrapper<double>*, GridGeometry2DWrapper<double>*);
