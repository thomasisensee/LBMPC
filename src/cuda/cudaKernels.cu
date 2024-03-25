#include <stdio.h>
#include <cuda_runtime.h>

#include "core/constants.h"
#include "core/lbmModel.h"
#include "core/gridGeometry.h"

#include "cudaKernels.h"
#include "cudaErrorHandler.h"

#include <iostream>
template<typename T, typename LBMModelClassType>
void launchCreateDeviceModel(LBMModelClassType** deviceModel)
{
    createDeviceModel<T, LBMModelClassType><<<1, 1>>>(deviceModel);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void launchCreateDeviceModel<float, D2Q9<float>>(D2Q9<float>**);
template void launchCreateDeviceModel<double, D2Q9<double>>(D2Q9<double>**);

template<typename T, typename LBMModelClassType>
__global__ void createDeviceModel(LBMModelClassType** deviceModel)
{
    printf("test\n");
    //printf("Before assignment: *deviceModel = %p\n", *deviceModel);
    
    // Allocate memory for the device-side object
    LBMModelClassType* obj = new LBMModelClassType;
    printf("cix = %d\n",obj->getCX(0));

    // Assign the object to the deviceModel pointer
    *deviceModel = obj;
    
    printf("After assignment: *deviceModel = %p\n", *deviceModel);
}
template __global__  void createDeviceModel<float, D2Q9<float>>(D2Q9<float>**);
template __global__  void createDeviceModel<double, D2Q9<double>>(D2Q9<double>**);

template<typename T, typename LBMModelClassType>
void test1(LBMModelClassType* lbmModel)
{
    useClass<T,LBMModelClassType><<<1,1>>>(lbmModel);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void test1<float, D2Q9<float>>(D2Q9<float>*);
template void test1<double, D2Q9<double>>(D2Q9<double>*);

template<typename T, typename LBMModelClassType>
__global__ void useClass(LBMModelClassType* lbmModel)
{
    unsigned int Q = lbmModel->getQ();
    unsigned int D = lbmModel->getD();
    printf("Q = %d, D = %d\n",Q,D);
    int cix,ciy;
    T w;
    for(unsigned int i=0; i<Q; ++i)
    {
        cix = lbmModel->LATTICE_VELOCITIES[i*2];
        ciy = lbmModel->LATTICE_VELOCITIES[i*2+1];
        w = lbmModel->LATTICE_WEIGHTS[i];
        //w = 5.;     
        //cix = lbmModel->getCX(i);
        //ciy = lbmModel->getCY(i);
        printf("cix = %d, ciy = %d, w = %g\n",cix,ciy,w);
    }
}
template __global__ void useClass<float, D2Q9<float>>(D2Q9<float>* lbmModel);
template __global__ void useClass<double, D2Q9<double>>(D2Q9<double>* lbmModel);

template<typename T, typename LBMModelClassType>
__global__ void initializeLBMDistributions(T* Collide, LBMModelClassType* lbmModel, GridGeometry2D<T>* gridGeometry)
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
template __global__ void initializeLBMDistributions<float>(float*, D2Q9<float>*, GridGeometry2D<float>*);
template __global__ void initializeLBMDistributions<double>(double*, D2Q9<double>*, GridGeometry2D<double>*);

template<typename T, typename LBMModelClassType>
void KERNEL_CALLER_initializeLBMDistributions(T* Collide, LBMModelWrapper<T, LBMModelClassType>* lbmModel, GridGeometry2DWrapper<T>* gridGeometry)
{
    /// Call CUDA kernel
    unsigned int blockDimX = 16;
    unsigned int blockDimY = 16;
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((gridGeometry->getHostGridGeometry()->getGhostNx() + blockDimX - 1) / blockDimX, (gridGeometry->getHostGridGeometry()->getGhostNy() + blockDimY - 1) / blockDimY);

    initializeLBMDistributions<T><<<gridDim,blockDim>>>(Collide, lbmModel->getDeviceModel(), gridGeometry->getDeviceGridGeometry());
}
template void KERNEL_CALLER_initializeLBMDistributions<float>(float*, LBMModelWrapper<float, D2Q9<float>>*, GridGeometry2DWrapper<float>*);
template void KERNEL_CALLER_initializeLBMDistributions<double>(double*, LBMModelWrapper<double, D2Q9<float>>*, GridGeometry2DWrapper<double>*);
