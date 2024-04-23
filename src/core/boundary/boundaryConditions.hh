#ifndef BOUNDARY_CONDITIONS_HH
#define BOUNDARY_CONDITIONS_HH

#include <cuda_runtime.h>

#include "boundaryConditions.h"
#include "core/functors/functors.h"

#include "core/constants.h"
#include "core/utilities.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T,typename DESCRIPTOR,typename ParamsWrapperType>
BoundaryCondition<T,DESCRIPTOR,ParamsWrapperType>::BoundaryCondition(BoundaryLocation loc) : _location(loc) {}

template<typename T,typename DESCRIPTOR,typename ParamsWrapperType>
BoundaryLocation BoundaryCondition<T,DESCRIPTOR,ParamsWrapperType>::getLocation() const {
    return _location;
}

template<typename T,typename DESCRIPTOR,typename ParamsWrapperType>
void BoundaryCondition<T,DESCRIPTOR,ParamsWrapperType>::prepareKernelParams(const BaseParams& baseParams) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        baseParams.Nx,
        baseParams.Ny,
        this->_location
    );

    // Set block and grid size for cuda kernel execution
    this->_threadsPerBlock = THREADS_PER_BLOCK_DIMENSION*THREADS_PER_BLOCK_DIMENSION;
    if (this->_location == BoundaryLocation::EAST || this->_location == BoundaryLocation::WEST) {
        this->_numBlocks = (baseParams.Ny + this->_threadsPerBlock - 1) / this->_threadsPerBlock;
    } else {
        this->_numBlocks = (baseParams.Nx + this->_threadsPerBlock - 1) / this->_threadsPerBlock;
    }
}

template<typename T,typename DESCRIPTOR,typename ParamsWrapperType>
void BoundaryCondition<T,DESCRIPTOR,ParamsWrapperType>::printBoundaryLocation() const {
    std::cout << "== Boundary Location: " << boundaryLocationToString(_location) << "\t==" << std::endl;
}

template<typename T,typename DESCRIPTOR,typename ParamsWrapperType>
void BoundaryCondition<T,DESCRIPTOR,ParamsWrapperType>::printParameters() const {
    printBoundaryLocation();
    std::cout << "== Condition: " << "Base" << "\t=="  << std::endl;
}


/**************************************/
/***** Derived class 01: Periodic *****/
/**************************************/
template<typename T,typename DESCRIPTOR>
Periodic<T,DESCRIPTOR>::Periodic(BoundaryLocation loc) : BoundaryCondition<T,DESCRIPTOR,PeriodicParamsWrapper<T>>(loc) {}

template<typename T,typename DESCRIPTOR>
void Periodic<T,DESCRIPTOR>::apply(T* lbmField) {

}

template<typename T,typename DESCRIPTOR>
void Periodic<T,DESCRIPTOR>::printParameters() const {
    BoundaryCondition<T,DESCRIPTOR,PeriodicParamsWrapper<T>>::printBoundaryLocation();
    std::cout << "== Condition: " << "Periodic" << "\t=="  << std::endl;
}

/*****************************************/
/***** Derived class 02: Bounce Back *****/
/*****************************************/
template<typename T,typename DESCRIPTOR>
BounceBack<T,DESCRIPTOR>::BounceBack(BoundaryLocation loc) : BoundaryCondition<T,DESCRIPTOR,BounceBackParamsWrapper<T>>(loc) {}

template<typename T,typename DESCRIPTOR>
void BounceBack<T,DESCRIPTOR>::apply(T* lbmField) {
    dim3 blockSize(this->_threadsPerBlock);
    dim3 gridSize(this->_numBlocks);
    applyBoundaryConditionCaller<T,DESCRIPTOR,functors::BounceBack<T,DESCRIPTOR>>(lbmField, this->_params.getDeviceParams(), gridSize, blockSize);
}

template<typename T,typename DESCRIPTOR>
void BounceBack<T,DESCRIPTOR>::printParameters() const {
    BoundaryCondition<T,DESCRIPTOR,BounceBackParamsWrapper<T>>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-back" << "\t=="  << std::endl;
}

/*******************************************************/
/***** Derived class 03: Moving Wall (Bounce Back) *****/
/*******************************************************/
template<typename T,typename DESCRIPTOR>
MovingWall<T,DESCRIPTOR>::MovingWall(BoundaryLocation loc, const std::vector<T>& velocity) : BoundaryCondition<T,DESCRIPTOR,MovingWallParamsWrapper<T>>(loc), _wallVelocity(velocity), _dxdt(0.0) {}

template<typename T,typename DESCRIPTOR>
void MovingWall<T,DESCRIPTOR>::prepareKernelParams(const BaseParams& baseParams) {
    BoundaryCondition<T,DESCRIPTOR,MovingWallParamsWrapper<T>>::prepareKernelParams(baseParams);
    this->_params.setWallVelocity(getWallVelocity());
}

template<typename T,typename DESCRIPTOR>
void MovingWall<T,DESCRIPTOR>::apply(T* lbmField) {
    dim3 blockSize(this->_threadsPerBlock);
    dim3 gridSize(this->_numBlocks);
    applyBoundaryConditionCaller<T,DESCRIPTOR,functors::MovingWall<T,DESCRIPTOR>>(lbmField, this->_params.getDeviceParams(), gridSize, blockSize);
}

template<typename T,typename DESCRIPTOR>
const std::vector<T>& MovingWall<T,DESCRIPTOR>::getWallVelocity() const {
    return _wallVelocity;
}

template<typename T,typename DESCRIPTOR>
void MovingWall<T,DESCRIPTOR>::printParameters() const {
    BoundaryCondition<T,DESCRIPTOR,MovingWallParamsWrapper<T>>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-Back with fixed velocity" << "\t=="  << std::endl;
    std::cout << "== Velocity = {" << getWallVelocity()[0] * _dxdt << ", " << getWallVelocity()[1] * _dxdt << "}\t=="  << std::endl;
}

/**********************************************/
/***** Derived class 04: Anti Bounce Back *****/
/**********************************************/
template<typename T,typename DESCRIPTOR>
AntiBounceBack<T,DESCRIPTOR>::AntiBounceBack(BoundaryLocation loc, T wallValue) : BoundaryCondition<T,DESCRIPTOR,AntiBounceBackParamsWrapper<T>>(loc), _wallValue(wallValue) {}

template<typename T,typename DESCRIPTOR>
void AntiBounceBack<T,DESCRIPTOR>::prepareKernelParams(const BaseParams& baseParams) {
    BoundaryCondition<T,DESCRIPTOR,AntiBounceBackParamsWrapper<T>>::prepareKernelParams(baseParams);
    this->_params.setWallValue(getWallValue());
}

template<typename T,typename DESCRIPTOR>
void AntiBounceBack<T,DESCRIPTOR>::apply(T* lbmField) {
    dim3 blockSize(this->_threadsPerBlock);
    dim3 gridSize(this->_numBlocks);
    applyBoundaryConditionCaller<T,DESCRIPTOR,functors::AntiBounceBack<T,DESCRIPTOR>>(lbmField, this->_params.getDeviceParams(), gridSize, blockSize);
}

template<typename T,typename DESCRIPTOR>
const T AntiBounceBack<T,DESCRIPTOR>::getWallValue() const {
    return _wallValue;
}

template<typename T,typename DESCRIPTOR>
void AntiBounceBack<T,DESCRIPTOR>::printParameters() const {
    BoundaryCondition<T,DESCRIPTOR,AntiBounceBackParamsWrapper<T>>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-Back with fixed velocity" << "\t=="  << std::endl;
    std::cout << "== Value = {" << getWallValue()  << "\t=="  << std::endl;
}

/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T,typename DESCRIPTOR>
BoundaryConditionManager<T,DESCRIPTOR>::BoundaryConditionManager() : _dxdt(1.0) {}

template<typename T,typename DESCRIPTOR>
void BoundaryConditionManager<T,DESCRIPTOR>::setDxdt(T dxdt) {
    _dxdt = dxdt;
}

template<typename T,typename DESCRIPTOR>
void BoundaryConditionManager<T,DESCRIPTOR>::addBoundaryCondition(std::unique_ptr<IBoundaryCondition<T>> condition) {
    for (const auto& existingCondition : boundaryConditions) {
        if (condition->getLocation() == existingCondition->getLocation()) {
            std::cerr << "Error: Boundary condition already exists for location " << boundaryLocationToString(condition->getLocation()) << std::endl;
            exit(EXIT_FAILURE);
        }
    }   
/*
    // If MovingWall, then set dxdt
    MovingWall<T,DESCRIPTOR>* MWCondition = dynamic_cast<MovingWall<T,DESCRIPTOR>*>(condition.get());
    if (MWCondition != nullptr) {
        MWCondition->setDxdt(_dxdt);
    }
*/
    boundaryConditions.push_back(std::move(condition));
}

template<typename T,typename DESCRIPTOR>
void BoundaryConditionManager<T,DESCRIPTOR>::prepareKernelParams(const BaseParams& baseParams) {
    for (const auto& condition : boundaryConditions) {
        condition->prepareKernelParams(baseParams);
    }
}

template<typename T,typename DESCRIPTOR>
void BoundaryConditionManager<T,DESCRIPTOR>::apply(T* lbmField) {
    for (const auto& condition : boundaryConditions) {
        condition->apply(lbmField);
    }
}

template<typename T,typename DESCRIPTOR>
void BoundaryConditionManager<T,DESCRIPTOR>::printParameters() const {
    std::cout << "====== Boundary conditions =======" << std::endl;
    for (const auto& condition : boundaryConditions) {
        condition->printParameters();
    }
    std::cout << "==================================\n" << std::endl;
}

#endif // BOUNDARY_CONDITIONS_HH
