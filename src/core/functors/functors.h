#ifndef FUNCTORS_H
#define FUNCTORS_H

#include "core/kernelParameters.h"

namespace functors {

template<typename T,typename DESCRIPTOR>
class StandardEquilibrium {
private:
    T _R, _U, _V;
public:
    __device__ StandardEquilibrium(T* population);
    __device__ T operator()(unsigned int l) const;
};

template<typename T,typename DESCRIPTOR>
class ScalarEquilibrium {
private:
    T _R, _U, _V;
public:
    __device__ ScalarEquilibrium(T* population);
    __device__ ScalarEquilibrium(T* population, T U, T V);
    __device__ T operator()(unsigned int l) const;
};

template<typename T,typename DESCRIPTOR>
class BounceBack {
public:
    /// Empty constructor
    __device__ BounceBack();
    __device__ T operator()(T* collision, const BaseParams* const params, unsigned int i, unsigned int j);
};

template<typename T,typename DESCRIPTOR>
class MovingWall {
public:
    /// Empty constructor
    __device__ MovingWall();
    __device__ T operator()(T* collision, const BaseParams* const params, unsigned int i, unsigned int j);
};

template<typename T,typename DESCRIPTOR>
class AntiBounceBack {
public:
    /// Empty constructor
    __device__ AntiBounceBack();
    __device__ T operator()(T* collision, const BaseParams* const params, unsigned int i, unsigned int j);
};


} // namespace functors

#include "functors.hh"

#endif // FUNCTORS_H