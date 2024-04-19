#ifndef FUNCTORS_H
#define FUNCTORS_H

#include "core/kernelParameters.h"

namespace functors {

template<typename T,typename DESCRIPTOR>
class StandardEquilibrium {
private:
    T _R, _U, _V;
public:
    __device__ explicit StandardEquilibrium(T* population);
    __device__ T operator()(unsigned int l) const;
};

template<typename T,typename DESCRIPTOR>
class ScalarEquilibrium {
private:
    T _R, _U, _V;
public:
    __device__ explicit ScalarEquilibrium(T* population);
    __device__ ScalarEquilibrium(T* population, T U, T V);
    __device__ T operator()(unsigned int l) const;
};

template<typename T,typename DESCRIPTOR>
class BounceBack {
public:
    /// Constructor
    BounceBack() = default;
    __device__ T operator()(T* collision, const BaseParams* const params, unsigned int i, unsigned int j);
};

template<typename T,typename DESCRIPTOR>
class MovingWall {
public:
    /// Constructor
    MovingWall() = default;
    __device__ T operator()(T* collision, const BaseParams* const params, unsigned int i, unsigned int j);
};

template<typename T,typename DESCRIPTOR>
class AntiBounceBack {
public:
    /// Constructor
    AntiBounceBack() = default;
    __device__ T operator()(T* collision, const BaseParams* const params, unsigned int i, unsigned int j);
};


} // namespace functors

#include "functors.hh"

#endif // FUNCTORS_H
