#ifndef FUNCTORS_H
#define FUNCTORS_H

#include "core/kernelParameters.h"

namespace functors {

namespace boundary {

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

} // namespace boundary

namespace force {

class NoForce {
public:
    /// Constructor
    NoForce() = default;
    __device__ void operator()() {};
};

template<typename T,typename DESCRIPTOR>
class ThermalBuoyancy {
public:
    /// Constructor
    ThermalBuoyancy() = default;
    __device__ T operator()(T* temperature, const CollisionParamsBGK<T>* const params, unsigned int i, unsigned int j);
};

} // namespace force

} // namespace functors

#include "functors.hh"

#endif // FUNCTORS_H
