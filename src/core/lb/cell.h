#ifndef CELL_H
#define CELL_H

template<typename T>
class Cell {
private:
    __device__ T computeEquilibriumPopulation(unsigned int l, const LBParams<T>*const params, T R, T U, T V) const;

public:
    /// Computes the 0th moment (density)
    __device__ T getZeroMoment(const T*const population, const LBParams<T>*const params) const;

    /// Computes the 1st moment X
    __device__ T getFirstMomentX(const T*const population, const LBParams<T>*const params) const;

    /// Computes the 1st moment Y
    __device__ T getFirstMomentY(const T*const population, const LBParams<T>*const params) const;

    /// Computes X-component of the macroscopic velocity
    __device__ T getVelocityX(const T*const population, const LBParams<T>*const params) const;
    __device__ T getVelocityX(const T*const population, const LBParams<T>*const params, T R) const;

    /// Computes Y-component of the macroscopic velocity
    __device__ T getVelocityY(const T*const population, const LBParams<T>*const params) const;
    __device__ T getVelocityY(const T*const population, const LBParams<T>*const params, T R) const;

    /// Computes the equilibrium distribution and writes in parsed array eqDistr
    __device__ void getEquilibriumDistribution(const T*const population, T* eqDistr, const LBParams<T>*const params, T R, T U, T V) const;

    /// Computes the equilibrium distribution and writes in parsed populations
    __device__ void setEquilibriumDistribution(T* population, const LBParams<T>*const params, T R, T U, T V) const;

    /// Computes the post-collision distribution via the BGK collision operator
    __device__ void computePostCollisionDistributionBGK(T* population, const CollisionParamsBGK<T>*const params, T R, T U, T V) const;

    /// Computes the post-collision distribution via the CHM MRT collision operator
    __device__ void computePostCollisionDistributionCHM(T* population, const CollisionParamsCHM<T>*const params, T R, T U, T V) const;
};

#include "cell.hh"

#endif // CELL_H
