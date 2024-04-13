#ifndef CELL_HH
#define CELL_HH

#include "cell.h"
#include "core/descriptors/latticeDescriptors.h"

using namespace latticeDescriptors;

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ T Cell<T,LATTICE_DESCRIPTOR>::computeEquilibriumPopulation(unsigned int l, T R, T U, T V) const {
    // Local constants for easier access
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    T cix = static_cast<T>(c<D,Q>(l, 0));
    T ciy = static_cast<T>(c<D,Q>(l, 1));
    T cixcs2 = cix * cix - cs2<T,D,Q>();
    T ciycs2 = ciy * ciy - cs2<T,D,Q>();
    T firstOrder = invCs2<T,D,Q>() * (U * cix + V * ciy);
    T secondOrder = 0.5 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * U * U + ciycs2 * V * V + 2.0 * cix * ciy * U * V);
    T thirdOrder = 0.5 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * ciy * U * U * V + ciycs2 * cix * U * V * V);
    T fourthOrder = 0.25 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * ciycs2 * U * U * V * V);

    return w<T,D,Q>(l) * (R * (1.0 + firstOrder + secondOrder + thirdOrder + fourthOrder) - 1.0); // minus one for suppress inaccuracy from round-off errors
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ T Cell<T,LATTICE_DESCRIPTOR>::getZerothMoment(const T* const population) const {
    // Local constants for easier access
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    T rho = 0.0;
    for (unsigned int l = 0; l < Q; ++l) { rho += population[l]; }
    return rho + 1.0; // plus one because of the above minus one
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ T Cell<T,LATTICE_DESCRIPTOR>::getFirstMomentX(const T* const population) const {
    // Local constants for easier access
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    T m1x = 0.0;
    for (unsigned int l = 0; l < Q; ++l) { m1x += population[l] * static_cast<T>(c<D,Q>(l, 0)); }
    return m1x;
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ T Cell<T,LATTICE_DESCRIPTOR>::getFirstMomentY(const T* const population) const {
    // Local constants for easier access
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    T m1y = 0.0;
    for (unsigned int l = 0; l < Q; ++l) { m1y += population[l] * static_cast<T>(c<D,Q>(l, 1)); }
    return m1y;
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ T Cell<T,LATTICE_DESCRIPTOR>::getVelocityX(const T* const population) const {
    // Local constants for easier access
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    T m1x = 0.0;
    T rho = 0.0;
    for (unsigned int l = 0; l < Q; ++l) { m1x += population[l] * static_cast<T>(c<D,Q>(l, 0)); rho += population[l]; }
    return m1x / rho;
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ T Cell<T,LATTICE_DESCRIPTOR>::getVelocityX(const T* const population, T R) const {
    // Local constants for easier access
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    T m1x = 0.0;
    for (unsigned int l = 0; l < Q; ++l) { m1x += population[l] * static_cast<T>(c<D,Q>(l, 0)); }
    return m1x / R;
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ T Cell<T,LATTICE_DESCRIPTOR>::getVelocityY(const T* const population) const {
    // Local constants for easier access
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    T m1x = 0.0;
    T rho = 0.0;
    for (unsigned int l = 0; l < Q; ++l) { m1x += population[l] * static_cast<T>(c<D,Q>(l, 1)); rho += population[l]; }
    return m1x / rho;
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ T Cell<T,LATTICE_DESCRIPTOR>::getVelocityY(const T* const population, T R) const {
    // Local constants for easier access
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    T m1x = 0.0;
    for (unsigned int l = 0; l < Q; ++l) { m1x += population[l] * static_cast<T>(c<D,Q>(l, 1)); }
    return m1x / R;
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ void Cell<T,LATTICE_DESCRIPTOR>::getEquilibriumDistribution(const T* const population, T* eqDistr, T R, T U, T V) const {
    // Local constants for easier access
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    for (unsigned int l = 0; l < Q; ++l)
    {
		eqDistr[l] = computeEquilibriumPopulation(l, R, U, V);
	}
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ void Cell<T,LATTICE_DESCRIPTOR>::setEquilibriumDistribution(T* population, T R, T U, T V) const {
    // Local constants for easier access
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    for (unsigned int l = 0; l < Q; ++l)
    {
		population[l] = computeEquilibriumPopulation(l, R, U, V);
	}
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ void Cell<T,LATTICE_DESCRIPTOR>::computePostCollisionDistributionBGK(T* population, const CollisionParamsBGK<T>*const params, T R, T U, T V) const {
    // Local constants for easier access
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    for (unsigned int l = 0; l < Q; ++l) {
        population[l] -= params->omegaShear * (population[l] - computeEquilibriumPopulation(l, R, U, V));// - Fext[l];	
	}
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ void Cell<T,LATTICE_DESCRIPTOR>::computePostCollisionDistributionCHM(T* population, const CollisionParamsCHM<T>*const params, T R, T U, T V) const {
    // Local constants for easier access
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

/****************************************************************************************************************************************/
/**** This implementation is specific to a predefined velocity set. Changing the velocity set would break the CHM collision operator ****/
/****************************************************************************************************************************************/

    // Raw moments
    T rm0, rm1, rm2, rm3, rm4, rm5, rm6, rm7, rm8;

    // Populations
    T f0=population[0], f1=population[1], f2=population[2], f3=population[3], f4=population[4], f5=population[5], f6=population[6], f7=population[7], f8=population[8];

    // Force
    T Fx = 0.0;
    T Fy = 0.0;

    // Central-hermite moments
    T chm3 = f0*(U*U + V*V - 2.0 * cs2<T,D,Q>()) + f1*((U-1.0)*(U-1.0) + V*V - 2.0 * cs2<T,D,Q>()) + f2*((U-1.0)*(U-1.0) + (V-1.0)*(V-1.0) - 2.0 * cs2<T,D,Q>()) + f3*(U*U + (V-1.0)*(V-1.0) - 2.0 * cs2<T,D,Q>()) + f4*((U+1.0)*(U+1.0) + (V-1.0)*(V-1.0) - 2.0 * cs2<T,D,Q>()) + f5*((U-1.0)*(U-1.0) + (V+1.0)*(V+1.0) - 2.0 * cs2<T,D,Q>()) + f6*(U*U + (V+1.0)*(V+1.0) - 2.0 * cs2<T,D,Q>()) + f7*((U+1.0)*(U+1.0) + (V+1.0)*(V+1.0) - 2.0 * cs2<T,D,Q>()) + f8*((U+1)*(U+1) + V*V - 2.0 * cs2<T,D,Q>());

    T chm4 = f0*(U*U - V*V) - f1*(-(U-1.0)*(U-1.0) + V*V) + f2*((U-1.0)*(U-1.0) - (V-1.0)*(V-1.0)) + f3*(U*U - (V-1.0)*(V-1.0)) + f4*((U+1.0)*(U+1.0) - (V-1.0)*(V-1.0)) + f5*((U-1.0)*(U-1.0) - (V+1.0)*(V+1.0)) + f6*(U*U - (V+1.0)*(V+1.0)) + f7*((U+1.0)*(U+1.0) - (V+1.0)*(V+1.0)) - f8*(-(U+1.0)*(U+1.0) + V*V);

    T chm5 = f0*U*V + f1*(U-1.0)*V + f2*(U-1.0)*(V-1.0) + f3*U*(V-1.0) + f4*(U+1.0)*(V-1.0) + f5*(U-1.0)*(V+1.0) + f6*U*(V+1.0) + f7*(U+1.0)*(V+1.0) + f8*(U+1.0)*V;

    // Compute raw moments from collision in contral-hermite moment space
    rm0 = R;
    rm1 = 0.5*Fx + U*R;
    rm2 = 0.5*Fy + V*R;
    rm3 = Fx*U + Fy*V + (U*U + V*V + 2.0/3.0)*R + (1.0 - params->omegaBulk)*chm3;
    rm4 = Fx*U - Fy*V + (1.0 - params->omegaShear)*chm4 + R*(U*U - V*V);
    rm5 = 0.5*(Fx*V + Fy*U) + U*V*R + (1.0 - params->omegaShear)*chm5;
    rm6 = Fx*U*V + Fy*(3.0*U*U + 1.0)/6.0 + 0.5*V*(1.0 - params->omegaBulk)*chm3 + 0.5*V*(1.0-params->omegaShear)*chm4 + 2.0*U*(1.0 - params->omegaShear)*chm5 + V*R*(3.0*U*U + 1.0)/3.0;
    rm7 = Fy*U*V + Fx*(3.0*V*V + 1.0)/6.0 + 0.5*U*(1.0 - params->omegaBulk)*chm3 - 0.5*U*(1.0-params->omegaShear)*chm4 + 2.0*V*(1.0 - params->omegaShear)*chm5 + U*R*(3.0*V*V + 1.0)/3.0;
    rm8 = (Fx*U*(3.0*V*V + 1.0) + Fy*V*(3.0*U*U + 1.0))/3.0 + 4.0*U*V*(1.0 - params->omegaShear)*chm5 + (1.0 - params->omegaBulk)*chm3*(3.0*U*U + 3.0*V*V + 2.0)/6.0 - 0.5*(1.0 - params->omegaShear)*chm4*(U*U - V*V) + R*(9.0*U*U*V*V + 3.0*U*U + 3.0*V*V + 1.0)/9.0;

    // Compute populations in real space from raw moments
	population[0] = rm0 - rm3 + rm8;	
	population[1] =  0.5*( rm1 + 0.5*(rm3 + rm4) - rm7 - rm8);
    population[2] = 0.25*(rm5  + rm6 + rm7 + rm8);
    population[3] =  0.5*( rm2 + 0.5*(rm3 - rm4) - rm6 - rm8);
    population[4] = 0.25*(-rm5 + rm6 - rm7 + rm8);
    population[5] = 0.25*(-rm5 - rm6 + rm7 + rm8);
    population[6] =  0.5*(-rm2 + 0.5*(rm3 - rm4) + rm6 - rm8);
	population[7] = 0.25*(rm5  - rm6 - rm7 + rm8);
	population[8] =  0.5*(-rm1 + 0.5*(rm3 + rm4) + rm7 - rm8);
}

#endif // CELL_HH
