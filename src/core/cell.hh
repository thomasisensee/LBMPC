#ifndef Cell_HH
#define Cell_HH

#include "cell.h"


template<typename T>
__device__ T Cell<T>::computeEquilibriumPopulation(unsigned int l, const LBParams<T>*const params, T R, T U, T V) const {
    int cix = params->LATTICE_VELOCITIES[l * params->D];
    int ciy = params->LATTICE_VELOCITIES[l * params->D+1];
    T cixcs2 = cix * cix - C_S_POW2;
    T ciycs2 = ciy * ciy - C_S_POW2;
    T firstOrder = C_S_POW2_INV * (U * cix + V * ciy);
    T secondOrder = 0.5 * C_S_POW4_INV * (cixcs2 * U * U + ciycs2 * V * V + 2.0 * cix * ciy * U * V);
    T thirdOrder = 0.5 * C_S_POW6_INV * (cixcs2 * ciy * U * U * V + ciycs2 * cix * U * V * V);
    T fourthOrder = 0.25 * C_S_POW8_INV * (cixcs2 * ciycs2 * U * U * V * V);

    return params->LATTICE_WEIGHTS[l] * R * (1.0 + firstOrder + secondOrder + thirdOrder + fourthOrder);
}

template<typename T>
__device__ T Cell<T>::getZeroMoment(const T*const population, const LBParams<T>*const params) const {
    T rho = 0.0;
    for (int l = 0; l < params->Q; ++l) { rho += population[l]; }
    return rho;
}

template<typename T>
__device__ T Cell<T>::getFirstMomentX(const T*const population, const LBParams<T>*const params) const {
    T m1x = 0.0;
    for (int l = 0; l < params->Q; ++l) { m1x += population[l] * params->LATTICE_VELOCITIES[l * params->D]; }
    return m1x;
}

template<typename T>
__device__ T Cell<T>::getFirstMomentY(const T*const population, const LBParams<T>*const params) const {
    T m1x = 0.0;
    for (int l = 0; l < params->Q; ++l) { m1x += population[l] * params->LATTICE_VELOCITIES[l * params->D + 1]; }
    return m1x;
}

template<typename T>
__device__ T Cell<T>::getVelocityX(const T*const population, const LBParams<T>*const params) const {
    T m1x = 0.0;
    for (int l = 0; l < params->Q; ++l) { m1x += population[l] * params->LATTICE_VELOCITIES[l * params->D]; }
    return m1x / getZeroMoment(population, params);
}

template<typename T>
__device__ T Cell<T>::getVelocityX(const T*const population, const LBParams<T>*const params, T R) const {
    T m1x = 0.0;
    for (int l = 0; l < params->Q; ++l) { m1x += population[l] * params->LATTICE_VELOCITIES[l * params->D]; }
    return m1x / R;
}

template<typename T>
__device__ T Cell<T>::getVelocityY(const T*const population, const LBParams<T>*const params) const {
    T m1x = 0.0;
    for (int l = 0; l < params->Q; ++l) { m1x += population[l] * params->LATTICE_VELOCITIES[l * params->D + 1]; }
    return m1x / getZeroMoment(population, params);
}

template<typename T>
__device__ T Cell<T>::getVelocityY(const T*const population, const LBParams<T>*const params, T R) const {
    T m1x = 0.0;
    for (int l = 0; l < params->Q; ++l) { m1x += population[l] * params->LATTICE_VELOCITIES[l * params->D + 1]; }
    return m1x / R;
}

template<typename T>
__device__ void Cell<T>::getEquilibriumDistribution(const T*const population, T* eqDistr, const LBParams<T>*const params, T R, T U, T V) const {
    for (int l = 0; l < params->Q; ++l)
    {
		eqDistr[l] = computeEquilibriumPopulation(l, params, R, U, V);
	}
}

template<typename T>
__device__ void Cell<T>::setEquilibriumDistribution(T* population, const LBParams<T>*const params, T R, T U, T V) const {
    for (int l = 0; l < params->Q; ++l)
    {
		population[l] = computeEquilibriumPopulation(l, params, R, U, V);
	}
}

template<typename T>
__device__ void Cell<T>::computePostCollisionDistributionBGK(T* population, const CollisionParamsBGK<T>*const params, T R, T U, T V) const {
    for (int l = 0; l < params->Q; ++l) {
        population[l] -= params->omegaShear * (population[l] - computeEquilibriumPopulation(l, params, R, U, V));// - Fext[l];	
	}
}

template<typename T>
__device__ void Cell<T>::computePostCollisionDistributionCHM(T* population, const CollisionParamsCHM<T>*const params, T R, T U, T V) const {
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
    T chm3 = f0*(U*U + V*V - 2.0*C_S_POW2) + f1*((U-1.0)*(U-1.0) + V*V - 2.0*C_S_POW2) + f2*((U+1.0)*(U+1.0) + V*V - 2.0*C_S_POW2) + f3*(U*U + (V-1.0)*(V-1.0) - 2.0*C_S_POW2) + f4*(U*U + (V+1.0)*(V+1.0) - 2.0*C_S_POW2) + f5*((U-1.0)*(U-1.0) + (V-1.0)*(V-1.0) - 2.0*C_S_POW2) + f6*((U+1.0)*(U+1.0) + (V+1.0)*(V+1.0) - 2.0*C_S_POW2) + f7*((U-1.0)*(U-1.0) + (V+1.0)*(V+1.0) - 2.0*C_S_POW2) + f8*((U+1)*(U+1) +(V-1.0)*(V-1.0) - 2.0*C_S_POW2);

    T chm4 = f0*(U*U - V*V) - f1*(-(U-1.0)*(U-1.0) + V*V) - f2*(-(U+1.0)*(U+1.0) + V*V) + f3*(U*U - (V-1.0)*(V-1.0)) + f4*(U*U - (V+1.0)*(V+1.0)) + f5*((U-1.0)*(U-1.0) - (V-1.0)*(V-1.0)) +f6*((U+1.0)*(U+1.0) - (V+1.0)*(V+1.0)) + f7*((U-1.0)*(U-1.0) - (V+1.0)*(V+1.0)) + f8*((U+1.0)*(U+1.0) - (V-1.0)*(V-1.0));

    T chm5 = f0*U*V + f1*(U-1.0)*V + f2*(U+1.0)*V + f3*U*(V-1.0) + f4*U*(V+1.0) + f5*(U-1.0)*(V-1.0) + f6*(U+1.0)*(V+1.0) + f7*(U-1.0)*(V+1.0) + f8*(U+1.0)*(V-1.0);

    // Compute raw moments from collision in contral-hermite moment space
    rm0 = R;
    rm1 = 0.5*Fx + U*R;
    rm2 = 0.5*Fy + V*R;
    rm3 = Fx*U + Fy*V + (U*U + V*V + 2.0/3.0)*R + (1.0-params->omegaBulk)*chm3;
    rm4 = Fx*U - Fy*V + (1.0-params->omegaShear)*chm4 + R*(U*U - V*V);
    rm5 = 0.5*(Fx*V + Fy*U) + U*V*R + (1.0-params->omegaShear)*chm5;
    rm6 = Fx*U*V + Fy*(3.0*U*U+1.0)/6.0 + 0.5*V*(1.0-params->omegaBulk)*chm3 + 0.5*V*(1.0-params->omegaShear)*chm4 + 2.0*U*(1.0-params->omegaShear)*chm5 + V*R*(3.0*U*U+1.0)/3.0;
    rm7 = Fy*U*V + Fx*(3.0*V*V+1.0)/6.0 + 0.5*U*(1.0-params->omegaBulk)*chm3 - 0.5*U*(1.0-params->omegaShear)*chm4 + 2.0*V*(1.0-params->omegaShear)*chm5 + U*R*(3.0*V*V+1.0)/3.0;
    rm8 = (Fx*U*(3.0*V*V+1.0) + Fy*V*(3.0*U*U+1.0))/3.0 + 4.0*U*V*(1.0-params->omegaShear)*chm5 + (1.0-params->omegaBulk)*chm3*(3.0*U*U+3.0*V*V+2.0)/6.0 - 0.5*(1.0-params->omegaShear)*chm4*(U*U-V*V) + R*(9.0*U*U*V*V+3.0*U*U+3.0*V*V+1.0)/9.0;

    // Compute populations in real space from raw moments
	population[0] = rm0 - rm3 + rm8;	
	population[1] =  0.5*( rm1 + 0.5*(rm3 + rm4) - rm7 - rm8);	
	population[2] =  0.5*(-rm1 + 0.5*(rm3 + rm4) + rm7 - rm8);
	population[3] =  0.5*( rm2 + 0.5*(rm3 - rm4) - rm6 - rm8);
	population[4] =  0.5*(-rm2 + 0.5*(rm3 - rm4) + rm6 - rm8);
	population[5] = 0.25*(rm5  + rm6 + rm7 + rm8);
	population[6] = 0.25*(rm5  - rm6 - rm7 + rm8);
	population[7] = 0.25*(-rm5 - rm6 + rm7 + rm8);
	population[8] = 0.25*(-rm5 + rm6 - rm7 + rm8);
}

#endif // CELL_HH
