#ifndef FUNCTORS_HH
#define FUNCTORS_HH

#include "functors.h"
#include "cuda/cell.h"

namespace functors {

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ StandardEquilibrium<T,LATTICE_DESCRIPTOR>::StandardEquilibrium(T* population) {
    Cell<T,LATTICE_DESCRIPTOR> cell;
    _R = cell.getZerothMoment(population);
    _U = cell.getVelocityX(population, _R);
    _V = cell.getVelocityY(population, _R);
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ T StandardEquilibrium<T,LATTICE_DESCRIPTOR>::operator()(unsigned int l) const {
    // Local constants for easier access
/*
    using namespace descriptors;
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;
*/

    Cell<T,LATTICE_DESCRIPTOR> cell;

    return cell.computeEquilibriumPopulation(l, _R, _U, _V);
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ ScalarEquilibrium<T,LATTICE_DESCRIPTOR>::ScalarEquilibrium(T* population) {
    Cell<T,LATTICE_DESCRIPTOR> cell;
    _R = cell.getZerothMoment(population);
    _U = 0.0;
    _V = 0.0;
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ ScalarEquilibrium<T,LATTICE_DESCRIPTOR>::ScalarEquilibrium(T* population, T U, T V) {
    Cell<T,LATTICE_DESCRIPTOR> cell;
    _R = cell.getZerothMoment(population);
    _U = U;
    _V = V;
}

template<typename T,typename LATTICE_DESCRIPTOR>
__device__ T ScalarEquilibrium<T,LATTICE_DESCRIPTOR>::operator()(unsigned int l) const {
    // Local constants for easier access
/*
    using namespace descriptors;
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;
*/
    Cell<T,LATTICE_DESCRIPTOR> cell;

    return cell.computeEquilibriumPopulation(l, _R, _U, _V);

/*
    T cix = static_cast<T>(c<D,Q>(l, 0));
    T ciy = static_cast<T>(c<D,Q>(l, 1));
    T cixcs2 = cix * cix - cs2<T,D,Q>();
    T ciycs2 = ciy * ciy - cs2<T,D,Q>();
    T firstOrder = invCs2<T,D,Q>() * (_U * cix + _V * ciy);
    T secondOrder = 0.5 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * _U * _U + ciycs2 * _V * _V + 2.0 * cix * ciy * _U * _V);
    T thirdOrder = 0.5 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * ciy * _U * _U * _V + ciycs2 * cix * _U * _V * _V);
    T fourthOrder = 0.25 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * ciycs2 * _U * _U * _V * _V);

    return w<T,D,Q>(l) * _R * (1.0 + firstOrder + secondOrder + thirdOrder + fourthOrder);
*/
}

}

#endif // FUNCTORS_HH