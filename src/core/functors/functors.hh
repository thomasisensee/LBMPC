#ifndef FUNCTORS_HH
#define FUNCTORS_HH

#include "functors.h"

namespace functors {

template<typename T,typename LATTICE_DESCRIPTOR>
void StandardEquilibrium<T,LATTICE_DESCRIPTOR>::operator()(T* population, T R, T U, T V) const {
    // Local constants for easier access
    using namespace descriptors;
    constexpr unsigned int D = LATTICE_DESCRIPTOR::D;
    constexpr unsigned int Q = LATTICE_DESCRIPTOR::Q;

    T cix, ciy, cixcs2, ciycs2, firstOrder, secondOrder, thirdOrder, fourthOrder;

    for (unsigned int l = 0; l < Q; ++l) {
        cix = static_cast<T>(c<D,Q>(l, 0));
        ciy = static_cast<T>(c<D,Q>(l, 1));
        cixcs2 = cix * cix - cs2<T,D,Q>();
        ciycs2 = ciy * ciy - cs2<T,D,Q>();
        firstOrder = invCs2<T,D,Q>() * (U * cix + V * ciy);
        secondOrder = 0.5 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * U * U + ciycs2 * V * V + 2.0 * cix * ciy * U * V);
        thirdOrder = 0.5 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * ciy * U * U * V + ciycs2 * cix * U * V * V);
        fourthOrder = 0.25 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * ciycs2 * U * U * V * V);

		population[l] = w<T,D,Q>(l) * R * (1.0 + firstOrder + secondOrder + thirdOrder + fourthOrder);
	}
}

}

#endif // FUNCTORS_HH