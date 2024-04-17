#ifndef DESCRIPTOR_ALIASES_H
#define DESCRIPTOR_ALIASES_h

#include "core/descriptors/descriptors.h"
#include "core/descriptors/latticeDescriptors.h"
#include "core/functors/functors.h"

namespace descriptors {

    template<typename T>
    using StandardD2Q9 = DESCRIPTOR<D2Q9,MomentumConservation>;

    template<typename T>
    using ScalarD2Q9 = DESCRIPTOR<D2Q9,EnergyConservation>;

    template<typename T>
    using ScalarD2Q5 = DESCRIPTOR<D2Q5,EnergyConservation>;

    template<typename T>
    using ScalarAdvectionD2Q9 = DESCRIPTOR<D2Q9,EnergyConservation>;

    template<typename T>
    using ScalarD2Q5 = DESCRIPTOR<D2Q5,EnergyConservation>;

} // namespace descriptors

#endif // DESCRIPTOR_ALIASES_H