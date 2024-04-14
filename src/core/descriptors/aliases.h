#ifndef DESCRIPTOR_ALIASES_H
#define DESCRIPTOR_ALIASES_h

#include "core/descriptors/descriptors.h"
#include "core/descriptors/latticeDescriptors.h"
#include "core/functors/functors.h"

namespace descriptors {

    template<typename T>
    using D2Q9Standard = DESCRIPTOR<D2Q9,functors::StandardEquilibrium>;

    template<typename T>
    using D2Q9Scalar = DESCRIPTOR<D2Q9,functors::ScalarEquilibrium>;

    template<typename T>
    using D2Q5Standard = DESCRIPTOR<D2Q5,functors::ScalarEquilibrium>;

} // namespace descriptors

#endif // DESCRIPTOR_ALIASES_H