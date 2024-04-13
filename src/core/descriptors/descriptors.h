#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include "core/platform_definitions.h"
#include "latticeDescriptors.h"
#include "core/functors/functors.h"

namespace descriptors {

    struct DESCRIPTOR_BASE {};

    /// Base descriptor of a D-dimensional lattice with Q directions and a list of additional fields
    template<typename LATTICE_DESCRIPTOR, typename EQUILIBRIUM_FUNCTOR, typename EXTERNAL_FORCE_FUNCTOR>
    struct DESCRIPTOR : public DESCRIPTOR_BASE {
        using LatticeType = LATTICE_DESCRIPTOR;
        using EquilibriumType = EQUILIBRIUM_FUNCTOR;
        using ForceType = EXTERNAL_FORCE_FUNCTOR;

    };

} // namespace descriptors

#endif // DESCRIPTORS_H