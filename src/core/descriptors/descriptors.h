#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include "core/platform_definitions.h"
#include "latticeDescriptors.h"
#include "fieldTags.h"
#include "core/functors/functors.h"

namespace descriptors {

    struct DESCRIPTOR_BASE {};

    template<typename LATTICE_DESCRIPTOR,template<typename, typename> typename EQUILIBRIUM_FUNCTOR>
    //template<typename LATTICE_DESCRIPTOR, typename... FieldTypes>
    struct DESCRIPTOR : public DESCRIPTOR_BASE {
        DESCRIPTOR() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type

        using LATTICE = LATTICE_DESCRIPTOR;

        template<typename T>
        using EQUILIBRIUM = EQUILIBRIUM_FUNCTOR<T,LATTICE_DESCRIPTOR>;
        //using FORCE = EXTERNAL_FORCE_FUNCTOR;
    };

} // namespace descriptors

#endif // DESCRIPTORS_H