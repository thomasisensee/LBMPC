#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include <tuple>

#include "core/platform_definitions.h"
#include "latticeDescriptors.h"
#include "fieldTags.h"
#include "core/functors/functors.h"

namespace descriptors {

    struct DESCRIPTOR_BASE {};

    template<typename LATTICE_DESCRIPTOR, typename... FIELD_TYPE>
    struct DESCRIPTOR : public DESCRIPTOR_BASE {
        DESCRIPTOR() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type

        using LATTICE = LATTICE_DESCRIPTOR;
        using FIELDS = std::tuple<FIELD_TYPE...>;

        //template<typename T>
        //using EQUILIBRIUM = EQUILIBRIUM_FUNCTOR<T,LATTICE_DESCRIPTOR>;
        //using FORCE = EXTERNAL_FORCE_FUNCTOR;
    };

    /// Contains checks if a type is in a parameter pack
    template<typename T, typename... List>
    struct Contains;

    template<typename T, typename First, typename... Rest>
    struct Contains<T, First, Rest...> : Contains<T, Rest...> {};

    template<typename T, typename... Rest>
    struct Contains<T, T, Rest...> : std::true_type {};

    template<typename T>
    struct Contains<T> : std::false_type {};


    /// Find the index of a type in a parameter pack
    template<typename T, typename Tuple>
    struct IndexOf;

    template<typename T, typename... Types>
    struct IndexOf<T, std::tuple<Types...>> {
        static const int value = [] {
            constexpr std::size_t n = sizeof...(Types);
            std::array<bool, n> matches = {std::is_same<T, Types>::value...};
            for (std::size_t i = 0; i < n; ++i) {
                if (matches[i]) return i;
            }
            return -1;
        }();
    };


    template<int IDX, typename T>
    __device__ T* get_field(T* first) {
        static_assert(IDX == 0, "Index out of range");
        return first;  // Base case: index 0, return the first element
    }

    template<int IDX, typename T, typename... Fields>
    __device__ T* get_field(T* first, Fields... rest) {
        if constexpr (IDX == 0) {
            return first;  // Base case: when index is 0, return the first element
        } else {
            return get_field<IDX - 1>(rest...);  // Recursive call peeling off the first element
        }
    }

} // namespace descriptors

#endif // DESCRIPTORS_H