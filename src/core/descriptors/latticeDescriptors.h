#ifndef LATTICE_DESCRIPTORS_H
#define LATTICE_DESCRIPTORS_H

#include "core/platform_definitions.h"
#include "utilities/fraction.h"

namespace descriptors {

    struct LATTICE_DESCRIPTOR_BASE {};

    /// Base descriptor of a D-dimensional lattice with Q directions and a list of additional fields
    template <unsigned int dim, unsigned int q, unsigned int npop>
    struct LATTICE_DESCRIPTOR : public LATTICE_DESCRIPTOR_BASE {
        LATTICE_DESCRIPTOR() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type

        /// Number of dimensions
        static constexpr int D = dim;

        /// Number of velocities
        static constexpr int Q = q;

        /// Number of populations per boundary
        static constexpr unsigned int nPop = npop;
    };

    /// D2Q9 lattice descriptor
    struct D2Q9 : public LATTICE_DESCRIPTOR<2,9,3> {
        D2Q9() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type
    };

    /// D2Q5 lattice descriptor
    struct D2Q5 : public LATTICE_DESCRIPTOR<2,5,1> {
        D2Q5() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type
    };

    namespace data {

        using utilities::Fraction;

        template <unsigned int D, unsigned int Q>
        platform_constant int c[Q][D] = {};

        template <unsigned int D, unsigned int Q>
        platform_constant Fraction w[Q] = {};

        template <unsigned int D, unsigned int Q>
        platform_constant Fraction cs2 = {};

        template <unsigned int D, unsigned int Q>
        platform_constant unsigned int b[Q][D] = {}; // Boundary mapping

        /// Specializations for D2Q9
        template <>
        platform_constant int c<2,9>[9][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}};

        template <>
        platform_constant Fraction w<2,9>[9] = {{4, 9}, {1, 9}, {1, 36}, {1, 9}, {1, 36}, {1, 36}, {1, 9}, {1, 36}, {1, 9}};

        template <>
        platform_constant Fraction cs2<2,9> = {1, 3};

        template <>
        platform_constant unsigned int b<2,9>[4][3] = {{1, 2, 5}, {4, 7, 8}, {2, 3, 4}, {5, 6, 7}};

        /// Specializations for D2Q5
        template <>
        platform_constant int c<2,5>[5][2] = {{0, 0}, {1, 0}, {0, 1}, {0, -1}, {-1, 0}};

        template <>
        platform_constant Fraction w<2,5>[5] = {{1, 3}, {1, 6}, {1, 6}, {1, 6}, {1, 6}};

        template <>
        platform_constant Fraction cs2<2,5> = {1, 3};

        template <>
        platform_constant unsigned int b<2,5>[4][1] = {{1}, {4}, {2}, {3}};

    } // namespace data

    /// Functions to access lattice descriptors
    template <unsigned int D, unsigned int Q>
    any_platform constexpr int c(unsigned int iPop, unsigned int iDim) {
        return data::c<D,Q>[iPop][iDim];
    }

    template <typename T, unsigned D, unsigned Q>
    any_platform constexpr T w(unsigned iPop) {
        return data::w<D,Q>[iPop].template as<T>();
    }

    template <typename T, unsigned D, unsigned Q>
    any_platform constexpr T cs2() {
        return data::cs2<D,Q>.template as<T>();
    }

    template <typename T, unsigned D, unsigned Q>
    any_platform constexpr T invCs2() {
        return data::cs2<D,Q>.template inverseAs<T>();
    }

    template <unsigned int D, unsigned int Q>
    any_platform constexpr unsigned int b(unsigned int iBoundary, unsigned int iPop) {
        return data::b<D,Q>[iBoundary][iPop];
    }
} // namespace latticeDescriptors

#endif // LATTICE_DESCRIPTORS_H