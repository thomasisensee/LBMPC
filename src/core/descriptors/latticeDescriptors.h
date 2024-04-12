#ifndef LATTICE_DESCRIPTORS_H
#define LATTICE_DESCRIPTORS_H

#include "utilities/fraction.h"

/// Define preprocessor macros for conditional compilation of device-side functions and constant storage
#ifdef __CUDACC__
  #define any_platform __device__ __host__
  #ifdef __CUDA_ARCH__
    #define platform_constant constexpr __constant__
  #else
    #define platform_constant constexpr
  #endif
#else
  #define any_platform
  #define platform_constant constexpr
#endif


namespace latticeDescriptors {

    struct LATTICE_DESCRIPTOR_BASE { };

    /// Base descriptor of a D-dimensional lattice with Q directions and a list of additional fields
    template <unsigned int dim, unsigned int q>
    struct LATTICE_DESCRIPTOR : public LATTICE_DESCRIPTOR_BASE {

        /// Number of dimensions
        static constexpr int D = dim;

        /// Number of velocities
        static constexpr int Q = q;
    };

    /// D2Q9 lattice descriptor
    struct D2Q9 : public LATTICE_DESCRIPTOR<2,9> {
        D2Q9() = delete; // Deleted default constructor prevents instantiation
    };

    /// D2Q5 lattice descriptor
    struct D2Q5 : public LATTICE_DESCRIPTOR<2,5> {
        D2Q5() = delete; // Deleted default constructor prevents instantiation
    };

    namespace data {

        using utilities::Fraction;

        template <unsigned int D, unsigned int Q>
        platform_constant int latticeVelocities[Q][D] = {};

        template <unsigned int D, unsigned int Q>
        platform_constant Fraction latticeWeights[Q] = {};

        template <unsigned int D, unsigned int Q>
        platform_constant Fraction cs2 = {};

        template <unsigned int D, unsigned int Q>
        platform_constant unsigned int boundaryMapping[Q][D] = {};

        /// Specializations for D2Q9
        template <>
        platform_constant int latticeVelocities<2,9>[9][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};

        template <>
        platform_constant Fraction latticeWeights<2,9>[9] = {{4, 9}, {1, 9}, {1, 36}, {1, 9}, {1, 36}, {1, 9}, {1, 36}, {1, 9}, {1, 36}};

        template <>
        platform_constant Fraction cs2<2,9> = {1, 3};

        template <>
        platform_constant unsigned int boundaryMapping<2,9>[4][3] = {{1, 2, 5}, {4, 7, 8}, {2, 3, 4}, {1, 2, 5}};

        /// Specializations for D2Q5
        template <>
        platform_constant int latticeVelocities<2,5>[5][2] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}};

        template <>
        platform_constant Fraction latticeWeights<2,5>[5] = {{1, 3}, {1, 6}, {1, 6}, {1, 6}, {1, 6}};

        template <>
        platform_constant Fraction cs2<2,5> = {1, 3};

        template <>
        platform_constant unsigned int boundaryMapping<2,5>[4][1] = {{1}, {3}, {2}, {4}};

    } // namespace data

    /// Functions to access lattice descriptors
    template <unsigned int D, unsigned int Q>
    any_platform constexpr int latticeVelocities(unsigned int iPop, unsigned int iDim) {
        return data::latticeVelocities<D,Q>[iPop][iDim];
    }

    template <typename T, unsigned D, unsigned Q>
    any_platform constexpr T latticeWeights(unsigned iPop) {
        return data::latticeWeights<D,Q>[iPop].template as<T>();
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
    any_platform constexpr unsigned int boundaryMapping(unsigned int iBoundary, unsigned int iPop) {
        return data::boundaryMapping<D,Q>[iBoundary][iPop];
    }

    template <typename LATTICE_DESCRIPTOR>
    constexpr int D() any_platform {
        return LATTICE_DESCRIPTOR::D;
    }

    template <typename LATTICE_DESCRIPTOR>
    constexpr int Q() any_platform {
        return LATTICE_DESCRIPTOR::Q;
    }
} // namespace latticeDescriptors

#endif // LATTICE_DESCRIPTORS_H