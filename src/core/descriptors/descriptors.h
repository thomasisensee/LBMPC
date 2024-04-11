#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#ifdef __CUDA_ARCH__
    #define platform_constant constexpr
#else
    #define platform_constant static constexpr
#endif

template<typename T>
struct D2Q9Descriptor {
    platform_constant unsigned int D = 2;
    platform_constant unsigned int Q = 9;
    platform_constant int latticeVelocities[9][2] = {{1, 0}, {0, 1}, {1, 1}, {-1, 1}, {0, 0}, {1, -1}, {-1, -1}, {0, -1}, {-1, 0}};
    platform_constant T latticeWeights[9] = {1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 4.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/9.0, 1.0/9.0};
    platform_constant unsigned int boundaryMapping[4][3] = {{0, 2, 5}, {1, 3, 6}, {3, 5, 7}, {2, 6, 8}};
};

template<typename T>
struct FlowOnlyDescriptor {
    using LatticeDescriptor = D2Q9Descriptor<T>;
    // Flow-only specific parameters
};

#endif // DESCRIPTORS_H