template<typename T>
struct D2Q9Descriptor {
    static constexpr unsigned int D = 2;
    static constexpr unsigned int Q = 9;
    static constexpr T latticeWeights[Q] = {1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 4.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/9.0, 1.0/9.0};
    static constexpr int latticeVelocities[Q][D] = {{1, 0}, {0, 1}, {1, 1}, {-1, 1}, {0, 0}, {1, -1}, {-1, -1}, {0, -1}, {-1, 0}};
    static constexpr unsigned int boundaryMapping[4][3] = {{0 ,2, 5}, {1, 3, 6}, {3, 5, 7}, {2, 6, 8}};
};

template<typename T>
struct FlowOnlyDescriptor {
    using LatticeDescriptor = D2Q9Descriptor<T>;
    // Flow-only specific parameters
};

template<typename T>
struct CoupledHeatFlowDescriptor {
    using LatticeDescriptor = D2Q9Descriptor<T>;
    // Parameters for coupled heat and flow simulation
};