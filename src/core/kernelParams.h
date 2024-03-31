#ifndef KERNEL_PARAMS_H
#define KERNEL_PARAMS_H

template<typename T>
struct BaseParams {
    /// Grid
    unsigned int D;
    unsigned int Nx;
    unsigned int Ny;
    
    /// Virtual destructor if using dynamic polymorphism
    virtual ~BaseParams() = default;
};

template<typename T>
struct LBParams : public BaseParams<T> {
    unsigned int Q;
    int *LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;
    /*
    ~LBParams() override {
        delete[] LATTICE_VELOCITIES; // Derived-specific cleanup
        // BaseParams' destructor is automatically called after
    }*/
};

template<typename T>
struct CollisionParamsBGK : public LBParams<T> {
    unsigned int Q;
    int *LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;
    T omegaShear;
};

template<typename T>
struct CollisionParamsCHM : public CollisionParamsBGK<T> {
    unsigned int Q;
    int *LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;
    T omegaShear;
    T omegaBulk;
};

template<typename T>
struct BoundaryParams : public BaseParams<T> {
    unsigned int Q;
    int *LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;
    T* wallVelocity;
};

#endif // KERNEL_PARAMS_H
