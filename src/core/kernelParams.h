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
struct LBMParams : public BaseParams<T> {
    // LBM
    unsigned int Q;
    int *LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;
};

template<typename T>
struct CollisionParamsBGK : public LBMParams<T> {
    // LBM
    unsigned int Q;
    int *LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;
    T omegaShear;
};

template<typename T>
struct CollisionParamsCHM : public CollisionParamsBGK<T> {
    // LBM
    unsigned int Q;
    int *LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;
    T omegaShear;
    T omegaBulk;
};

template<typename T>
struct BoundaryParams : public BaseParams<T> {
    // LBM
    unsigned int Q;
    int *LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;
};

#endif // KERNEL_PARAMS_H
