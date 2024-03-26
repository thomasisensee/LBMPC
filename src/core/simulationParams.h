#ifndef SIMULATION_PARAMS_H
#define SIMULATION_PARAMS_H

template<typename T>
struct BaseParams {
    /// Grid
    unsigned int Nx;
    unsigned int Ny;    
    
    /// Virtual destructor if using dynamic polymorphism
    virtual ~BaseParams() = default;
};

template<typename T>
struct FluidParams : public BaseParams<T> {
    // LBM
    unsigned int Q;
    T omegaShear;
    int *LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;    
};

template<typename T>
struct ThermalParams : public BaseParams<T> {
    // LBM
    unsigned int Q;
    T omegaDiffusion;
    int *LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;    
};

#endif // SIMULATION_PARAMS_H
