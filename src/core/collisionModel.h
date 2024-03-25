#ifndef LBM_COLLISION_MODEL_H
#define LBM_COLLISION_MODEL_H

#include <stdio.h>

template<typename T>
struct CollisionProperties {
    T density;
    std::vector<T> velocity;
    std::vector<T> force;
    CollisionProperties(const std::vector<T>& velocity, Te density, const std::vector<T>& force) : velocity(velocity), density(density), force(force) {}
};

template<typename T>
class CollisionModel {
protected:
    /// Relaxation parameter associated with shear viscosity
    T omegaShear;
public:
    /// Destructor
    virtual ~CollisionModel() = default;
    virtual void doCollision(T* distribution) = 0;
};

class BGKCollisionModel : public CollisionModel {
public:
    void doCollision(T* distribution, T* equilibriumDistribution) override;
};

class MRTCHMCollisionModel : public CollisionModel { // only implemented for D2Q9 lattices
protected:
    /// Relaxation parameter associated with bulk viscosity
    T omegaBulk;
public:
    void doCollision(T* distribution, const CollisionProperties& properties) override;
};

#include "collisionModel.hh"

#endif
