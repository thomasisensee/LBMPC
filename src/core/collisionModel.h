#ifndef LBM_COLLISION_MODEL_H
#define LBM_COLLISION_MODEL_H

#include <stdio.h>
#include <vector>

template<typename T>
struct CollisionProperties {
    T density;
    std::vector<T> velocity;
    std::vector<T> force;
    CollisionProperties(const std::vector<T>& velocity, T density, const std::vector<T>& force) : velocity(velocity), density(density), force(force) {}
};

template<typename T>
class CollisionModel {
protected:
    /// Relaxation parameter associated with shear viscosity
    T omegaShear;
public:
    /// Destructor
    virtual ~CollisionModel() = default;
    void setOmegaShear(T omegaShear);
    T getOmegaShear() const;
    virtual void doCollision(T* distribution, const CollisionProperties<T>& properties) = 0;
    virtual void print() = 0;
};

template<typename T>
class BGKCollisionModel : public CollisionModel<T> {
public:
    virtual void doCollision(T* distribution, const CollisionProperties<T>& properties) override;
    void doCollision(T* distribution, const CollisionProperties<T>& properties, T* equilibriumDistribution);
    virtual void print();
};

template<typename T>
class MRTCHMCollisionModel : public CollisionModel<T> { // only implemented for D2Q9 lattices
protected:
    /// Relaxation parameter associated with bulk viscosity
    T omegaBulk;
public:
     void setOmegaBulk(T omegaBulk);
    T getOmegaBulk() const;
    virtual void doCollision(T* distribution, const CollisionProperties<T>& properties) override;
    virtual void print() override;
};

#include "collisionModel.hh"

#endif
