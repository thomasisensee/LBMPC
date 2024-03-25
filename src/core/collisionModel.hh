#ifndef LBM_COLLISION_MODEL_HH
#define LBM_COLLISION_MODEL_HH

#include "collisionModel.h"

template<typename T>
void CollisionModel<T>::setOmegaShear(T omegaShear)
{
    this->omegaShear = omegaShear;
}

template<typename T>
T CollisionModel<T>::getOmegaShear() const
{
    return this->omegaShear;
}

template<typename T>
void BGKCollisionModel<T>::doCollision(T* distribution, const CollisionProperties<T>& properties)
{

}

template<typename T>
void BGKCollisionModel<T>::doCollision(T* distribution, const CollisionProperties<T>& properties, T* equilibriumDistribution)
{

}

template<typename T>
void BGKCollisionModel<T>::print()
{
    std::cout << "============= Collision Model: BGK ===============" << std::endl;
    std::cout << "== Omega shear:" << this->getOmegaShear() << "\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}

template<typename T>
void MRTCHMCollisionModel<T>::doCollision(T* distribution, const CollisionProperties<T>& properties)
{

}

template<typename T>
void MRTCHMCollisionModel<T>::setOmegaBulk(T omegaBulk)
{
    this->omegaBulk = omegaBulk;
}

template<typename T>
T MRTCHMCollisionModel<T>::getOmegaBulk() const
{
    return this->omegaBulk;
}

template<typename T>
void MRTCHMCollisionModel<T>::print()
{
    std::cout << "============= Collision Model: CHM ===============" << std::endl;
    std::cout << "== Omega shear:" << this->getOmegaShear() << "\t\t\t\t==" << std::endl;
    std::cout << "== Omega bulk:" << this->getOmegaBulk() << "\t\t\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}


#endif
