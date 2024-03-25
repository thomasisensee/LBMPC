#ifndef LBM_COLLISION_MODEL_HH
#define LBM_COLLISION_MODEL_HH

#include "collisionModel.h"

template<typename T>
void BGKCollisionModel<T>::doCollision(T* distribution, T* equilibriumDistribution)
{
/*
    for (size_t i = 0; i < 9; ++i)// needs to get Q(9) from somewhere
    {
        distribution[i] += omegaShear*(equilibriumDistribution[i] - distribution[i]);
    }
*/
}

template<typename T>
void MRTCHMCollisionModel<T>::doCollision(T* distribution, const CollisionProperties& properties)
{

}


#endif
