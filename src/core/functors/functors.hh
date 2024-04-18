#ifndef FUNCTORS_HH
#define FUNCTORS_HH

#include "functors.h"
#include "cuda/cell.h"



namespace functors {

    template<typename T,typename DESCRIPTOR>
    __device__ StandardEquilibrium<T,DESCRIPTOR>::StandardEquilibrium(T* population) {
        Cell<T,DESCRIPTOR> cell;
        _R = cell.getZerothMoment(population);
        _U = cell.getVelocityX(population, _R);
        _V = cell.getVelocityY(population, _R);
    }

    template<typename T,typename DESCRIPTOR>
    __device__ T StandardEquilibrium<T,DESCRIPTOR>::operator()(unsigned int l) const {
        Cell<T,DESCRIPTOR> cell;

        return cell.computeEquilibriumPopulation(l, _R, _U, _V);
    }

    template<typename T,typename DESCRIPTOR>
    __device__ ScalarEquilibrium<T,DESCRIPTOR>::ScalarEquilibrium(T* population) {
        Cell<T,DESCRIPTOR> cell;
        _R = cell.getZerothMoment(population);
        _U = 0.0;
        _V = 0.0;
    }

    template<typename T,typename DESCRIPTOR>
    __device__ ScalarEquilibrium<T,DESCRIPTOR>::ScalarEquilibrium(T* population, T U, T V) {
        Cell<T,DESCRIPTOR> cell;
        _R = cell.getZerothMoment(population);
        _U = U;
        _V = V;
    }

    template<typename T,typename DESCRIPTOR>
    __device__ T ScalarEquilibrium<T,DESCRIPTOR>::operator()(unsigned int l) const {
        Cell<T,DESCRIPTOR> cell;

        return cell.computeEquilibriumPopulation(l, _R, _U, _V);
    }

    template<typename T,typename DESCRIPTOR>
    __device__ BounceBack<T,DESCRIPTOR>::BounceBack() {}

    template<typename T,typename DESCRIPTOR>
    __device__ T BounceBack<T,DESCRIPTOR>::operator()(T* collision, const BounceBackParams<T>* const params, unsigned int i, unsigned int j) {
        // Local constants for easier access
        constexpr unsigned int D = DESCRIPTOR::LATTICE::D;
        constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

        unsigned int idx, idxNeighbor, iPop, iPopRev;
        int cix, ciy;
        Cell<T,DESCRIPTOR> cell;
        T R;

        idx = pos(i, j, params->Nx);

        for (unsigned int l = 0; l < 3; ++l) {
            iPop = descriptors::b<D,Q>(static_cast<unsigned int>(params->location), l);
            iPopRev = Q - iPop;
            cix = static_cast<T>(descriptors::c<D,Q>(iPop, 0));
            ciy = static_cast<T>(descriptors::c<D,Q>(iPop, 1));

            // Check if the neighbor is outside the domain (i.e., a ghost cell)
            if (static_cast<int>(i) + cix < 1 || static_cast<int>(i) + cix > params->Nx - 2 || static_cast<int>(j) + ciy < 1 || static_cast<int>(j) + ciy > params->Ny - 2) { continue; }

            // Compute the index of the neighbor
            idxNeighbor = pos(static_cast<int>(i) + cix, static_cast<int>(j) + ciy, params->Nx);

            // Apply the bounce-back boundary condition
            collision[idx * Q + iPop] = collision[idxNeighbor * Q + iPopRev];
        }
    }

    template<typename T,typename DESCRIPTOR>
    __device__ MovingWall<T,DESCRIPTOR>::MovingWall() {}

    template<typename T,typename DESCRIPTOR>
    __device__ T MovingWall<T,DESCRIPTOR>::operator()(T* collision, const MovingWallParams<T>* const params, unsigned int i, unsigned int j) {
        // Local constants for easier access
        constexpr unsigned int D = DESCRIPTOR::LATTICE::D;
        constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

        unsigned int idx, idxNeighbor, iPop, iPopRev;
        int cix, ciy;
        Cell<T,DESCRIPTOR> cell;
        T R;
        T uWall = 0.0, vWall = 0.0;
        T cixcs2 = 0.0, ciycs2 = 0.0;
        T firstOrder = 0.0, secondOrder = 0.0, thirdOrder = 0.0, fourthOrder = 0.0;

        idx = pos(i, j, params->Nx);

        for (unsigned int l = 0; l < 3; ++l) {
            iPop = descriptors::b<D,Q>(static_cast<unsigned int>(params->location), l);
            iPopRev = Q - iPop;
            cix = static_cast<T>(descriptors::c<D,Q>(iPop, 0));
            ciy = static_cast<T>(descriptors::c<D,Q>(iPop, 1));


            // Check if the neighbor is outside the domain (i.e., a ghost cell)
            if (static_cast<int>(i) + cix < 1 || static_cast<int>(i) + cix > params->Nx - 2 || static_cast<int>(j) + ciy < 1 || static_cast<int>(j) + ciy > params->Ny - 2) { continue; }

            // Compute the index of the neighbor
            idxNeighbor = pos(static_cast<int>(i) + cix, static_cast<int>(j) + ciy, params->Nx);

            // Compute the dot product if WALL_VELOCITY is not null
            if (params->WALL_VELOCITY != nullptr) {
                uWall = params->WALL_VELOCITY[0];
                vWall = params->WALL_VELOCITY[1];
                cixcs2 = cix * cix - cs2<T,D,Q>();
                ciycs2 = ciy * ciy - cs2<T,D,Q>();
                firstOrder = descriptors::invCs2<T,D,Q>() * (uWall * cix + vWall * ciy);
                secondOrder = 0.5 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * uWall * uWall + ciycs2 * vWall * vWall + 2.0 * cix * ciy * uWall * vWall);
                thirdOrder = 0.5 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * ciy * uWall * uWall * vWall + ciycs2 * cix * uWall * vWall * vWall);
                fourthOrder = 0.25 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * ciycs2 * uWall * uWall * vWall * vWall);
            }

            R = cell.getZerothMoment(&collision[idxNeighbor * Q]);

            // Apply the bounce-back boundary condition
            collision[idx * Q + iPop] = collision[idxNeighbor * Q + iPopRev] + 2.0 * R * descriptors::w<T,D,Q>(iPop) * (firstOrder + secondOrder + thirdOrder + fourthOrder);
        }
    }

    template<typename T,typename DESCRIPTOR>
    __device__ AntiBounceBack<T,DESCRIPTOR>::AntiBounceBack() {}

    template<typename T,typename DESCRIPTOR>
    __device__ T AntiBounceBack<T,DESCRIPTOR>::operator()(T* collision, const AntiBounceBackParams<T>* const params, unsigned int i, unsigned int j) {
        // Local constants for easier access
        constexpr unsigned int D = DESCRIPTOR::LATTICE::D;
        constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

        unsigned int idx, idxNeighbor, iPop, iPopRev;
        int cix, ciy;
        Cell<T,DESCRIPTOR> cell;
        T R;

        idx = pos(i, j, params->Nx);

        for (unsigned int l = 0; l < 3; ++l) {
            iPop = descriptors::b<D,Q>(static_cast<unsigned int>(params->location), l);
            iPopRev = Q - iPop;
            cix = static_cast<T>(descriptors::c<D,Q>(iPop, 0));
            ciy = static_cast<T>(descriptors::c<D,Q>(iPop, 1));
            T uWall = 0.0, vWall = 0.0;
            T cixcs2 = 0.0, ciycs2 = 0.0;
            T firstOrder = 0.0, secondOrder = 0.0, thirdOrder = 0.0, fourthOrder = 0.0;

            // Check if the neighbor is outside the domain (i.e., a ghost cell)
            if (static_cast<int>(i) + cix < 1 || static_cast<int>(i) + cix > params->Nx - 2 || static_cast<int>(j) + ciy < 1 || static_cast<int>(j) + ciy > params->Ny - 2) { continue; }

            // Compute the index of the neighbor
            idxNeighbor = pos(static_cast<int>(i) + cix, static_cast<int>(j) + ciy, params->Nx);

            R = cell.getZerothMoment(&collision[idxNeighbor * Q]);

            // Apply the bounce-back boundary condition
            collision[idx * Q + iPop] = -collision[idxNeighbor * Q + iPopRev] + 2.0 * descriptors::w<T,D,Q>(iPop) * params->WALL_VALUE;
        }
    }

}

#endif // FUNCTORS_HH