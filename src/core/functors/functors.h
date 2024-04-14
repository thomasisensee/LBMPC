#ifndef FUNCTORS_H
#define FUNCTORS_H

namespace functors {

template<typename T,typename LATTICE_DESCRIPTOR>
class StandardEquilibrium {
private:
    T _R, _U, _V;
public:
    __device__ StandardEquilibrium(T* population);
    __device__ T operator()(unsigned int l) const;
};

} // namespace functors

#include "functors.hh"

#endif // FUNCTORS_H