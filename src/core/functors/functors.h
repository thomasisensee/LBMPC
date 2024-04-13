#ifndef FUNCTORS_H
#define FUNCTORS_H

namespace functors {

template<typename T,typename LATTICE_DESCRIPTOR>
class StandardEquilibrium {
public:
    void operator()(T* population, T R, T U, T V) const;
};

} // namespace functors

#include "functors.hh"

#endif // FUNCTORS_H