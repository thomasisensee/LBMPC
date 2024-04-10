#ifndef LB_MODEL_HH
#define LB_MODEL_HH

#include <iostream>
#include <vector>

#include "lbModel.h"
#include "core/utilities.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T, typename LatticeDescriptor>
LBModel<T, LatticeDescriptor>::LBModel() {}

template<typename T, typename LatticeDescriptor>
unsigned int LBModel<T, LatticeDescriptor>::getD() const {
    return LatticeDescriptor::D;
}

template<typename T, typename LatticeDescriptor>
unsigned int LBModel<T, LatticeDescriptor>::getQ() const {
    return LatticeDescriptor::Q;
}

template<typename T, typename LatticeDescriptor>
int LBModel<T, LatticeDescriptor>::getCX(unsigned int i) const  {
    return LatticeDescriptor::latticeVelocities[i][0];
}

template<typename T, typename LatticeDescriptor>
int LBModel<T, LatticeDescriptor>::getCY(unsigned int i) const  {
    return LatticeDescriptor::latticeVelocities[i][1];
}

template<typename T, typename LatticeDescriptor>
T LBModel<T, LatticeDescriptor>::getWEIGHT(unsigned int i) const  {
    return LatticeDescriptor::latticeWeights[i];
}

template<typename T, typename LatticeDescriptor>
void LBModel<T, LatticeDescriptor>::printParameters() const {
    std::cout << "============================== LB Model Details ==============================" << std::endl;
    std::cout << "==                                   D" << getD() << "Q" << getQ() << "                                    ==" << std::endl;
    std::cout << "== Cx ="; for (size_t i=0; i<LatticeDescriptor::Q; ++i) {std::cout << "\t" << getCX(i); } std::cout << "    ==" << std::endl;
    std::cout << "== Cy ="; for (size_t i=0; i<LatticeDescriptor::Q; ++i) {std::cout << "\t" << getCY(i); } std::cout << "   ==" << std::endl;
    std::cout << "== w  ="; for (size_t i=0; i<LatticeDescriptor::Q; ++i) {std::cout << "\t" << getWEIGHT(i); } std::cout << "   ==" << std::endl;
    std::cout << "===============================================================================\n" << std::endl;
}

#endif // LB_MODEL_HH
