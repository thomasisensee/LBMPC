#ifndef LB_MODEL_HH
#define LB_MODEL_HH

#include <iostream>

#include "lbModel.h"


/**********************/
/***** Base class *****/
/**********************/
template<typename T, unsigned int Dim, unsigned int Q>
LBModel<T, Dim, Q>::LBModel() {}

template<typename T, unsigned int Dim, unsigned int Q>
LBModel<T, Dim, Q>::~LBModel() {
    delete[] this->_LATTICE_VELOCITIES;
    delete[] this->_LATTICE_WEIGHTS;
    delete[] this->_OPPOSITE_POPULATION;
}

template<typename T, unsigned int Dim, unsigned int Q>
unsigned int LBModel<T, Dim, Q>::getD() const {
    return _D;
}

template<typename T, unsigned int Dim, unsigned int Q>
unsigned int LBModel<T, Dim, Q>::getQ() const {
    return _Q;
}

template<typename T, unsigned int Dim, unsigned int Q>
int* LBModel<T, Dim, Q>::getLatticeVelocitiesPtr() const {
    return _LATTICE_VELOCITIES;
}

template<typename T, unsigned int Dim, unsigned int Q>
T* LBModel<T, Dim, Q>::getLatticeWeightsPtr() const {
    return _LATTICE_WEIGHTS;
}

template<typename T, unsigned int Dim, unsigned int Q>
unsigned int* LBModel<T, Dim, Q>::getOppositePopualationPtr() const {
    return _OPPOSITE_POPULATION;
}

template<typename T, unsigned int Dim, unsigned int Q>
void LBModel<T, Dim, Q>::print() const {
    std::cout << "============================== LB Model Details ==============================" << std::endl;
    std::cout << "==                                   D" << getD() << "Q" << getQ() << "                                    ==" << std::endl;
    std::cout << "== Cx ="; for (int i=0; i<Q; ++i) {std::cout << "\t" << _LATTICE_VELOCITIES[i*Dim]; } std::cout << "    ==" << std::endl;
    std::cout << "== Cy ="; for (int i=0; i<Q; ++i) {std::cout << "\t" << _LATTICE_VELOCITIES[i*Dim+1]; } std::cout << "   ==" << std::endl;
    std::cout << "== w  ="; for (int i=0; i<Q; ++i) {std::cout << "\t" << _LATTICE_WEIGHTS[i]; } std::cout << "   ==" << std::endl;
    std::cout << "== op ="; for (int i=0; i<Q; ++i) {std::cout << "\t" << _OPPOSITE_POPULATION[i]; } std::cout << "   ==" << std::endl;
    std::cout << "===============================================================================\n" << std::endl;
}


/****************************************/
/***** Derived class 01: D2Q9 model *****/
/****************************************/
template<typename T>
D2Q9<T>::D2Q9() {
    this->_LATTICE_WEIGHTS = new T[9];
    this->_LATTICE_WEIGHTS[0] = 4.0/9.0;
    this->_LATTICE_WEIGHTS[1] = 1.0/9.0;
    this->_LATTICE_WEIGHTS[2] = 1.0/9.0;
    this->_LATTICE_WEIGHTS[3] = 1.0/9.0;
    this->_LATTICE_WEIGHTS[4] = 1.0/9.0;
    this->_LATTICE_WEIGHTS[5] = 1.0/36.0;
    this->_LATTICE_WEIGHTS[6] = 1.0/36.0;
    this->_LATTICE_WEIGHTS[7] = 1.0/36.0;
    this->_LATTICE_WEIGHTS[8] = 1.0/36.0;

    this->_LATTICE_VELOCITIES = new int[18];
    int velocities[9][2] = {{0, 0},{1, 0},{-1, 0},{0, 1},{0, -1},{1, 1},{-1, -1},{1, -1},{-1, 1}};
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 2; ++j) {
            this->_LATTICE_VELOCITIES[i*2+j] = velocities[i][j];
        }
    }

    // Initialize and compute opposites
    this->_OPPOSITE_POPULATION = new unsigned int[9];
    for (unsigned int i = 0; i < 9; ++i) {
        for (unsigned int j = 0; j < 9; ++j) {
            if (velocities[j][0] == -velocities[i][0] && velocities[j][1] == -velocities[i][1]) {
                this->_OPPOSITE_POPULATION[i] = j;
                break;
            }
        }
    }
}

template<typename T>
D2Q9<T>::~D2Q9() {}

template<typename T>
int D2Q9<T>::getCX(unsigned int i) const  {
    return this->_LATTICE_VELOCITIES[i*2];
}

template<typename T>
int D2Q9<T>::getCY(unsigned int i) const  {
    return this->_LATTICE_VELOCITIES[i*2+1];
}

template<typename T>
T D2Q9<T>::getWEIGHT(unsigned int i) const  {
    return this->_LATTICE_WEIGHTS[i];
}

template<typename T>
unsigned int D2Q9<T>::getOppositePopualation(unsigned int i) const  {
    return this->_OPPOSITE_POPULATION[i];
}

template<typename T>
LBModel<T, 2, 9>* D2Q9<T>::getDerivedModel() const {
    return new D2Q9<T>(*this); // Return a pointer to a new D2Q9 object
}

#endif // LB_MODEL_HH
