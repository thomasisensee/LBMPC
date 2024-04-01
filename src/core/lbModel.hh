#ifndef LB_MODEL_HH
#define LB_MODEL_HH

#include <iostream>

#include "lbModel.h"


/**********************/
/***** Base class *****/
/**********************/
template<typename T>
LBModel<T>::LBModel(unsigned int d, unsigned int q) : D(d), Q(q) {}

template<typename T>
LBModel<T>::~LBModel() {
    delete[] this->LATTICE_VELOCITIES;
    delete[] this->LATTICE_WEIGHTS;
}

template<typename T>
unsigned int LBModel<T>::getD() const {
    return D;
}

template<typename T>
unsigned int LBModel<T>::getQ() const {
    return Q;
}

template<typename T>
int* LBModel<T>::getLatticeVelocitiesPtr() const {
    return LATTICE_VELOCITIES;
}

template<typename T>
T* LBModel<T>::getLatticeWeightsPtr() const {
    return LATTICE_WEIGHTS;
}

template<typename T>
void LBModel<T>::print() const {
    std::cout << "============================== LB Model Details ==============================" << std::endl;
    std::cout << "==                                   D" << getD() << "Q" << getQ() << "                                    ==" << std::endl;
    std::cout << "== Cx ="; for (int i=0; i<Q; ++i) {std::cout << "\t" << LATTICE_VELOCITIES[i*D]; } std::cout << "    ==" << std::endl;
    std::cout << "== Cy ="; for (int i=0; i<Q; ++i) {std::cout << "\t" << LATTICE_VELOCITIES[i*D+1]; } std::cout << "   ==" << std::endl;
    std::cout << "== w  ="; for (int i=0; i<Q; ++i) {std::cout << "\t" << LATTICE_WEIGHTS[i]; } std::cout << "   ==" << std::endl;
    std::cout << "===============================================================================\n" << std::endl;
}


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
D2Q9<T>::D2Q9() : LBModel<T>(2, 9) {
    this->LATTICE_WEIGHTS = new T[9];
    this->LATTICE_WEIGHTS[0] = 4.0/9.0;
    this->LATTICE_WEIGHTS[1] = 1.0/9.0;
    this->LATTICE_WEIGHTS[2] = 1.0/9.0;
    this->LATTICE_WEIGHTS[3] = 1.0/9.0;
    this->LATTICE_WEIGHTS[4] = 1.0/9.0;
    this->LATTICE_WEIGHTS[5] = 1.0/36.0;
    this->LATTICE_WEIGHTS[6] = 1.0/36.0;
    this->LATTICE_WEIGHTS[7] = 1.0/36.0;
    this->LATTICE_WEIGHTS[8] = 1.0/36.0;

    this->LATTICE_VELOCITIES = new int[18];
    int velocities[9][2] = {{0, 0},{1, 0},{0, 1},{-1, 0},{0, -1},{1, 1},{-1, 1},{-1, -1},{1, -1}};
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 2; ++j) {
            this->LATTICE_VELOCITIES[i*this->D+j] = velocities[i][j];
        }
    }

    // Initialize and compute opposites
    this->OPPOSITE_POPULATION = new unsigned int[9];
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (velocities[j][0] == -velocities[i][0] && velocities[j][1] == -velocities[i][1]) {
                this->OPPOSITE_POPULATION[i] = j;
                break;
            }
        }
    }
}

/*
template<typename T>
D2Q9<T>::~D2Q9() {

}
*/
template<typename T>
int D2Q9<T>::getCX(unsigned int i) const  {
    return this->LATTICE_VELOCITIES[i*2];
}

template<typename T>
int D2Q9<T>::getCY(unsigned int i) const  {
    return this->LATTICE_VELOCITIES[i*2+1];
}

template<typename T>
T D2Q9<T>::getWEIGHT(unsigned int i) const  {
    return this->LATTICE_WEIGHTS[i];
}

template<typename T>
__host__ LBModel<T>* D2Q9<T>::getDerivedModel() const {
    return new D2Q9<T>(*this); // Return a pointer to a new D2Q9 object
}

#endif // LB_MODEL_HH