#ifndef LB_MODEL_HH
#define LB_MODEL_HH

#include <iostream>
#include <vector>

#include "lbModel.h"
#include "core/utilities.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
LBModel<T>::LBModel(unsigned int d, unsigned int q) : _D(d), _Q(q) {}

template<typename T>
LBModel<T>::~LBModel() {
    delete[] this->_LATTICE_VELOCITIES;
    delete[] this->_LATTICE_WEIGHTS;
    delete[] this->_OPPOSITE_POPULATION;
}

template<typename T>
unsigned int LBModel<T>::getD() const {
    return _D;
}

template<typename T>
unsigned int LBModel<T>::getQ() const {
    return _Q;
}

template<typename T>
const int* LBModel<T>::getLatticeVelocitiesPtr() const {
    return _LATTICE_VELOCITIES;
}

template<typename T>
const T* LBModel<T>::getLatticeWeightsPtr() const {
    return _LATTICE_WEIGHTS;
}

template<typename T>
const unsigned int* LBModel<T>::getPopulationPtr(BoundaryLocation location) const {
// Check if the boundary location exists in the mapping and if the vector is not empty
    auto it = this->_BOUNDARY_MAPPING.find(location);
    if (it != this->_BOUNDARY_MAPPING.end() && !it->second.empty()) {
        return it->second.data();
    }
    return nullptr; // Return nullptr if the location does not exist or the vector is empty
}

template<typename T>
const unsigned int* LBModel<T>::getOppositePopualationPtr() const {
    return _OPPOSITE_POPULATION;
}

template<typename T>
void LBModel<T>::printParameters() const {
    std::cout << "============================== LB Model Details ==============================" << std::endl;
    std::cout << "==                                   D" << getD() << "Q" << getQ() << "                                    ==" << std::endl;
    std::cout << "== Cx ="; for (int i=0; i<this->_Q; ++i) {std::cout << "\t" << _LATTICE_VELOCITIES[i*this->_D]; } std::cout << "    ==" << std::endl;
    std::cout << "== Cy ="; for (int i=0; i<this->_Q; ++i) {std::cout << "\t" << _LATTICE_VELOCITIES[i*this->_D+1]; } std::cout << "   ==" << std::endl;
    std::cout << "== w  ="; for (int i=0; i<this->_Q; ++i) {std::cout << "\t" << _LATTICE_WEIGHTS[i]; } std::cout << "   ==" << std::endl;
    std::cout << "== op ="; for (int i=0; i<this->_Q; ++i) {std::cout << "\t" << _OPPOSITE_POPULATION[i]; } std::cout << "   ==" << std::endl;
    std::cout << "===============================================================================\n" << std::endl;
}

template<typename T>
void LBModel<T>::printBoundaryMapping() const {
    std::cout << "============ LB Model boundary populations ============" << std::endl;
    for (const auto& pair : _BOUNDARY_MAPPING) {
        std::cout << "== " << boundaryLocationToString(pair.first) << ":\t";
        
        for (unsigned int index : pair.second) {
            std::cout << index << " ";
        }
    std::cout << std::endl;       
    }
    std::cout << "=======================================================" << std::endl; 
}


/****************************************/
/***** Derived class 01: D2Q9 model *****/
/****************************************/
template<typename T>
D2Q9<T>::D2Q9() : LBModel<T>(2, 9) {
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

    // Set the desired velocity set here
    int velocities[9][2] = {{0, 0},{1, 0},{-1, 0},{0, 1},{0, -1},{1, 1},{-1, -1},{1, -1},{-1, 1}};

    this->_LATTICE_VELOCITIES = new int[18];
    this->_OPPOSITE_POPULATION = new unsigned int[9];

    for (unsigned int i = 0; i < 9; ++i) {
        // Fill _LATTICE_VELOCITIES
        this->_LATTICE_VELOCITIES[i*2]      = velocities[i][0];
        this->_LATTICE_VELOCITIES[i*2+1]    = velocities[i][1];
  
        // Fill _BOUNDARY_MAPPING with (vector) values
        if (velocities[i][0] > 0)       { this->_BOUNDARY_MAPPING[BoundaryLocation::WEST].push_back(i); }
        else if (velocities[i][0] < 0)  { this->_BOUNDARY_MAPPING[BoundaryLocation::EAST].push_back(i); }
        if (velocities[i][1] > 0)       { this->_BOUNDARY_MAPPING[BoundaryLocation::SOUTH].push_back(i); }
        else if (velocities[i][1] < 0)  { this->_BOUNDARY_MAPPING[BoundaryLocation::NORTH].push_back(i); }
        
        // Compute opposite populations
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
LBModel<T>* D2Q9<T>::getDerivedModel() const {
    return new D2Q9<T>(*this); // Return a pointer to a new D2Q9 object
}

/****************************************/
/***** Derived class 02: D2Q5 model *****/
/****************************************/
template<typename T>
D2Q5<T>::D2Q5() : LBModel<T>(2, 5) {
    this->_LATTICE_WEIGHTS = new T[5];
    this->_LATTICE_WEIGHTS[0] = 1.0/3.0;
    this->_LATTICE_WEIGHTS[1] = 1.0/6.0;
    this->_LATTICE_WEIGHTS[2] = 1.0/6.0;
    this->_LATTICE_WEIGHTS[3] = 1.0/6.0;
    this->_LATTICE_WEIGHTS[4] = 1.0/6.0;

    // Set the desired velocity set here
    int velocities[5][2] = {{0, 0},{1, 0},{-1, 0},{0, 1},{0, -1}};

    this->_LATTICE_VELOCITIES = new int[10];
    this->_OPPOSITE_POPULATION = new unsigned int[5];

    for (unsigned int i = 0; i < 5; ++i) {
        // Fill _LATTICE_VELOCITIES
        this->_LATTICE_VELOCITIES[i*2]      = velocities[i][0];
        this->_LATTICE_VELOCITIES[i*2+1]    = velocities[i][1];
  
        // Fill _BOUNDARY_MAPPING with (vector) values
        if (velocities[i][0] < 0)       { this->_BOUNDARY_MAPPING[BoundaryLocation::EAST].push_back(i); }
        else if (velocities[i][0] > 0)  { this->_BOUNDARY_MAPPING[BoundaryLocation::WEST].push_back(i); }
        if (velocities[i][1] < 0)       { this->_BOUNDARY_MAPPING[BoundaryLocation::SOUTH].push_back(i); }
        else if (velocities[i][1] > 0)  { this->_BOUNDARY_MAPPING[BoundaryLocation::NORTH].push_back(i); }
        
        // Compute opposite populations
        for (unsigned int j = 0; j < 5; ++j) {
            if (velocities[j][0] == -velocities[i][0] && velocities[j][1] == -velocities[i][1]) {
                this->_OPPOSITE_POPULATION[i] = j;
                break;
            }
        }
    }
}

template<typename T>
D2Q5<T>::~D2Q5() {}

template<typename T>
int D2Q5<T>::getCX(unsigned int i) const  {
    return this->_LATTICE_VELOCITIES[i*2];
}

template<typename T>
int D2Q5<T>::getCY(unsigned int i) const  {
    return this->_LATTICE_VELOCITIES[i*2+1];
}

template<typename T>
T D2Q5<T>::getWEIGHT(unsigned int i) const  {
    return this->_LATTICE_WEIGHTS[i];
}

template<typename T>
unsigned int D2Q5<T>::getOppositePopualation(unsigned int i) const  {
    return this->_OPPOSITE_POPULATION[i];
}

template<typename T>
LBModel<T>* D2Q5<T>::getDerivedModel() const {
    return new D2Q5<T>(*this); // Return a pointer to a new D2Q9 object
}

#endif // LB_MODEL_HH
