#ifndef GRID_GEOMETRY_BASE_HH
#define GRID_GEOMETRY_BASE_HH

/******************************/
/***** Base grid geometry *****/
/******************************/
template<typename T>
GridGeometry<T>::GridGeometry(T delta) : _delta(delta) {}

template<typename T>
T GridGeometry<T>::getDelta() const {
    return _delta;
}

#endif // GRID_GEOMETRY_BASE_HH