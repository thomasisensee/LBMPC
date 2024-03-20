#ifndef Cell_2D_H
#define Cell_2D_H

template<typename T>
class Cell2D {

private:
    /// Populations
    std::vector<T> population; 

public:
    /// Construction of a 2D cell
    Cell2D();
    /// Initializes the 2D cell
    void init();
    /// Computes the density (0th moment)
    T getDensity() const;
    /// Computes the density (0th moment)
    std::vector<T> getVelocity() const;
};

#endif
