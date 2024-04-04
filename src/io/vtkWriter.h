#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class VTKWriter {
private:
    /// Inherent members
    std::string _outputDir;
    std::string _baseFilename;
    unsigned int _outputFrequency;
    unsigned int _currentIter;

    /// Non-inherent members (need to be set)
    unsigned int _nX, _nY;
    float _delta;

    /// Helper functions for templated writeScalarField function
    std::string constructFilename(const std::string& fieldName, unsigned int iter);
    std::string getVtkDataTypeString(const float&);
    std::string getVtkDataTypeString(const double&);
    std::string getVtkDataTypeString(const int&);

    /// Helper functions for writing binary data
    template<typename T>
    inline T SwapBytes(float f);

protected:
    void writeSnapshot() {
        // Example: Write multiple fields at the current timestep
        // writeField(velocityField, "velocity");
        // writeField(pressureField, "pressure");
        // Add more fields as necessary
    }

public:
    /// Constructor
    VTKWriter(const std::string& outputDir, const std::string& baseFilename, unsigned int outputFrequency);

    /// Destructor
    ~VTKWriter() = default;

    /// Set member variables necessary for output: grid dimensions nX and nY, and grid spacing delta
    void setNonInherent(unsigned int nX, unsigned int nY, float delta);

    /// Update current timeStep and check for output
    template<typename T>
    void update(unsigned int iter, std::vector<T>& hostData, T* deviceData);

    /// Fetch data from device
    template<typename T>
    void copyToHost(std::vector<T>& hostData, T* deviceData);

    /// Write a scalar field to a VTK file
    template<typename T>
    void writeScalarField(const std::vector<T>& field, const std::string& fieldName, unsigned int iter);

    /// Write a vector field to a VTK file
    template<typename T>
    void writeVectorField(const std::vector<T>& field, const std::string& fieldName, unsigned int iter);
};

#include "vtkWriter.hh"

#endif // VTK_WRITER_H