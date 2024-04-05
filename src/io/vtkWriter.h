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

    /// Non-inherent members (need to be set)
    unsigned int _nX, _nY;
    float _delta;

    /// Helper functions for templated writeScalarField function
    std::string constructFilename(const std::string& fieldName, unsigned int iter);
    std::string getVtkDataTypeString(const float&);
    std::string getVtkDataTypeString(const double&);
    std::string getVtkDataTypeString(const int&);

    /// Create the output directory if it doesn't exist
    void setOutputDirectory();


    /// Helper functions for writing binary data
    template<typename T>
    inline T SwapBytes(T value);

public:
    /// Constructor
    VTKWriter(const std::string& outputDir, const std::string& baseFilename);

    /// Destructor
    ~VTKWriter() = default;

    /// Set member variables necessary for output: grid dimensions nX and nY, and grid spacing delta
    void setNonInherent(unsigned int nX, unsigned int nY, float delta);

    /// Write a scalar field to a VTK file
    template<typename T>
    void writeScalarField(const std::vector<T>& field, const std::string& fieldName, unsigned int iter);

    /// Write a vector field to a VTK file
    template<typename T>
    void writeVectorField(const std::vector<T>& field, const std::string& fieldName, unsigned int iter);
};

#include "vtkWriter.hh"

#endif // VTK_WRITER_H