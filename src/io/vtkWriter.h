#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class VTKWriter {
private:
    std::string _outputDir;
    std::string _baseFilename;
    unsigned int _outputFrequency;
    unsigned int _currentTimeStep;

    std::string constructFilename(const std::string& fieldName, unsigned int timeStep);
    std::string getVtkDataTypeString(const float&);
    std::string getVtkDataTypeString(const double&);
    std::string getVtkDataTypeString(const int&);

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

    void update(unsigned int timeStep);

    template<typename T>
    void writeScalarField(const std::vector<T>& field, const std::string& fieldName, unsigned int timeStep);
};

#include "vtkWriter.hh"

#endif // VTK_WRITER_H