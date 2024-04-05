#ifndef VTK_WRITER_HH
#define VTK_WRITER_HH

#include <filesystem>

#include "vtkWriter.h"

VTKWriter::VTKWriter(
    const std::string& outputDir,
    const std::string& baseFilename)
    : _outputDir(outputDir), _baseFilename(baseFilename) {
        setOutputDirectory();
    }

std::string VTKWriter::constructFilename(const std::string& fieldName, unsigned int iter) {
    return _outputDir + "/" + _baseFilename + "_" + fieldName + "_" + std::to_string(iter) + ".vtk";
}

std::string VTKWriter::getVtkDataTypeString(const float&) {
    return "float";
}

std::string VTKWriter::getVtkDataTypeString(const double&) {
    return "float";
}

std::string VTKWriter::getVtkDataTypeString(const int&) {
    return "int";
}

void VTKWriter::setOutputDirectory() { 
    std::filesystem::create_directories(_outputDir);
}

template<typename T>
inline T VTKWriter::SwapBytes(T value) {
    static_assert(sizeof(T) == 4, "SwapBytes function is designed for 4-byte types, i.e., int and float.");

    union {
        T val;
        unsigned char b[4];
    } dat1, dat2;

    dat1.val = value;
    dat2.b[0] = dat1.b[3];
    dat2.b[1] = dat1.b[2];
    dat2.b[2] = dat1.b[1];
    dat2.b[3] = dat1.b[0];

    return dat2.val;
}

void VTKWriter::setNonInherent(unsigned int nX, unsigned int nY, float delta) {
    _nX = nX;
    _nY = nY;
    _delta = delta;
}

template<typename T>
void VTKWriter::writeScalarField(const std::vector<T>& field, const std::string& fieldName, unsigned int iter) {
    std::string filename = constructFilename(fieldName, iter);
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    T swapped;      // Variable to store swapped bytes
    T typeDummy;    // Create a dummy variable to deduce type for getVtkDataTypeString

    outFile << "# vtk DataFile Version 3.0\n";
    outFile << "Scalar Field\n";
    outFile << "BINARY\n";
    outFile << "DATASET STRUCTURED_POINTS\n";
    outFile << "DIMENSIONS " << _nX << " " << _nY << " 1\n";
    outFile << "SPACING " << _delta << " " << _delta << " 0.0\n";
    outFile << "ORIGIN 0 0 0\n";
    outFile << "POINT_DATA " << _nX*_nY << "\n";
    outFile << "SCALARS " << fieldName << " " << getVtkDataTypeString(typeDummy) << " 1\n";
    outFile << "LOOKUP_TABLE default\n";

    for (const auto& value : field) {
        swapped = SwapBytes(value); // Swap bytes if necessary
        outFile.write(reinterpret_cast<const char*>(&swapped), sizeof(T));
    }
    outFile.close();
}

template<typename T>
void VTKWriter::writeVectorField(const std::vector<T>& field, const std::string& fieldName, unsigned int iter) {
    std::string filename = constructFilename(fieldName, iter);
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    T swapped;      // Variable to store swapped bytes
    T typeDummy;    // Create a dummy variable to deduce type for getVtkDataTypeString

    outFile << "# vtk DataFile Version 3.0\n";
    outFile << "Vector Field\n";
    outFile << "BINARY\n";
    outFile << "DATASET STRUCTURED_POINTS\n";
    outFile << "DIMENSIONS " << _nX << " " << _nY << " 1\n";
    outFile << "SPACING " << _delta << " " << _delta << " 0.0\n";
    outFile << "ORIGIN 0 0 0\n";
    outFile << "POINT_DATA " << _nX*_nY << "\n";
    outFile << "VECTORS " << fieldName << " " << getVtkDataTypeString(typeDummy) << " 3\n";

    for(size_t i = 0; i < _nY * _nX; ++i) {
        // x-value
        swapped = SwapBytes(field[i]);
        outFile.write(reinterpret_cast<const char*>(&swapped), sizeof(T));
        // y-value
        swapped = SwapBytes(field[i + 1]);
        outFile.write(reinterpret_cast<const char*>(&swapped), sizeof(T));
        // z-value
        swapped = 0.0;
        outFile.write(reinterpret_cast<const char*>(&swapped), sizeof(T));   
    }
    outFile.close();
}




#endif // VTK_WRITER_HH