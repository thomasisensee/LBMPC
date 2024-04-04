#ifndef VTK_WRITER_HH
#define VTK_WRITER_HH

#include "vtkWriter.h"

VTKWriter::VTKWriter(
    const std::string& outputDir,
    const std::string& baseFilename,
    unsigned int outputFrequency)
    : _outputDir(outputDir), _baseFilename(baseFilename),_outputFrequency(outputFrequency) ,_currentIter(0) {}

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

template<typename T>
inline T SwapBytes(T value) {
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
void VTKWriter::update(unsigned int iter, std::vector<T>& hostData, T* deviceData) {
    _currentIter = iter;
    if (_currentIter % _outputFrequency == 0) {
        copyToHost(hostData, deviceData);
        writeSnapshot(hostData);
    }
}

template<typename T>
void VTKWriter::copyToHost(std::vector<T>& hostData, T* deviceData) { // different for velocity or scalar!!!
    cudaMemcpy(hostData.data(), deviceData, _nX * _nY * sizeof(T), cudaMemcpyDeviceToHost);
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
    outFile << "DIMENSIONS " << _nX << _nY << " 1\n";
    outFile << "SPACING " << _delta << _delta << " 0.0\n";
    outFile << "ORIGIN 0 0 0\n";
    outFile << "POINT_DATA " << _nX*_nY << "\n";
    outFile << "SCALARS " << fieldName << " " << getVtkDataTypeString(typeDummy) << " 1\n";
    outFile << "LOOKUP_TABLE default\n";

    for(int j = 1; j <= _nY; ++j)
    {
        for(int i = 1; i <= _nX; ++i)
        {
            swapped = SwapBytes(field[j * _nX + i]); // Swap bytes if necessary
            outFile.write(reinterpret_cast<const char*>(&swapped), sizeof(T));
        }
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
    outFile << "DIMENSIONS " << _nX << _nY << " 1\n";
    outFile << "SPACING " << _delta << _delta << " 0.0\n";
    outFile << "ORIGIN 0 0 0\n";
    outFile << "POINT_DATA " << _nX*_nY << "\n";
    outFile << "VECTORS " << fieldName << " " << getVtkDataTypeString(typeDummy) << " 3\n";

    for(int j = 1; j <= _nY; ++j)
    {
        for(int i = 1; i <= _nX; ++i)
        {
            // x-value
            swapped = SwapBytes(field[j * _nX + i]);
            outFile.write(reinterpret_cast<const char*>(&swapped), sizeof(T));
            // y-value
            swapped = SwapBytes(field[j * _nX + i + 1]);
            outFile.write(reinterpret_cast<const char*>(&swapped), sizeof(T));
        }
    }
    outFile.close();
}




#endif // VTK_WRITER_HH