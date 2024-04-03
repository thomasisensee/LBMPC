#ifndef VTK_WRITER_HH
#define VTK_WRITER_HH

#include "vtkWriter.h"

std::string VTKWriter::constructFilename(const std::string& fieldName, unsigned int timeStep) {
    return _outputDir + "/" + _baseFilename + "_" + fieldName + "_" + std::to_string(timeStep) + ".vtk";
}

std::string VTKWriter::getVtkDataTypeString(const float&) {
    return "float";
}

std::string VTKWriter::getVtkDataTypeString(const double&) {
    return "float"; // VTK typically uses 'float' for both float and double
}

std::string VTKWriter::getVtkDataTypeString(const int&) {
    return "int";
}

void VTKWriter::update(unsigned int timeStep) {
    _currentTimeStep = timeStep;
    if (_currentTimeStep % _outputFrequency == 0) {
        writeSnapshot();
    }
}

template<typename T>
void VTKWriter::writeScalarField(const std::vector<T>& field, const std::string& fieldName, unsigned int timeStep) {
    std::string filename = constructFilename(fieldName, timeStep);
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    T typeDummy; // Create a dummy variable to deduce type for getVtkDataTypeString

    outFile << "# vtk DataFile Version 3.0\n";
    outFile << "Scalar Field\n";
    outFile << "BINARY\n";
    outFile << "DATASET STRUCTURED_POINTS\n";
    outFile << "DIMENSIONS " << _nX << _nY << " 1\n";
    outFile << "SPACING " << _delta << _delta << " 0.0\n";
    outFile << "ORIGIN 1 1 0\n";
    outFile << "POINT_DATA " << _nX*_nY << "\n";
    outFile << "SCALARS " << fieldName << " " << getVtkDataTypeString(typeDummy) << " 1\n";
    outFile << "LOOKUP_TABLE default\n";
    //for (const auto& value : field) {
    //    outFile << value << "\n";
    //}
    for(int j = 1; j <= _nY; ++j)
    {
        for(int i = 1; i <= _nX; ++i)
        {
			//ComputeDensityCPU(&h_CollideV[Q_LBM*pos(i,j)],&density);
			
            //floatValu = FloatSwap(((float) density));
            //fwrite((void*)&floatValu,sizeof(float),1,OutFile);
        }
    }
    outFile.close();
}




#endif // VTK_WRITER_HH