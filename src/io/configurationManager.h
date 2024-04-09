#ifndef CONFIGURATION_MANAGER_H
#define CONFIGURATION_MANAGER_H

#include <memory>
#include <string>

#include "tinyxml2.h"

#include "core/simulation.h"
#include "core/gridGeometry.h"
#include "core/lb/boundaryConditions.h"

class ConfigurationManager {
private:
    std::string configFile;

    /// Helper function to read the boundary conditions from the XML file
    template<typename T>
    void readBoundaryCondition(BoundaryLocation location, const tinyxml2::XMLElement* boundary, std::unique_ptr<BoundaryConditionManager<T>>& boundaryConditionManager, T dtdx);

public:
    ConfigurationManager(const std::string& configFile);

    template<typename T>
    std::shared_ptr<LBFluidSimulation<T>> buildSimulation();
};

#include "configurationManager.hh"

#endif // CONFIGURATION_MANAGER_H