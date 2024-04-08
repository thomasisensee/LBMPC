#ifndef CONFIGURATION_MANAGER_H
#define CONFIGURATION_MANAGER_H

#include "tinyxml2.h"
#include <memory>
#include <string>


#include "core/GridGeometry2D.h"
#include "core/collisionModel.h"

class ConfigurationManager {
public:
    ConfigurationManager(const std::string& configFile);
    void parseConfiguration();
    std::shared_ptr<GridGeometry2D> getGridGeometry() const;
    std::shared_ptr<CollisionModel> getCollisionModel() const;

private:
    std::string configFile;
    std::shared_ptr<GridGeometry2D> gridGeometry;
    std::shared_ptr<CollisionModel> collisionModel;

    void createGridGeometry(tinyxml2::XMLElement* root);
    void createCollisionModel(tinyxml2::XMLElement* root);
};

#endif // CONFIGURATION_MANAGER_H