#ifndef CONFIGURATION_MANAGER_HH
#define CONFIGURATION_MANAGER_HH

#include "tinyxml2.h"

#include "configurationManager.h"
#include "core/constants.h"
#include "core/lb/boundaryConditions.h"


ConfigurationManager::ConfigurationManager(const std::string& configFile) : configFile(configFile) {}

template<typename T>
void ConfigurationManager::readBoundaryCondition(BoundaryLocation location, const tinyxml2::XMLElement* boundary, std::unique_ptr<BoundaryConditionManager<T>>& boundaryConditionManager, T dtdx) {
    const char* type = boundary->FirstChildElement("type")->GetText();
    if      (std::string(type) == "bounceBack"  ) { boundaryConditionManager->addBoundaryCondition(std::make_unique<BounceBack<T>>(location)); }
    else if (std::string(type) == "periodic"    ) { boundaryConditionManager->addBoundaryCondition(std::make_unique<Periodic<T>>(location)); }
    else if (std::string(type) == "movingWall"  ) {
        const tinyxml2::XMLElement* vel = boundary->FirstChildElement("velocity");
        if (!vel) {
            std::cerr << "Failed to read boundary condition information." << std::endl;
            return;
        } else {
            T u = vel->FirstChildElement("x")->FloatText();
            T v = vel->FirstChildElement("y")->FloatText();
            u *= dtdx;
            v *= dtdx;
            std::vector<T> wallVelocity = {u,v};
            boundaryConditionManager->addBoundaryCondition(std::make_unique<MovingWall<T>>(location, wallVelocity));
        }
    } else {
        std::cerr << "Failed to read boundary condition information." << std::endl;
    }
}

template<typename T>
std::shared_ptr<LBFluidSimulation<T>> ConfigurationManager::buildSimulation() {
//int ConfigurationManager::buildSimulation() {
    // Load the XML configuration file
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError result = doc.LoadFile(configFile.c_str());
    if (result != tinyxml2::XML_SUCCESS) {
        std::cerr << "Failed to load " << configFile.c_str() << std::endl;
        std::cout << "XML error: " << doc.ErrorIDToName(result) << std::endl;
    }
    tinyxml2::XMLElement* root = doc.RootElement();

    // Load physical properties
    T reynoldsNumber = 0.0;

    const tinyxml2::XMLElement* physicalProperties = root->FirstChildElement("physicalProperties");
    if (!physicalProperties) {
        std::cerr << "Failed to load physicalProperties." << std::endl;
    } else {
        const tinyxml2::XMLElement* reynolds = physicalProperties->FirstChildElement("Reynolds");
        if (!reynolds) {
            std::cerr << "Failed to load Reynolds number." << std::endl;
        } else {
            reynoldsNumber = reynolds->FloatText();
            std::cout << "Reynolds number: " << reynoldsNumber << std::endl;
        }
    }

    // Load grid properties
    unsigned int nX = 0;
    unsigned int nY = 0;
    T dx = 0.0;

    const tinyxml2::XMLElement* grid = root->FirstChildElement("grid");
    if (!grid) {
        std::cerr << "Failed to read grid information." << std::endl;
    } else {
        const tinyxml2::XMLElement* Nx = grid->FirstChildElement("nX");
        const tinyxml2::XMLElement* Ny = grid->FirstChildElement("nY");
        const tinyxml2::XMLElement* Dx = grid->FirstChildElement("dx");

        if (!Nx || !Ny) {
            std::cerr << "Failed to read grid information." << std::endl;
        } else {
            nX = Nx->UnsignedText();
            nY = Ny->UnsignedText();
        }
        if (!Dx) {
            const tinyxml2::XMLElement* LengthX = grid->FirstChildElement("LengthX");
            if (!LengthX) {
                std::cerr << "Failed to read grid information." << std::endl;
            }
            T lengthX = LengthX->FloatText();
            dx = lengthX / nX;
        } else {
            dx = Dx->FloatText();
        }
    }

    // Load LB model
    std::unique_ptr<LBModel<T>> lbModel;

    const tinyxml2::XMLElement* LBModel = root->FirstChildElement("lbModel");
    if (!LBModel) {
        std::cerr << "Failed to read LB model information." << std::endl;
    } else {
        const char* modelType = LBModel->FirstChildElement("type")->GetText();
        if (std::string(modelType) == "D2Q9") {
            lbModel = std::make_unique<D2Q9<T>>();
        } else if (std::string(modelType) == "D2Q5") {
            lbModel = std::make_unique<D2Q5<T>>();
        } else {
            std::cerr << "Failed to read LB model information." << std::endl;
        }
    }

    // Load collision model
    std::unique_ptr<CollisionModel<T>> collisionModel;
    T omegaShear = 0.0;
    T omegaBulk = 0.0;
    const tinyxml2::XMLElement* CollisionModel = root->FirstChildElement("collisionModel");
    if (!LBModel) {
        std::cerr << "Failed to read LB model information." << std::endl;
    } 
    const tinyxml2::XMLElement* oS = CollisionModel->FirstChildElement("relaxationShear");
    if (!oS) {
        std::cerr << "Failed to read LB model information: omega shear." << std::endl;
    } else {
        omegaShear = oS->FloatText();
        const char* modelType = CollisionModel->FirstChildElement("type")->GetText();
        if (std::string(modelType) == "BGK") {
            collisionModel = std::make_unique<CollisionBGK<T>>(omegaShear);
        } else if (std::string(modelType) == "CHM") {
            const tinyxml2::XMLElement* oB = CollisionModel->FirstChildElement("relaxationBulk");
            if (!oB) {
                std::cerr << "Failed to read LB model information: omega bulk." << std::endl;
            } else {
                omegaBulk = oB->FloatText();
                collisionModel = std::make_unique<CollisionCHM<T>>(omegaShear, omegaBulk);
            }
        }
    }

    // Compute the time step
	T dt = (omegaShear - 0.5) * dx * dx * reynoldsNumber * C_S_POW2;


    // Load boundary conditions
    auto boundaryConditionManager = std::make_unique<BoundaryConditionManager<T>>();
    boundaryConditionManager->setDxdt(dx/dt);

    const tinyxml2::XMLElement* boundaryConditions = root->FirstChildElement("boundaryConditions");
    if(!oS) {
        std::cerr << "Failed to read boundary condition information." << std::endl;
    } else {
        const tinyxml2::XMLElement* west    = boundaryConditions->FirstChildElement("WEST");
        const tinyxml2::XMLElement* east    = boundaryConditions->FirstChildElement("EAST");
        const tinyxml2::XMLElement* south   = boundaryConditions->FirstChildElement("SOUTH");
        const tinyxml2::XMLElement* north   = boundaryConditions->FirstChildElement("NORTH");
        if (!west || !east || !south || !north) {
            std::cerr << "Failed to read boundary condition information." << std::endl;
        } else {
            readBoundaryCondition<T>(BoundaryLocation::WEST,    west,   boundaryConditionManager, dt/dx);
            readBoundaryCondition<T>(BoundaryLocation::EAST,    east,   boundaryConditionManager, dt/dx);
            readBoundaryCondition<T>(BoundaryLocation::SOUTH,   south,  boundaryConditionManager, dt/dx);
            readBoundaryCondition<T>(BoundaryLocation::NORTH,   north,  boundaryConditionManager, dt/dx);
        }
    }

    // Load simulation info
    T simTime = 0.0;
  
    const tinyxml2::XMLElement* sim = root->FirstChildElement("simulation");
    if (!sim) {
        std::cerr << "Failed to load physicalProperties." << std::endl;
    } else {
        const tinyxml2::XMLElement* simT = sim->FirstChildElement("simTime");
        if (!simT) {
            std::cerr << "Failed to load simulation time." << std::endl;
        } else {
            simTime = simT->FloatText();
        }
    }


    // Load output info
    unsigned int nOut = 0;
    std::string outputDirectory;
  
    const tinyxml2::XMLElement* out = root->FirstChildElement("output");
    if (!sim) {
        std::cerr << "Failed to load information on simulation output." << std::endl;
    } else {
        const tinyxml2::XMLElement* Nout = out->FirstChildElement("numberOutputFiles");
        if (!Nout) {
            std::cerr << "Failed to load number of file outputs." << std::endl;
        } else {
            nOut = Nout->UnsignedText();
        }

        const tinyxml2::XMLElement* outDir = out->FirstChildElement("outputDirectory");
        if (!outDir) {
            std::cerr << "Failed to load output directory." << std::endl;
        } else {
            outputDirectory = outDir->GetText();
        }
    }

    // Create the objects that are needed for the simulation
    auto gridGeometry = std::make_unique<GridGeometry2D<T>>(dx, nX, nY);
    auto lbGrid = std::make_unique<LBGrid<T>>(
        std::move(lbModel),
        std::move(collisionModel), 
        std::move(gridGeometry), 
        std::move(boundaryConditionManager)
    );
    auto vtkWriter = std::make_unique<VTKWriter>(outputDirectory, "cavity");

    // Create and return the simulation object
    return std::make_shared<LBFluidSimulation<T>>(std::move(lbGrid), std::move(vtkWriter), dt, simTime, nOut);
}

#endif // CONFIGURATION_MANAGER_HH