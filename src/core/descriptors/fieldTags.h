#ifndef FIELD_TAGS_H
#define FIELD_TAGS_H

namespace descriptors {

    /************************************************************************************************************************************************/
    /***** These empty structs are used in the descriptor to transport information about the nature of the distribution (momentum, energy, ...) *****/
    /************************************************************************************************************************************************/
    struct MomentumConservation {
        MomentumConservation() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type
    };

    struct EnergyConservation {
        EnergyConservation() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type
    };

    /*************************************************************************************************************************************************/
    /***** These empty structs are used in the descriptor to transport information about the additional fields passed alongside the distribution *****/
    /*************************************************************************************************************************************************/
    struct NoField {
    };

    struct VelocityField {
        VelocityField() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type
    };

    struct TemperatureField {
        TemperatureField() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type
    };

    struct ConcentrationField {
        ConcentrationField() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type
    };

    struct LiquidField {
        LiquidField() = delete; // Deleted default constructor prevents instantiation, enforces pure usage as type
    };
}

#endif // FIELD_TAGS_H