#ifndef FIELD_TAGS_H
#define FIELD_TAGS_H

namespace descriptors {

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