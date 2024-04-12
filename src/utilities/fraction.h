#ifndef UTILITIES_FRACTION_H
#define UTILITIES_FRACTION_H

#include <stdexcept>

namespace utilities {

    /// Floating-point independent fraction type
    class Fraction {
    private:
        const int _numerator;
        const int _denominator;

    public:
        constexpr Fraction(int num, int denum): _numerator(num), _denominator(denum) {
            if (_denominator == 0) {
                throw std::invalid_argument("denominator must not be zero");
            }
        }

        constexpr Fraction(int parts[2]):
            Fraction(parts[0], parts[1]) { }

        constexpr Fraction(int num):
            Fraction(num, 1) { }

        constexpr Fraction():
            Fraction(0) { }

        template <typename T>
        __host__ __device__ constexpr T as() const {
            return T(_numerator) / T(_denominator);
        }

        template <typename T>
        __host__ __device__ constexpr T inverseAs() const {
            //return _numerator != 0 ? T(_denominator) / T(_numerator) : throw std::invalid_argument("inverse of zero is undefined");
            return T(_denominator) / T(_numerator);
        }
    };

}

#endif // UTILITIES_FRACTION_H