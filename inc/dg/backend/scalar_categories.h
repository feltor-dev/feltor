#ifndef _DG_SCALAR_CATEGORIES_
#define _DG_SCALAR_CATEGORIES_

#include "vector_categories.h"

namespace dg{

/**
 * @brief Scalar Tag base class
 */
struct AnyScalarTag : public AnyVectorTag{};

/// Types where \c std::is_arithmetic is true
struct ArithmeticTag : public AnyScalarTag{};
/// Types where \c std::is_floating_point is true
struct FloatingPointTag : public ArithmeticTag{};
/// Types where \c std::is_integral is true
struct IntegralTag : public ArithmeticTag{};

/// complex number type
struct ComplexTag : public AnyScalarTag{};


}//namespace dg

#endif //_DG_SCALAR_CATEGORIES_
