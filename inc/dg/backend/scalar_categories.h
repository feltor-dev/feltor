#ifndef _DG_SCALAR_CATEGORIES_
#define _DG_SCALAR_CATEGORIES_

#include "vector_categories.h"

namespace dg{

/**
 * @brief Scalar Tag base class, indicates the basic Scalar Tensor concept
 */
struct AnyScalarTag : public AnyVectorTag{};
struct ScalarTag : public AnyScalarTag{};

}//namespace dg

#endif //_DG_SCALAR_CATEGORIES_
