#ifndef _DG_SCALAR_CATEGORIES_
#define _DG_SCALAR_CATEGORIES_

#include "vector_categories.h"

namespace dg{

//here we introduce the concept of data access

///@addtogroup dispatch
///@{
/**
 * @brief Scalar Tag base class, indicates the basic Scalar Tensor concept
 *
 * @note any scalar can serve as a vector
 */
struct AnyScalarTag : public AnyVectorTag{};
///@}
struct ScalarTag : public AnyScalarTag{};

}//namespace dg

#endif //_DG_SCALAR_CATEGORIES_
