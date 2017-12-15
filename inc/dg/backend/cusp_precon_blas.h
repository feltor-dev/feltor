#pragma once 

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <cusp/array1d.h>
#include "matrix_traits.h"
#include "matrix_categories.h"

namespace dg{

///@cond
template<class T,class M>
struct MatrixTraits<cusp::array1d<T,M> >
{
    using value_type = typename cusp::array1d<T,M>::value_type;
    using matrix_category = CuspPreconTag; 
};
template<class T,class M>
struct MatrixTraits<const cusp::array1d<T,M> >
{
    using value_type = typename cusp::array1d<T,M>::value_type;
    using matrix_category = CuspPreconTag; 
};

///@endcond
} //namespace dg
