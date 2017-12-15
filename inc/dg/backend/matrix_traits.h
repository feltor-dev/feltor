#ifndef _DG_MATRIX_TRAITS_
#define _DG_MATRIX_TRAITS_

#include "matrix_categories.h"
namespace dg{

/*! @brief The matrix traits 

Specialize this struct with the SelfMadeMatrixTag as matrix_category if you want to enable your class for the use in blas2 functions
@note if you have problems with the compiler choosing CuspMatrixTag even if you don't want it to and you specialized the MatrixTraits for 
your matrix try to specialize for const Matrix as well
*/
template< class Matrix>
struct MatrixTraits {
    using value_type        = typename Matrix::value_type;//!< default value type
    using matrix_category   = CuspMatrixTag; //!< default is a CuspMatrix
};

template<class Matrix>
using get_matrix_category = typename MatrixTraits<Matrix>::matrix_category;

}//namespace dg

#endif //_DG_MATRIX_TRAITS_
