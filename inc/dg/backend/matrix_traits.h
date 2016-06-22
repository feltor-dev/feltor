#ifndef _DG_MATRIX_TRAITS_
#define _DG_MATRIX_TRAITS_

#include "matrix_categories.h"
namespace dg{

    //Note to developpers: if you have problems with the compiler choosing CuspMatrixTag even if you don't want it to and you specialized the MatrixTraits for 
    //your matrix try to specialize for const Matrix as well
///@cond
template< class Matrix>
struct MatrixTraits {
    typedef typename Matrix::value_type value_type;//!< default value type
    typedef CuspMatrixTag matrix_category; //!< default is a CuspMatrix
};
///@endcond


}//namespace dg

#endif //_DG_MATRIX_TRAITS_
