#ifndef _DG_MATRIX_TRAITS_
#define _DG_MATRIX_TRAITS_

//#include <vector>
#include "matrix_categories.h"
namespace dg{

template< class Matrix>
struct MatrixTraits {
    typedef typename Matrix::value_type value_type;//!< default value type
    typedef CuspMatrixTag matrix_category; //!< default is a CuspMatrix
};
//template< class Matrix>
//struct MatrixTraits<Matrix*> {
//    typedef typename Matrix::value_type value_type;//!< default value type
//    typedef typename MatrixTraits<Matrix>::matrix_category; 
//};

//template< class Matrix>
//struct MatrixTraits< std::vector<Matrix> >{
//    typedef typename Matrix::value_type value_type;
//    typedef StdMatrixTag matrix_category; 
//};
//template< class Matrix>
//struct MatrixTraits< std::vector<Matrix*> >{
//    typedef typename Matrix::value_type value_type;
//    typedef StdMatrixPointerTag matrix_category; 
//};


}//namespace dg

#endif //_DG_MATRIX_TRAITS_
