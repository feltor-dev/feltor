#ifndef _DG_MATRIX_TRAITS_
#define _DG_MATRIX_TRAITS_

namespace dg{

template< class Matrix>
struct MatrixTraits {
    typedef typename Matrix::value_type value_type;
    typedef CuspMatrixTag matrix_category; //default is a CuspMatrix
};

}//namespace dg

#endif //_DG_MATRIX_TRAITS_
