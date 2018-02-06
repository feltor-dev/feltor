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
    typedef typename cusp::array1d<T,M>::value_type value_type;
    typedef CuspPreconTag matrix_category; //default is a ThrustVector
};
template<class T,class M>
struct MatrixTraits<const cusp::array1d<T,M> >
{
    typedef typename cusp::array1d<T,M>::value_type value_type;
    typedef CuspPreconTag matrix_category; //default is a ThrustVector
};



namespace blas2{
namespace detail{


//template< class Matrix, class Vector>
//inline typename MatrixTraits<Matrix>::value_type doDot( const Vector& x, const Matrix& m, const Vector& y, CuspPreconTag, CuspVectorTag)
//{
//#ifdef DG_DEBUG
//    assert( x.size() == y.size() && x.size() == m.size() );
//#endif //DG_DEBUG
//    Vector temp(x);
//    cusp::blas::xmy( x,m, temp);
//    return cusp::blas::dot( temp, y);
//}
//
//template< class Matrix, class Vector>
//inline typename MatrixTraits<Matrix>::value_type doDot( const Matrix& m, const Vector& x, CuspPreconTag, CuspVectorTag)
//{
//#ifdef DG_DEBUG
//    assert( m.size() == x.size());
//#endif //DG_DEBUG
//    Vector temp(x);
//    cusp::blas::xmy( x,m, temp);
//    return cusp::blas::dot( temp, x);
//}


}//namespace detail
} //namespace blas2

///@endcond
} //namespace dg
