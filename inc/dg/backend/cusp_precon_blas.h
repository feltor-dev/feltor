#pragma once 

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <cusp/array1d.h>

namespace dg{

    
template<>
struct MatrixTraits< cusp::array1d<double,cusp::device_memory> >
{
    typedef double value_type;
    typedef CuspPreconTag matrix_category;
};
template<>
struct MatrixTraits< cusp::array1d<float,cusp::device_memory> >
{
    typedef float value_type;
    typedef CuspPreconTag matrix_category;
};


namespace blas2{
    ///@cond
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
    ///@endcond
} //namespace blas2
} //namespace dg
