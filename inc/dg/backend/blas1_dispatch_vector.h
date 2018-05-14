#ifndef _DG_BLAS_STD_VECTOR_
#define _DG_BLAS_STD_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <vector>
#include <array>
#include "blas1_dispatch_shared.h"
#include "vector_categories.h"
#include "vector_traits.h"
#ifdef _OPENMP
#include <omp.h>
#endif //_OPENMP

///@cond
namespace dg
{
namespace blas1
{
namespace detail
{

template<class To, class From>
To doTransfer( const From& src, ArrayVectorTag, AnyVectorTag)
{
    To t;
    for (unsigned i=0; i<t.size(); i++)
        t[i] = src;
    return t;
}


template< class Vector, class Vector2>
inline get_value_type<Vector> doDot( const Vector& x1, const Vector2& x2, VectorVectorTag)
{
    static_assert( std::is_base_of<VectorVectorTag,
        get_data_layout<Vector2>>::value,
        "All container types must derive from the same vector category (VectorVectorTag in this case)!");
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
#endif //DG_DEBUG
    std::vector<std::vector<int64_t>> acc( x1.size());
    for( unsigned i=0; i<x1.size(); i++)
        acc[i] = doDot_superacc( x1[i], x2[i], get_data_layout<typename Vector::value_type>());
    for( unsigned i=1; i<x1.size(); i++)
    {
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
        imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
        for( int k=exblas::IMIN; k<exblas::IMAX; k++)
            acc[0][k] += acc[i][k];
    }
    return exblas::cpu::Round(&(acc[0][0]));
}
#ifdef _OPENMP
//omp tag implementation
template< class Subroutine, class container, class ...Containers>
inline void doSubroutine_dispatch( VectorVectorTag, OmpTag, Subroutine f, container&& x, Containers&&... xs)
{
    using inner_container = typename std::decay<container>::type::value_type;
    if( !omp_in_parallel())//to catch recursive calls
    {
        #pragma omp parallel
        {
            for( unsigned i=0; i<x.size(); i++) {
                doSubroutine( get_data_layout<inner_container>(), f, x[i], xs[i]...);
            }
        }
    }
    else //we are already in a parallel omp region
        for( unsigned i=0; i<x.size(); i++) {
            doSubroutine( get_data_layout<inner_container>(), f, x[i], xs[i]...);
        }
}
#endif //_OPENMP

//any tag implementation
template< class Subroutine, class container, class ...Containers>
inline void doSubroutine_dispatch( VectorVectorTag, AnyPolicyTag, Subroutine f, container&& x, Containers&&... xs)
{
    using inner_container = typename std::decay<container>::type::value_type;
    for( unsigned i=0; i<x.size(); i++) {
        doSubroutine( get_data_layout<inner_container>(), f, x[i], xs[i]...);
    }
}

//dispatch
template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( VectorVectorTag, Subroutine f, container&& x, Containers&&... xs)
{
    static_assert( all_true<std::is_base_of<VectorVectorTag,
        get_data_layout<Containers>>::value...>::value,
        "All container types must derive from the same vector category (VectorVectorTag in this case)!");
    static_assert( all_true<std::is_same<get_execution_policy<container>,
        get_execution_policy<Containers> >::value...>::value,
        "All container types must share the same execution policy!");
#ifdef DG_DEBUG
    //is this possible?
    //assert( !x.empty());
    //assert( x.size() == xs.size() );
#endif //DG_DEBUG
    doSubroutine_dispatch( VectorVectorTag(), get_execution_policy<container>(), f, std::forward<container>( x), std::forward<Containers>( xs)...);
}

} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_STD_VECTOR_
