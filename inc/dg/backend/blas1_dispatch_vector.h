#ifndef _DG_BLAS_STD_VECTOR_
#define _DG_BLAS_STD_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <vector>
#include <array>
#include "blas1_dispatch_shared.h"
#include "vector_categories.h"
#include "tensor_traits.h"
#ifdef _OPENMP
#include <omp.h>
#endif //_OPENMP

///@cond
namespace dg
{
namespace blas1
{

template<class to_ContainerType, class from_ContainerType>
inline to_ContainerType transfer( const from_ContainerType& src);

namespace detail
{

template<class To, class From>
To doTransfer( const From& src, ArrayVectorTag, AnyVectorTag)
{
    To t;
    using inner_vector = typename To::value_type;
    for (unsigned i=0; i<t.size(); i++)
        t[i] = dg::blas1::transfer<inner_vector>(src);
    return t;
}

template< class Vector1, class Vector2>
inline std::vector<int64_t> doDot_superacc( const Vector1& x1, const Vector2& x2, RecursiveVectorTag)
{
    //find out which one is the RecursiveVector and determine category
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    auto size = std::get<vector_idx>(std::forward_as_tuple(x1,x2)).size();
    std::vector<std::vector<int64_t>> acc( size);
    for( unsigned i=0; i<size; i++)
        acc[i] = doDot_superacc( get_element(x1,i), get_element(x2,i));
    for( unsigned i=1; i<size; i++)
    {
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
        imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
        for( int k=exblas::IMIN; k<exblas::IMAX; k++)
            acc[0][k] += acc[i][k];
    }
    return acc[0];
}
/////////////////////////////////////////////////////////////////////////////////////
#ifdef _OPENMP
//omp tag implementation
template< class size_type, class Subroutine, class container, class ...Containers>
inline void doSubroutine_dispatch( RecursiveVectorTag, OmpTag, size_type size, Subroutine f, container&& x, Containers&&... xs)
{
    //using inner_container = typename std::decay<container>::type::value_type;
    if( !omp_in_parallel())//to catch recursive calls
    {
        #pragma omp parallel
        {
            for( int i=0; i<(int)size; i++) {//omp sometimes has problems if loop variable is not int
                dg::blas1::subroutine( f, get_element(std::forward<container>(x),i), get_element(std::forward<Containers>(xs),i)...);
            }
        }
    }
    else //we are already in a parallel omp region
        for( int i=0; i<(int)size; i++) {
            dg::blas1::subroutine( f, get_element(std::forward<container>(x),i), get_element(std::forward<Containers>(xs),i)...);
        }
}
#endif //_OPENMP



//any tag implementation (recursively call subroutine)
template<class size_type, class Subroutine, class container, class ...Containers>
inline void doSubroutine_dispatch( RecursiveVectorTag, AnyPolicyTag, size_type size, Subroutine f, container&& x, Containers&&... xs)
{
    for( int i=0; i<(int)size; i++) {
        dg::blas1::subroutine( f, get_element(std::forward<container>(x),i), get_element(std::forward<Containers>(xs),i)...);
    }
}

//dispatch
template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( RecursiveVectorTag, Subroutine f, container&& x, Containers&&... xs)
{
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, get_value_type<container>, container, Containers...>::value;
    auto size = std::get<vector_idx>(std::forward_as_tuple(x,xs...)).size();
    using vector_type = find_if_t<dg::has_not_any_policy, get_value_type<container>, container, Containers...>;
    doSubroutine_dispatch( RecursiveVectorTag(), get_execution_policy<vector_type>(), size, f, std::forward<container>( x), std::forward<Containers>( xs)...);
}

} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_STD_VECTOR_
