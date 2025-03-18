#pragma once
#include <vector>
#include <array>
#include <map>
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

namespace detail
{

template< class Vector1, class Vector2>
inline std::vector<int64_t> doDot_superacc( int* status, const Vector1& x1, const Vector2& x2, StdMapTag)
{
    //find out which one is the RecursiveVector and determine size
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    std::vector<int64_t> acc( exblas::BIN_COUNT, (int64_t)0);
    int i=0;
    for( auto el : get_idx<vector_idx>(x1,x2))
    {
        try{
        std::vector<int64_t> temp = doDot_superacc(
            status,
            do_get_vector_element(x1,el.first,get_tensor_category<Vector1>()),
            do_get_vector_element(x2,el.first,get_tensor_category<Vector2>()));
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(temp[0]), imin, imax);
        for( int k=exblas::IMIN; k<=exblas::IMAX; k++)
            acc[k] += temp[k];
        if( (i+1)%128 == 0)
        {
            imin = exblas::IMIN, imax = exblas::IMAX;
            exblas::cpu::Normalize( &(acc[0]), imin, imax);
        }
        i++;
        }catch( std::out_of_range& err)
        {
            throw dg::Error( dg::Message(_ping_)<<"Wrong key '"<<el.first<<"' in blas1::dot "<<err.what());
        }
    }
    return acc;
}
/////////////////////////////////////////////////////////////////////////////////////

//dispatch
template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( StdMapTag, Subroutine f, container&& x, Containers&&... xs)
{
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, get_value_type<container>, container, Containers...>::value;
    for( auto el : get_idx<vector_idx>( std::forward<container>(x), std::forward<Containers>(xs)...))
    {
        try{
        dg::blas1::subroutine( f, do_get_vector_element(std::forward<container>(x),el.first,get_tensor_category<container>()), do_get_vector_element(std::forward<Containers>(xs),el.first,get_tensor_category<Containers>())...);
        }catch( std::out_of_range& err)
        {
            throw dg::Error( dg::Message(_ping_)<<"Wrong key '"<<el.first<<"' in subroutine");
        }
    }
}

} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond
