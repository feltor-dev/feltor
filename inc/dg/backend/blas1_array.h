#ifndef _DG_BLAS_ARRAY_
#define _DG_BLAS_ARRAY_

#include <array>
#include <type_traits>
#include "exblas/exdot_serial.h"
#include "config.h"
#include "vector_categories.h"
#include "tensor_traits.h"

//UNUSED AT THE MOMENT
///@cond
namespace dg
{
namespace blas1
{
namespace detail
{

template< class T, std::size_t N>
std::vector<int64_t> doDot_superacc( const std::array<T,N>& x, const std::array<T,N>& y, StdArrayTag)
{
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    int status = 0;
    exblas::exdot_cpu( N, x.begin(),y.begin(), &h_superacc[0], &status) ;
    if(status != 0)
        throw dg::Error(dg::Message(_ping_)<<"CPU Dot failed since one of the inputs contains NaN or Inf");
    return h_superacc;
}

template<class T,std::size_t N>
T doDot( const std::array<T,N>& x, const std::array<T,N>& y, StdArrayTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,y,StdArrayTag());
    return exblas::cpu::Round(acc.data());
}

//This implementation is wrong since now x and y are always rvalue references instead of universal references
template< class Subroutine, std::size_t N, class T, class ...Ts>
inline void doSubroutine( StdArrayTag, Subroutine f, std::array<T,N>&& x, std::array<Ts,N>&&... xs) {
    for( size_t i=0; i<N; i++)
        f( x[i], xs[i]...);
}

} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_ARRAY_
