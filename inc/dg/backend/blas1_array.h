#ifndef _DG_BLAS_ARRAY_
#define _DG_BLAS_ARRAY_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <array>
#include <type_traits>
#include "exblas/exdot_serial.h"
#include "config.h"
#include "vector_categories.h"
#include "vector_traits.h"

namespace dg
{
///@addtogroup vec_list
///@{
/**
 * @brief There is a special implementation of blas1 functions for a \c std::array different from the one indicated by SerialTag
 *
 * @tparam T arithmetic value type
 * @tparam N size of the array
 */
template<class T, std::size_t N>
struct VectorTraits<std::array<T, N>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using vector_category   = StdArrayTag;
    using execution_policy  = SerialTag;
};
template<class T, std::size_t N>
struct VectorTraits<std::array<T, N>,
    typename std::enable_if< !std::is_arithmetic<T>::value>::type>
{
    using value_type        = get_value_type<T>;
    using vector_category   = ArrayVectorTag;
    using execution_policy  = get_execution_policy<T>;
};
///@}

///@cond
namespace blas1
{
namespace detail
{

template< class T, std::size_t N>
std::vector<int64_t> doDot_superacc( const std::array<T,N>& x, const std::array<T,N>& y, StdArrayTag)
{
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    exblas::exdot_cpu( N, x.begin(),y.begin(), &h_superacc[0]) ;
    return h_superacc;
}

template<class T,std::size_t N>
T doDot( const std::array<T,N>& x, const std::array<T,N>& y, StdArrayTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,y,StdArrayTag());
    return exblas::cpu::Round(acc.data());
}

template< class Subroutine, std::size_t N, class T, class ...Ts>
inline void doSubroutine( StdArrayTag, Subroutine f, std::array<T,N>&& x, std::array<Ts,N>&&... xs) {
    for( size_t i=0; i<N; i++)
        f( x[i], xs[i]...);
}

} //namespace detail

} //namespace blas1
///@endcond
} //namespace dg

#endif //_DG_BLAS_ARRAY_
