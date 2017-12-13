#ifndef _DG_BLAS_ARRAY_
#define _DG_BLAS_ARRAY_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <array>
#include "vector_categories.h"
#include "vector_traits.h"

///@cond
namespace dg
{
template<class T, std::size_t N>
struct std::array<T,N>Traits<std::array<T, N>, 
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using vector_category   = StdArrayTag;
    using execution_policy  = SerialTag;
};
template<class T, std::size_t N>
struct std::array<T,N>Traits<std::array<T, N>, 
    typename std::enable_if< !std::is_arithmetic<T>::value>::type>
{
    using value_type        = get_value_type<T>;
    using vector_category   = VectorVectorTag;
    using execution_policy  = get_execution_policy<T>;
};

namespace blas1
{
namespace detail
{

template< class T, std::size_t N>
exblas::Superaccumulator doDot_superacc( const std::array<T,N>& x, const std::array<T,N>& y, StdArrayTag)
{
    return exblas::Superaccumulator(  exblas::exdot_cpu( N, x.begin(),y.begin(),8,true)) ;
}

template<class T,std::size_t N>
T doDot( const std::array<T,N>& x, const std::array<T,N>& y, StdArrayTag)
{
    exblas::Superaccumulator acc = doDot_superacc( x,y,StdArrayTag());
    return acc.Round();
}

template< class T,std::size_t N, class UnaryOp>
inline void doTransform(  const std::array<T,N>& x, std::array<T,N>& y, UnaryOp op, StdArrayTag) {
    for( int i=0; i<N; i++)
        y[i] = op(x[i]);
}

template< class T, std::size_t N>
inline void doScal( std::array<T,N>& x, T alpha, StdArrayTag) {
    for( int i=0; i<N; i++)
        x[i] *= alpha;
}

template< class T, std::size_t N>
inline void doPlus(  std::array<T,N>& x, T alpha, StdArrayTag) {
    for( int i=0; i<N; i++)
        x[i] += alpha;
}

template< class T, std::size_t N>
inline void doAxpby( T alpha, 
              const std::array<T,N>& x, 
              T beta, 
              std::array<T,N>& y, 
              StdArrayTag) {
    for( int i=0; i<N; i++)
        y[i] = alpha*x[i]+beta*y[i];
}

template< class T, std::size_t N>
inline void doAxpbypgz( T alpha, 
              const std::array<T,N>& x, 
              T beta, 
              const std::array<T,N>& y, 
              T gamma, 
              std::array<T,N>& z, 
              StdArrayTag) {
    for( int i=0; i<N; i++)
        z[i] = alpha*x[i]+beta*y[i]+gamma*z[i];
}

template<class std::array<T,N>>
inline void doPointwiseDot(  
              T alpha, 
              const std::array<T,N>& x1,
              const std::array<T,N>& x2, 
              T beta, 
              std::array<T,N>& y, 
              StdArrayTag) {
    for( int i=0; i<N; i++)
        y[i] = alpha*x1[i]*x2[i]+beta*y[i];
}

template<class std::array<T,N>>
inline void doPointwiseDivide(  
              T alpha, 
              const std::array<T,N>& x1,
              const std::array<T,N>& x2, 
              T beta, 
              std::array<T,N>& y, 
              StdArrayTag) {
    for( int i=0; i<N; i++)
        y[i] = alpha*x1[i]/x2[i]+beta*y[i];
}

template<class std::array<T,N>>
inline void doPointwiseDot(  
              T alpha, 
              const std::array<T,N>& x1,
              const std::array<T,N>& y1, 
              T beta, 
              const std::array<T,N>& x2,
              const std::array<T,N>& y2, 
              T gamma, 
              std::array<T,N>& z, 
              StdArrayTag)
{
    for( int i=0; i<N; i++)
        z[i] = alpha*x1[i]*y1[i]+beta*x2[i]*y2[i]+gamma*z[i];
}
template<class std::array<T,N>>
inline void doPointwiseDot(  
              T alpha, 
              const std::array<T,N>& x1,
              const std::array<T,N>& x2,
              const std::array<T,N>& x3, 
              T beta, 
              std::array<T,N>& y, 
              StdArrayTag)
{
    for( int i=0; i<N; i++)
        y[i] = alpha*x1[i]*x2[i]*x3[i]*y2[i]+beta*y[i];
}

}//namespace detail

} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_ARRAY_
