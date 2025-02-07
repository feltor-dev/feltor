#ifndef _DG_BLAS_SERIAL_
#define _DG_BLAS_SERIAL_
#include "config.h"
#include "exceptions.h"
#include "execution_policy.h"
#include "exblas/exdot_serial.h"
#include "exblas/fpedot_serial.h"

namespace dg
{
namespace blas1
{
namespace detail
{
template<class T, size_t N, class Functor, class ...PointerOrValues>
inline void doDot_fpe_dispatch( SerialTag, int * status, unsigned size, std::array<T,N>& fpe,
    Functor f, PointerOrValues ...xs_ptr)
{
    exblas::fpedot_cpu<T,N,Functor,PointerOrValues...>( status, size, fpe, f, xs_ptr...);
}
template<class PointerOrValue1, class PointerOrValue2>
inline std::vector<int64_t> doDot_dispatch( SerialTag, int* status, unsigned size,
    PointerOrValue1 x_ptr, PointerOrValue2 y_ptr)
{
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    exblas::exdot_cpu( size, x_ptr,y_ptr, &h_superacc[0], status) ;
    return h_superacc;
}
template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3>
inline std::vector<int64_t> doDot_dispatch( SerialTag, int* status, unsigned size,
    PointerOrValue1 x_ptr, PointerOrValue2 y_ptr, PointerOrValue3 z_ptr)
{
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    exblas::exdot_cpu( size, x_ptr,y_ptr,z_ptr, &h_superacc[0], status) ;
    return h_superacc;
}

template<class T>
inline T get_element( T x, int i){
	return x;
}
template<class T>
inline T& get_element( T* x, int i){
	return *(x+i);
}
template< class Subroutine, class PointerOrValue, class ...PointerOrValues>
inline void doSubroutine_dispatch( SerialTag, int size, Subroutine f, PointerOrValue x, PointerOrValues... xs)
{
    for( int i=0; i<size; i++)
    {
        f(get_element(x,i), get_element(xs,i)...);
        //f(x[i], xs[i]...);
        //f(thrust::raw_reference_cast(*(x+i)), thrust::raw_reference_cast(*(xs+i))...);
    }
}

template<class T, class Pointer, class BinaryOp, class UnaryOp>
inline T doReduce_dispatch( SerialTag, int size, Pointer x, T init, BinaryOp
        op, UnaryOp unary_op)
{
    for(int i=0; i<size; i++)
        init = op( init, unary_op(x[i]));
    return init;
}
template<class B, class F, class Pointer, std::size_t ...I, class ...PointerOrValues>
void call_host_F( B binary, F f, Pointer y, unsigned u, size_t* a, std::index_sequence<I...>, PointerOrValues ... xs)
{
    binary( f( get_element( xs, a[I])...), y[u]);
}
template<class B, class F, size_t N, class Pointer, class ...PointerOrValues>
void doKronecker_dispatch( dg::SerialTag, Pointer y, size_t size, B binary, F f, const std::array<size_t, N>& sizes, PointerOrValues ...xs)
{
    std::array<size_t, N> current = {0};
    for( unsigned u=0; u<size; u++)
    {
        //current[0] = u%sizes[0];
        //size_t remain = u/sizes[0];
        //for( unsigned k=1; k<N; k++)
        //{
        //    current[k] = remain%sizes[k];
        //    remain = remain/sizes[k];
        //}
        call_host_F( binary, f, y, u, &current[0], std::make_index_sequence<N>(), xs ...);
        // Counting is MUCH faster than modulo operations on CPU
        for( unsigned k=0; k<N; k++)
        {
            current[k] ++;
            if( current[k] == sizes[k])
                current[k] = 0;
            else
                break;
        }
    }
}


}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_SERIAL_
