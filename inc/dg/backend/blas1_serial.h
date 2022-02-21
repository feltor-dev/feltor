#ifndef _DG_BLAS_SERIAL_
#define _DG_BLAS_SERIAL_
#include "config.h"
#include "exceptions.h"
#include "execution_policy.h"
#include "exblas/exdot_serial.h"

namespace dg
{
namespace blas1
{
namespace detail
{
template<class PointerOrValue1, class PointerOrValue2>
inline std::vector<int64_t> doDot_dispatch( SerialTag, unsigned size, PointerOrValue1 x_ptr, PointerOrValue2 y_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    int status = 0;
    exblas::exdot_cpu( size, x_ptr,y_ptr, &h_superacc[0],&status) ;
    if(status != 0)
        throw dg::Error(dg::Message(_ping_)<<"CPU Dot failed since one of the inputs contains NaN or Inf");
    return h_superacc;
}
template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3>
inline std::vector<int64_t> doDot_dispatch( SerialTag, unsigned size, PointerOrValue1 x_ptr, PointerOrValue2 y_ptr, PointerOrValue3 z_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    int status = 0;
    exblas::exdot_cpu( size, x_ptr,y_ptr,z_ptr, &h_superacc[0], &status) ;
    if(status != 0)
        throw dg::Error(dg::Message(_ping_)<<"CPU Dot failed since one of the inputs contains NaN or Inf");
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


}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_SERIAL_
