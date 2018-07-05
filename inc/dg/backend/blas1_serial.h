#ifndef _DG_BLAS_SERIAL_
#define _DG_BLAS_SERIAL_
#include "config.h"
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
    exblas::exdot_cpu( size, x_ptr,y_ptr, &h_superacc[0]) ;
    return h_superacc;
}
template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3>
inline std::vector<int64_t> doDot_dispatch( SerialTag, unsigned size, PointerOrValue1 x_ptr, PointerOrValue2 y_ptr, PointerOrValue3 z_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    exblas::exdot_cpu( size, x_ptr,y_ptr,z_ptr, &h_superacc[0]) ;
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


}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_SERIAL_
