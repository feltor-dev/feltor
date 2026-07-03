#pragma once

#include <cmath>
#include <complex>
#include <thrust/complex.h>

namespace dg
{
///@cond
namespace detail
{
// Making T0, T1, T2 different fixes problem that any of them may be const reference types
// overload A in std::fma
// for cuda all types must be equal?
template<class T0, class T1, class T2, class = std::enable_if_t<std::is_floating_point_v<T2> >>
DG_DEVICE
auto dg_fma( T0 x, T1 y, T2 z)
{
    return fma( (T2)x, (T2)y, z);
}
// Integer multiplication
template<class T0, class T1, class T2>
DG_DEVICE
std::enable_if_t< std::is_integral_v<T2>,decltype(T0()*T1()+T2())> dg_fma( T0 x, T1 y, T2 z)
{
    return x*y+z;
}
//////////////////////////////STD/////////////////////////
template<class T0, class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
std::complex<T> dg_fma( T0 x, std::complex<T> y, std::complex<T> z)
{
    return {
        std::fma( (T)x, y.real(), z.real()),
        std::fma( (T)x, y.imag(), z.imag())
    };
}
template<class T0, class T1, class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
std::complex<T> dg_fma( T0 x, T1 y, std::complex<T> z)
{
    return {
        std::fma( (T)x, (T)y, z.real()),
        z.imag()
    };
}
template<class T0, class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
std::complex<T> dg_fma( std::complex<T> x, T0 y, std::complex<T> z)
{
    return {
        fma( x.real(), (T)y, z.real()),
        fma( x.imag(), (T)y, z.imag())
    };
}
template<class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
std::complex<T> dg_fma( std::complex<T> x, std::complex<T> y, std::complex<T> z)
{
    std::complex<T> out = {
        std::fma( x.real(), y.real(), z.real()),
        std::fma( x.real(), y.imag(), z.imag())
    };
    return {
        std::fma( -x.imag(), y.imag(), out.real()),
        std::fma( x.imag(), y.real(), out.imag())
    };
}
//////////////////////////////THRUST/////////////////////////
template<class T0, class T1, class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
DG_DEVICE
thrust::complex<T> dg_fma( T0 x, T1 y, thrust::complex<T> z)
{
    return {
        std::fma( (T)x, (T)y, z.real()),
        z.imag()
    };
}
template<class T0, class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
DG_DEVICE
thrust::complex<T> dg_fma( T0 x, thrust::complex<T> y, thrust::complex<T> z)
{
    return {
        fma( (T)x, y.real(), z.real()),
        fma( (T)x, y.imag(), z.imag())
    };
}
template<class T0, class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
DG_DEVICE
thrust::complex<T> dg_fma( thrust::complex<T> x, T0 y, thrust::complex<T> z)
{
    return {
        fma( x.real(), (T)y, z.real()),
        fma( x.imag(), (T)y, z.imag())
    };
}
template<class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
DG_DEVICE
thrust::complex<T> dg_fma( thrust::complex<T> x, thrust::complex<T> y, thrust::complex<T> z)
{
    thrust::complex<T> out = {
        fma( x.real(), y.real(), z.real()),
        fma( x.real(), y.imag(), z.imag())
    };
    return {
        fma( -x.imag(), y.imag(), out.real()),
        fma( x.imag(), y.real(), out.imag())
    };
}


} // namespace detail
///@endcond
} // namespace dg
