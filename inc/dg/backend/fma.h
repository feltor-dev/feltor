#pragma once

#include <cmath>
#include <complex>
#include <thrust/complex.h>
#include "config.h"

namespace dg
{
///@cond
namespace detail
{
template<class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
DG_DEVICE
T dg_fma( T x, T y, T z)
{
    return fma( x, y, z);
}
template<class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
std::complex<T> dg_fma( T x, std::complex<T> y, std::complex<T> z)
{
    return {
        std::fma( x, y.real(), z.real()),
        std::fma( x, y.imag(), z.imag())
    };
}
template<class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
std::complex<T> dg_fma( std::complex<T> x, std::complex<T> y, std::complex<T> z)
{
    std::complex<T> out = {
        fma( x.real(), y.real(), z.real()),
        fma( x.real(), y.imag(), z.imag())
    };
    return {
        fma( -x.imag(), y.imag(), out.real()),
        fma( x.imag(), y.real(), out.imag())
    };
}
template<class T, class = std::enable_if_t<std::is_floating_point_v<T> >>
DG_DEVICE
thrust::complex<T> dg_fma( T x, thrust::complex<T> y, thrust::complex<T> z)
{
    return {
        fma( x, y.real(), z.real()),
        fma( x, y.imag(), z.imag())
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
