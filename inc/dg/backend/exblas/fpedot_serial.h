/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Matthias Wiesenberger, 2020, within FELTOR license
 */
/**
 *  @file fpedot_serial.h
 *  @brief Serial version of fpedot
 *
 *  @authors
 *    Developers : \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#pragma once

#include <array>
#include "accumulate.h"

namespace dg
{
namespace exblas{

/*!@brief serial version of extended precision general dot product
 *
 * Computes the extended precision reduction \f[ \sum_{i=0}^{N-1} f(x_{0i}, x_{1i}, ... )\f]
 * using floating point expansions
 * @tparam T the return type of \c Functor.
 * @tparam N size of the floating point expansion (should be between 3 and 8)
 * @tparam Functor a Functor
 * @tparam PointerOrValues must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is a scalar type. If it is a pointer type,
 *  then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param status 0 indicates success, 2 means the FPE overflowed
 * @param size size of the arrays to sum
 * @param fpe the FPE holding the result (write-only)
 * @param f the functor
 * @param xs_ptr the input arrays to sum
 * @sa \c exblas::cpu::Round  to convert the FPE into a double precision number
*/
template<class T, size_t N, class Functor, class ...PointerOrValues>
void fpedot_cpu(int * status, unsigned size, std::array<T,N>& fpe, Functor f, PointerOrValues ...xs_ptr)
{
    for( unsigned i=0; i<N; i++)
        fpe[i] = T(0);
    for(unsigned i = 0; i < size; i++) {
        T res = f( cpu::get_element( xs_ptr, i)...);
        cpu::Accumulate(res, fpe, status);
    }
}


}//namespace exblas
} //namespace dg
