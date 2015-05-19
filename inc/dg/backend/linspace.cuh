#pragma once
#include <cassert>
/*! @file
Implements 1d linscpace
*/
namespace dg
{
    namespace create
    {
        ///@addtogroup lowlevel
        ///@{
        /**
        *
        * @brief Create equidistant vector
        * @tparam T value type
        * @param t0: start value
        * @param tend: end value
        * @param delta: step value
        *
        * @return Host vector
        **/
        template<typename T>
        dg::DVec linspace(T start, T end, T delta)
        {
            size_t num_el = static_cast<size_t>((end - start) / delta);
            dg::DVec result(num_el);
            for(size_t n = 0; n < num_el; n++)
            {
                result[n] = start + static_cast<T>(n) * delta;
            }
            return(result);
        }
    }
}