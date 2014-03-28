#ifndef _TL_MATRIX_ARRAY_
#define _TL_MATRIX_ARRAY_
#include <array>
#include "matrix.h"

namespace toefl{

/*! @brief Make an array of matrices 
 *
 * @ingroup containers
 * @tparam T same as Matrix
 * @tparam P same as Matrix
 * @tparam n Size of the array to be constructed
 * @attention This class exists mainly for internal reasons!
 */
template< class T, enum Padding P, size_t n>
struct MatrixArray
{
    /*! @brief Construct and return an array of Matrices
     *
     * @param rows Rows of each Matrix
     * @param cols Columns of each Matrix
     * @param value initial value of matrices
     * @return An Array of Matrices
     */
    static std::array<Matrix<T, P>,n> construct( size_t rows, size_t cols, T value=(T)0);
};
///@cond
template< class T, enum Padding P>
struct MatrixArray<T,P,1>
{
    static std::array<Matrix<T,P>,1> construct( size_t rows, size_t cols, T value=(T)0)
    {
        std::array<Matrix<T,P>,1> a{{
            Matrix<T,P>( rows, cols, value)
        }};
        return a;
    }
};



template< class T, enum Padding P>
struct MatrixArray<T,P, 2>
{
    static std::array<Matrix<T,P>,2> construct( size_t rows, size_t cols, T value=(T)0)
    {
        std::array<Matrix<T,P>,2> a{{
            Matrix<T,P>( rows, cols, value),
            Matrix<T,P>( rows, cols, value)
        }};
        return a;
    }
};

template< class T, enum Padding P>
struct MatrixArray<T,P,3>
{
    static std::array<Matrix<T,P>,3> construct( size_t rows, size_t cols, T value=(T)0)
    {
        std::array<Matrix<T,P>,3> a{{ 
            Matrix<T,P>( rows, cols, value),
            Matrix<T,P>( rows, cols, value), 
            Matrix<T,P>( rows, cols, value)
        }};
        return a;
    }
};
template< class T, enum Padding P>
struct MatrixArray<T,P,4>
{
    static std::array<Matrix<T,P>,4> construct( size_t rows, size_t cols, T value=(T)0)
    {
        std::array<Matrix<T,P>,4> a{{ 
            Matrix<T,P>( rows, cols, value),
            Matrix<T,P>( rows, cols, value),
            Matrix<T,P>( rows, cols, value), 
            Matrix<T,P>( rows, cols, value)
        }};
        return a;
    }
};
template< class T, enum Padding P>
struct MatrixArray<T,P,5>
{
    static std::array<Matrix<T,P>,5> construct( size_t rows, size_t cols, T value=(T)0)
    {
        std::array<Matrix<T,P>,5> a{{ 
            Matrix<T,P>( rows, cols, value),
            Matrix<T,P>( rows, cols, value),
            Matrix<T,P>( rows, cols, value), 
            Matrix<T,P>( rows, cols, value), 
            Matrix<T,P>( rows, cols, value)
        }};
        return a;
    }
};
template< class T, enum Padding P>
struct MatrixArray<T,P,6>
{
    static std::array<Matrix<T,P>,6> construct( size_t rows, size_t cols, T value=(T)0)
    {
        std::array<Matrix<T,P>,6> a{{ 
            Matrix<T,P>( rows, cols, value),
            Matrix<T,P>( rows, cols, value),
            Matrix<T,P>( rows, cols, value), 
            Matrix<T,P>( rows, cols, value),
            Matrix<T,P>( rows, cols, value), 
            Matrix<T,P>( rows, cols, value)
        }};
        return a;
    }
};
///@endcond

}//namespace toefl

#endif //_TL_MATRIX_ARRAY_

