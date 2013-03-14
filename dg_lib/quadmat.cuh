/*!
 * @file
 * @brief Implementation of the QuadMat class and the invert function
 * @author Matthias Wiesenberger
 *  Matthias.Wiesenberger@uibk.ac.at
 *
 */
#ifndef _DG_QUADMAT_
#define _DG_QUADMAT_
#include <iostream>
#include <assert.h>

#include "array.cuh"

namespace dg{
/*! @brief POD container for quadratic fixed size matrices
 *
 * The QuadMat stores its values inside the object.
 * i.e. a QuadMat<double, 2> stores four double variables continously
 * in memory. Therefore it is well suited for the use in the Matrix
 * class (because memcpy and memset correctly work on this type)
 * \note T and n should be of small size to reduce object size.
 * \note QuadMat is an aggregate so you can use initializer lists in c++11..
 * @tparam T tested with double and std::complex<double> 
 * @tparam n size of the Matrix, assumed to be small 
 */
template <class T, size_t n>
class QuadMat : public dg::Array<T, n*n>
{
  public:
    /*! @brief No values are assigned*/
    __host__ __device__ QuadMat(){}
    /*! @brief Initialize elements to a value
     *
     * @param value The initial value
     */
    __host__ __device__ QuadMat( const T& value):dg::Array<T, n*n>( value)
    {
    }

    /*! @brief access operator
     *
     * Performs a range check if DG_DEBUG is defined
     * @param i row index
     * @param j column index
     * @return reference to value at that location
     */
    __host__ __device__ T& operator()(const size_t i, const size_t j){
#ifdef DG_DEBUG
        assert( i < n && j < n);
#endif
        return (*this)[ i*n+j];
    }
    /*! @brief const access operator
     *
     * Performs a range check if DG_DEBUG is defined
     * @param i row index
     * @param j column index
     * @return const value at that location
     */
    __host__ __device__ const T& operator()(const size_t i, const size_t j) const {
#ifdef DG_DEBUG
        assert( i < n && j < n);
#endif
        return (*this)[ i*n+j];
    }


    /*! @brief puts a matrix linewise in output stream
     *
     * @param os the outstream
     * @param mat the matrix to output
     * @return the outstream
     */
    friend std::ostream& operator<<(std::ostream& os, const QuadMat<T,n>& mat)
    {
        for( size_t i=0; i < n ; i++)
        {
            for( size_t j = 0;j < n; j++)
                os << mat(i,j) << " ";
            os << "\n";
        }
        return os;
    }
    /*! @brief Read values into a Matrix from given istream
     *
     * The values are filled linewise into the matrix. Values are seperated by 
     * whitespace charakters. (i.e. newline, blank, etc)
     * @param is The istream
     * @param mat The Matrix into which the values are written
     * @return The istream
     */
    friend std::istream& operator>> ( std::istream& is, QuadMat<T,n>& mat)
    {
        for( size_t i=0; i<n; i++)
            for( size_t j=0; j<n; j++)
                is >> mat(i, j);
        return is;
    }
};

/*! @brief Return the One Matrix
 * @return Matrix containing ones on the diagonal and zeroes elsewhere
 */
template< size_t n>
QuadMat<double, n> One()
{
    QuadMat<double, n> E(0);
    for( unsigned i=0; i<n; i++)
        E(i,i) = 1;
    return E;
}
/*! @brief Return the Zero QuadMat
 * @return QuadMat containing only zeroes 
 */
template< size_t n>
QuadMat<double, n> Zero()
{
    QuadMat<double, n> E(0);
    return E;
}

} //namespace dg
#endif //_DG_QUADMAT_
