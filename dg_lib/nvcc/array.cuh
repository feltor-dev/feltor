#ifndef _DG_ARRAY_CUH
#define _DG_ARRAY_CUH

#include <iostream>
#include <cassert>

namespace dg{

template< class T, size_t n>
class Array
{
  public:
    /*! @brief No values are assigned*/
    __host__ __device__ Array(){}
    /*! @brief Initialize elements to a value
     *
     * @param value The initial value
     */
    __host__ __device__ Array( const T& value)
    {
        for( unsigned i=0; i<n; i++)
            data[i] = value;
    }

    /*! @brief set memory to 0
     */
    __host__ __device__ void zero()
    {
        for( size_t i = 0; i < n; i++)
            data[i] = 0;
    }

    __host__ __device__ size_t size() const {return n;}

    /*! @brief access operator
     *
     * Performs a range check if DG_DEBUG is defined
     * @param i row index
     * @return reference to value at that location
     */
    __host__ __device__ T& operator[](const size_t i){
#ifdef DG_DEBUG
        assert( i < n);
#endif
        return data[ i];
    }
    /*! @brief const access operator
     *
     * Performs a range check if DG_DEBUG is defined
     * @param i row index
     * @return const reference to value at that location
     */
    __host__ __device__ const T& operator[](const size_t i) const {
#ifdef DG_DEBUG
        assert( i < n );
#endif
        return data[ i];
    }


    friend std::ostream& operator<<(std::ostream& os, const Array<T,n>& mat)
    {
        for( size_t j = 0;j < n; j++)
            os << mat[j] << " ";
        os << "\n";
        return os;
    }
    /*! @brief Read values into an Array from given istream
     *
     * The values are filled linewise into the matrix. Values are seperated by 
     * whitespace charakters. (i.e. newline, blank, etc)
     * @param is The istream
     * @param mat The Array into which the values are written
     * @return The istream
     */
    friend std::istream& operator>> ( std::istream& is, Array<T,n>& mat)
    {
        for( size_t j=0; j<n; j++)
            is >> mat[j];
        return is;
    }
    friend __host__ __device__ void daxpby( T alpha, const Array& x, T beta, Array& y)
    {
        for( unsigned i=0; i<n; i++)
            y.data[i] = alpha*x.data[i] + beta*y.data[i];
    }

  private:
    T data[n];
};

} //namespace dg

#endif //_DG_ARRAY_CUH
