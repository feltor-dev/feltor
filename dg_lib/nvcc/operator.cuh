#ifndef _DG_OPERATORS_
#define _DG_OPERATORS_

#include "array.cuh"
#include "matrix_traits.h"
#include "matrix_categories.h"

namespace dg{

/**
* @brief Helper class mainly for the assembly of Matrices
*
* @ingroup lowlevel
* In principle it's an enhanced quadratic static array
* but it's not meant for performance critical code. 
* @tparam T value type
* @tparam n size 
*/
template< class T, size_t n>
class Operator
{
  public:
    typedef T value_type; //!< typically double or float 
    typedef Array<T, n> array_type; //type that Operator operates on 
    typedef Array<T, n*n> matrix_type; //used in operator_tuple
    /**
    * @brief Construct empty Operator
    */
    __host__ __device__
    Operator(){}
    /**
    * @brief Initialize elements.
    *
    * @param value Every element is innitialized to.
    */
    __host__ __device__
    Operator( const T& value)
    {
        for( unsigned i=0; i<n*n; i++)
            ptr[i] = value;
    }
    /**
    * @brief Construct from existing array.
    *
    * @param arr Filled 2d array 
    */
    __host__ __device__
    Operator( const T arr[n][n]) {
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<n; j++)
                ptr[i*n+j] = arr[i][j];
    }

    /**
    * @brief Construct from a function.
    *
    * @param f Elements are initialized by calling f with their indices.
    */
    __host__ __device__
    Operator( double (&f)(unsigned, unsigned))
    {
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<n; j++)
                ptr[i*n+j] = f(i,j);
    }

    /*! @brief access operator
     *
     * Performs a range check if TL_DEBUG is defined
     * @param i row index
     * @param j column index
     * @return reference to value at that location
     */
    __host__ __device__
    T& operator()(const size_t i, const size_t j){
        return ptr[ i*n+j];
    }
    /*! @brief const access operator
     *
     * Performs a range check if TL_DEBUG is defined
     * @param i row index
     * @param j column index
     * @return const value at that location
     */
    __host__ __device__
    const T& operator()(const size_t i, const size_t j) const {
        return ptr[ i*n+j];
    }

    /**
    * @brief Transposition
    *
    * @return  A newly generated Operator containing the transpose.
    */
    Operator transpose() const 
    {
        double temp;
        Operator o(*this);
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<i; j++)
            {
                temp = o.ptr[i*n+j];
                o.ptr[i*n+j] = o.ptr[j*n+i];
                o.ptr[j*n+i] = temp;
            }
        return o;
    }

    Operator operator-() const
    {
        Operator temp;
        for( unsigned i=0; i<n*n; i++)
            temp.ptr[i] = -ptr[i];
        return temp;
    } 
    Operator& operator+=( const Operator& op)
    {
        for( unsigned i=0; i<n*n; i++)
            ptr[i] += op.ptr[i];
        return *this;
    }
    Operator& operator-=( const Operator& op)
    {
        for( unsigned i=0; i<n*n; i++)
            ptr[i] -= op.ptr[i];
        return *this;
    }
    Operator& operator*=( const T& value )
    {
        for( unsigned i=0; i<n*n; i++)
            ptr[i] *= value;
        return *this;
    }
    friend Operator operator+( const Operator& lhs, const Operator& rhs) 
    {
        Operator temp(lhs); 
        temp+=rhs;
        return temp;
    }
    friend Operator operator-( const Operator& lhs, const Operator& rhs)
    {
        Operator temp(lhs); 
        temp-=rhs;
        return temp;
    }
    friend Operator operator*( const T& value, const Operator& rhs )
    {
        Operator temp(rhs); 
        temp*=value;
        return temp;
    }
    friend Operator operator*( const Operator& lhs, const T& value)
    {
        return  value*lhs;
    }
    friend Operator operator*( const Operator& lhs, const Operator& rhs)
    {
        Operator temp;
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<n; j++)
            {
                temp.ptr[i*n+j] = (T)0;
                for( unsigned k=0; k<n; k++)
                    temp.ptr[i*n+j] += lhs.ptr[i*n+k]*rhs.ptr[k*n+j];
            }
        return temp;
    }
    /*! @brief puts a matrix linewise in output stream
     *
     * @tparam Ostream The stream e.g. std::cout
     * @param os the outstream
     * @param mat the matrix to output
     * @return the outstream
     */
    template< class Ostream>
    friend Ostream& operator<<(Ostream& os, const Operator<T,n>& mat)
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
     * whitespace characters. (i.e. newline, blank, etc)
     * @tparam Istream The stream e.g. std::cin
     * @param is The istream
     * @param mat The Matrix into which the values are written
     * @return The istream
     */
    template< class Istream>
    friend Istream& operator>> ( Istream& is, Operator<T,n>& mat){
        for( size_t i=0; i<n; i++)
            for( size_t j=0; j<n; j++)
                is >> mat(i, j);
        return is;
    }

  private:
    T ptr[n*n];
};

///@cond
template< class T, size_t n>
struct MatrixTraits< Operator<T, n> >
{
    typedef T value_type;
    typedef typename Operator<T, n>::array_type operand_type;
    typedef OperatorMatrixTag matrix_category;
};

///@endcond

}

#endif //_DG_OPERATORS_
