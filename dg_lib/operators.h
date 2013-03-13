#ifndef _DG_OPERATORS_
#define _DG_OPERATORS_

#include <iostream>
#include "array.h"

namespace dg{

template< class T, size_t n>
class Operator
{
  public:
    typedef T body_type;
    Operator(){}
    Operator( const T& value)
    {
        for( unsigned i=0; i<n*n; i++)
            ptr[i] = value;
    }
    Operator( const T arr[n][n]) {
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<n; j++)
                ptr[i*n+j] = arr[i][j];
    }
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
    const T& operator()(const size_t i, const size_t j) const {
        return ptr[ i*n+j];
    }

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
    Array<T,n> operator*(const Array<T,n>& arr) 
    {
        Array<T,n> temp{{(T)0}};
        for(unsigned i=0; i<n; i++)
            for( unsigned j=0; j<n; j++)
                temp[i] += ptr[i*n+j]*arr[j];
        return temp;
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
     * @param os the outstream
     * @param mat the matrix to output
     * @return the outstream
     */
    friend std::ostream& operator<<(std::ostream& os, const Operator<T,n>& mat)
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
    friend std::istream& operator>> ( std::istream& is, Operator<T,n>& mat){
        for( size_t i=0; i<n; i++)
            for( size_t j=0; j<n; j++)
                is >> mat(i, j);
        return is;
    }

  private:
    T ptr[n*n];
};


}

#endif //_DG_OPERATORS_
