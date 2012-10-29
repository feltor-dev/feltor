#ifndef _QUADMAT_
#define _QUADMAT_
#include <iostream>
#include "exceptions.h"

namespace toefl{
    /*! @brief Container for quadratic fixed size matrices
     *
     * The QuadMat stores its values itself 
     * so T and n should be of small size to reduce object size.
     * @tparam T double, complex<double> 
     * @tparam n size of the Matrix, assumed to be small (either 2 or 3)
     */
    template <class T, size_t n>
    class QuadMat
    {
      private:
        T data[n*n];
      public:
        QuadMat(){}
        QuadMat( QuadMat& src){
            for( size_t i=0; i < n*n; i++)
                data[i] = src.data[i];
        }
        const QuadMat& operator=( const QuadMat& rhs){
            for( size_t i = 0; i < n*n; i++)
                data[i] = rhs.data[i];
            return *this;
        }
    
        T& operator()(const size_t i, const size_t j){
    #ifdef TL_DEBUG
            if( i >= n || j >= n)
                throw BadIndex( i, n, j, n, ping);
    #endif
            return data[ i*n+j];
        }
        const T& operator()(const size_t i, const size_t j) const {
    #ifdef TL_DEBUG
            if( i >= n || j >= n)
                throw BadIndex( i, n, j, n, ping);
    #endif
            return data[ i*n+j];
        }
        const bool operator!=( const QuadMat& rhs) const{
            for( size_t i = 0; i < n*n; i++)
                if( data[i] != rhs.data[i])
                    return true;
            return false;
        }
        const bool operator==( const QuadMat& rhs) const {return !((*this != rhs));}
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
    };
    /*! @brief inverts a 2x2 matrix of given type
     *
     * \note throws a Message if Determinant is zero. 
     * @tparam T The type must support basic algorithmic functionality (i.e. +, -, * and /)
     * @param m the matrix contains its invert on return
     */
    template<class T>
    void invert(QuadMat<T, 2>& m) 
    {
        T det, temp;
        det = m(0,0)*m(1,1) - m(0,1)*m(1,0);
        if( det== (T)0) throw Message("Determinant is Zero\n", ping);
        temp = m(0,0);
        m(0,0) = m(1,1)/det;
        m(0,1) /=-det;
        m(1,0) /=-det;
        m(1,1) = temp/det;
    }
    
    /*! @brief inverts a 3x3 matrix of given type
     *
     * (overloads the 2x2 version)
     * \note throws a Message if Determinant is zero. 
     * @tparam The type must support basic algorithmic functionality (i.e. +, -, * and /)
     * @param m the matrix contains its invert on return
     */
    template< typename T>
    void invert( QuadMat< T, 3>& m) 
    {
        T det, temp00, temp01, temp02, temp10, temp11, temp20;
        det = m(0,0)*(m(1,1)*m(2,2)-m(2,1)*m(1,2))+m(0,1)*(m(1,2)*m(2,0)-m(1,0)*m(2,2))+m(0,2)*(m(1,0)*m(2,1)-m(2,0)*m(1,1));
        if( det== (T)0) throw Message("Determinant is Zero\n", ping);
        temp00 = m(0,0);
        temp01 = m(0,1);
        temp02 = m(0,2);
        m(0,0) = (m(1,1)*m(2,2) - m(1,2)*m(2,1))/det;
        m(0,1) = (m(0,2)*m(2,1) - m(0,1)*m(2,2))/det;
        m(0,2) = (temp01*m(1,2) - m(0,2)*m(1,1))/det;
    
        temp10 = m(1,0);
        temp11 = m(1,1);
        m(1,0) = (m(1,2)*m(2,0) - m(1,0)*m(2,2))/det;
        m(1,1) = (temp00*m(2,2) - temp02*m(2,0))/det;
        m(1,2) = (temp02*temp10 - temp00*m(1,2))/det;
    
        temp20 = m(2,0);
        m(2,0) = (temp10*m(2,1) - temp11*m(2,0))/det;
        m(2,1) = (temp01*temp20 - temp00*m(2,1))/det;
        m(2,2) = (temp00*temp11 - temp10*temp01)/det;
    }
}
#endif //_QUADMAT_
