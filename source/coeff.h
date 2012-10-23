#ifndef _COEFF_
#define _COEFF_

#include "exceptions.h"

//check for function overloads vs. specialisation
namespace toefl{

    /*! @brief Container for quadratic fixed size matrices
     *
     * The QuadMat stores its values on the stack so it should be of small size
     * @tparam T double, complex<double> or Matrix<T>
     * @tparam n size of the Matrix, assumed to be small (either 2 or 3)
     */
    template <class T, size_t n>
    class QuadMat
    {
      private:
        T data[n*n];
      public:
        QuadMat(){}
        QuadMat(T value){
            for( size_t i; i < n; i++)
                for( size_t j; j < n; j++)
                    data[i][j] = T;
        }
        QuadMat( QuadMat& src){
            for( size_t i; i < n; i++)
                for( size_t j; j < n; j++)
                    data[i][j] = src.data[i][j];
        }
        QuadMat& operator=( const QuadMat& rhs){
            for( size_t i; i < n; i++)
                for( size_t j; j < n; j++)
                    data[i][j] = rhs.data[i][j];
            return *this;
        }
        QuadMat& operator=( const T value){
            for( size_t i; i < n; i++)
                for( size_t j; j < n; j++)
                    data[i][j] = value;
            return *this;
        }
    
        T& operator()(const size_t i, const size_t j){
    #ifdef _TL_DEBUG_
            if( i >= n || j >= n)
                throw BadIndex( i, n, j, n, ping)
    #endif
            return data[ j*n+i];
        }
    
        const T& operator()(const size_t i, const size_t j) const {
    #ifdef _TL_DEBUG_
            if( i >= n || j >= n)
                throw BadIndex( i, n, j, n, ping)
    #endif
            return data[ j*n+i];
        }
        /*! @brief puts a matrix linewise in output stream
         *
         * @param os the outstream
         * @param mat the matrix to output
         * @return the outstream
         */
        ostream& operator<<(ostream& os, const QuadMat& mat)
        {
            for( size_t i=0; i < n ; i++)
            {
                for( size_t j = 0;j < n; j++)
                    os << mat[i][j] << " ";
                os << "\n";
            }
            return os;
        }
    };
        
    
    
    
    /*! @brief inverts a 2x2 matrix of given type
     *
     * @tparam The type must support basic algorithmic functionality (i.e. +, -, * and /)
     * @param m the matrix contains its invert on return
     */
    template<class T>
    void invert(QuadMat<T, 2>& m)
    {
        T det, temp;
        det = m(0,0)*m(1,1) - m(0,1)*m(1,0);
    #ifdef _TL_DEBUG_
        if( det==0) throw Message("Determinant is Zero\n", ping);
    #endif
        temp = m(0,0);
        m(0,0) = m(1,1)/det;
        m(0,1) /=-det;
        m(1,0) /=-det;
        m(1,1) = temp/det;
    }
    
    /*! @brief inverts a 3x3 matrix of given type
     *
     * (overloads the 2x2 version)
     * @tparam The type must support basic algorithmic functionality (i.e. +, -, * and /)
     * @param m the matrix contains its invert on return
     */
    template< typename T>
    void invert( QuadMat< T, 3>& m)
    {
        T det, temp00, temp01, temp02, temp10, temp11, temp20;
        det = m(0,0)*(m(1,1)*m(2,2)-m(2,1)*m(1,2))+m(0,1)*(m(1,2)*m(2,0)-m(1,0)*m(2,2))+m(0,2)*(m(1,0)*m(2,1)-m(2,0)*m(1,1));
    #ifdef _TL_DEBUG_
        if( det==0) throw Message("Determinant is Zero\n", ping);
    #endif
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
    
    
    //Data_t is either double or complex or any type that supports +,-,* and /
    //Functor_t is a Functor or function that overloads void operator( array< array< complex<double>, n>, n>&, int, int)
    // n can be 2 or 3
    /*! @brief this function inverts the coefficient matrix and places them in a container  
     *
     * @tparam T double or complex<double>
     * @tparam F the Functor class or function type
     * @tparam n the size of the problem ( 2 or 3)
     * @param functor is a Functor or function that overloads void operator( QuadMat<T, n>&, size_t, size_t) 
     * @param coeff the Matrix of coefficient arrays
     * @param nrows number of rows to calculate
     * @param ncols number of columns to calculate
     */
    template< class T, class F, size_t n> //f( QuadMat< T, n>, size_t k, size_t l)
    void make_coeff( QuadMat< Matrix<T>, n>& coeff, const F& f, size_t nrows, size_t ncols)
    {
        //DEBUG should test whether every coeff(i,j) has nrows and ncols
        QuadMat<T,n> temp;
        for( size_t k = 0; k < nrows; k++)
            for( size_t q = 0; q < ncols; q++)
            {
                f( temp, k, q);
                invert< T, n>( temp);
                for( size_t i = 0; i < n; k++)
                    for( size_t j = 0; j < n; q++)
                        (coeff(i,j))(k,q) = temp(i,j);
            }
    }
    
    //does that compile regarding specialization?
    template < class T, class F>
    inline void make_coeff<T, F, 1>( Matrix<T>& coeff, const F& f, size_t nrows, size_t ncols)
    {
        for( size_t k = 0; k < nrows; k++)
            for( size_t q = 0; q < ncols; q++)
                coeff( k, q) /= f( k, q);
    }
}

#endif //_COEFF_

