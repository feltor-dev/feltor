#ifndef _COEFF_
#define _COEFF_

#include "matrix.h"
#include "quadmat.h"
#include "exceptions.h"
#include <iostream>

//check for function overloads vs. specialisation
namespace toefl{

    /*! @brief partial specialisation of QuadMat for Matrix<T> 
     *
     * The QuadMat stores its values inside so T and n should be of small size
     * @tparam T double, complex<double> 
     * @tparam n size of the Matrix, assumed to be small (either 2 or 3)
     */
    template <class T, size_t n>
    class QuadMat<Matrix<T, TL_NONE>, n>
    {
      private:
        Matrix<T, TL_NONE> * data[n*n]; //data is an array of pointers to Matrix
        const size_t rows, cols;
      public:
        QuadMat(const size_t rows, const size_t cols):rows(rows), cols(cols) {
            for( size_t i = 0; i< n*n; i++)
                data[i] = new Matrix<T, TL_NONE>(rows, cols); 
        }
        QuadMat( const QuadMat& src): rows(src.rows), cols(src.cols){
            for( size_t i = 0; i< n*n; i++)
                data[i] = new Matrix<T, TL_NONE>(rows, cols); 
            for( size_t i = 0; i < n; i++)
                for( size_t j = 0; j < n; j++)
                    *(data[i*n + j]) = *(src.data[i*n + j]);
        }
        const QuadMat& operator=( const QuadMat& rhs){
#ifdef TL_DEBUG
            if( rows != rhs.rows || cols != rhs.cols)
                throw Message( "Assignment of QuadMat not possible. Sizes not equal\n", ping);
#endif
            for( size_t i = 0; i < n; i++)
                for( size_t j = 0; j < n; j++)
                    (*(data[i*n + j])) = (*(rhs.data[i*n + j]));

            return *this;
        }
        ~QuadMat()
        {
            for( size_t i = 0; i< n*n; i++)
                delete data[i]; 
        }
    
        Matrix<T, TL_NONE>& operator()(const size_t i, const size_t j) {
    #ifdef _TL_DEBUG_
            if( i >= n || j >= n)
                throw BadIndex( i, n, j, n, ping)
    #endif
            return *(data[ i*n+j]);
        }
    
        const Matrix<T, TL_NONE>& operator()(const size_t i, const size_t j) const {
    #ifdef _TL_DEBUG_
            if( i >= n || j >= n)
                throw BadIndex( i, n, j, n, ping)
    #endif
            return *(data[ i*n+j]);
        }
    };
        
    
    
    
    
    /*! @brief this function inverts the coefficient matrix and places them in a container  
     *
     * @tparam T double or complex<double>
     * @tparam F the Functor class or function type (provides F( QuadMat<T, n>, kx, ky) 
     *          and will be called for kx = 0..nrows, ky = 0..ncols
     * @tparam n the size of the problem ( 2 or 3)
     * @param functor is a Functor or function that overloads void operator( QuadMat<T, n>&, size_t, size_t) 
     * @param coeff the Matrix of coefficient Matrices (the size of each has to be (nkx times nky)
     * @param nkx number of rows to calculate
     * @param nky number of columns to calculate
     * \note If you use a transposing fourier transform algorithm the first index of your field matrices must be y
     * and the second x (which is the way you draw a matrix), i.e. the transpose of your coefficient matrices!
     */
    template< class F, class T, size_t n> //f( QuadMat< T, n>, size_t k, size_t l)
    QuadMat< Matrix<T, TL_NONE>, n> const * const make_coeff( const F& f, size_t nkx, size_t nky)
    {
        //DEBUG should test whether every coeff(i,j) has nrows and ncols
        QuadMat<Matrix<T, TL_NONE>, n>* const coeff = new QuadMat< Matrix<T, TL_NONE>, n>(nkx, nky);
        QuadMat<T,n> temp;
        for( size_t k = 0; k < nkx; k++)
            for( size_t q = 0; q < nky; q++)
            {
                f( temp, k, q);
                invert< T>( temp);
                for( size_t i = 0; i < n; i++)
                    for( size_t j = 0; j < n; j++)
                        ((*coeff)(i,j))(k,q) = temp(i,j);
            }
        return coeff;
    }
    
    //does that compile regarding specialization? NO
    /*
    template < class T, class F>
    inline Matrix<T, TL_NONE>* make_coeff<T, F, 1>( const F& f, size_t nrows, size_t ncols)
    {
        Matrix<T, TL_NONE>* coeff = new Matrix<T, TL_NONE>( nrows, ncols);
        for( size_t k = 0; k < nrows; k++)
            for( size_t q = 0; q < ncols; q++)
                (*coeff)( k, q) /= f( k, q);
        return coeff;
    }
    */
}

#endif //_COEFF_

