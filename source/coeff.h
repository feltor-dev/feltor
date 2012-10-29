#ifndef _COEFF_
#define _COEFF_

#include "matrix.h"
#include "quadmat.h"
#include "exceptions.h"
#include <iostream>

//check for function overloads vs. specialisation
namespace toefl{

    /*! @brief this function inverts the coefficient matrix and places them in a container  
     *
     * @tparam T double or complex<double>
     * @tparam F the Functor class or function type (provides F( QuadMat<T, n>, kx, ky) 
     *          and will be called for kx = 0..nrows, ky = 0..ncols
     * @tparam n the size of the problem ( 2 or 3)
     * @param functor is a Functor or function that overloads void operator( QuadMat<T, n>&, size_t, size_t) 
     * @param coeff the Matrix of coefficient Matrices (the size of each has to be (nkx times nky)
     * \note If you use a transposing fourier transform algorithm the first index of your field matrices must be y
     * and the second x (which is the way you draw a matrix), i.e. the transpose of your coefficient matrices!
     */
    template< class F, class T, size_t n> //f( QuadMat< T, n>, size_t k, size_t l)
    void make_coeff( F& f, Matrix< QuadMat<T,n>, TL_NONE>& coeff)
    {
        const size_t nkx = coeff.rows(), nky = coeff.cols();
        QuadMat<T,n> temp;
        for( size_t k = 0; k < nkx; k++)
            for( size_t q = 0; q < nky; q++)
            {
                f( temp, k, q);
                invert< T>( temp);
                coeff(k,q) = temp;
            }
    }

    template< typename coeff_T, typename matrix_T>
    void multiply_coeff( const Matrix< QuadMat< coeff_T,2>, TL_NONE>& coeff, Matrix< matrix_T, TL_NONE>& m0, Matrix< matrix_T, TL_NONE>& m1)
    {
#ifdef TL_DEBUG
        if( coeff.rows() != m0.rows() || m0.rows() != m1.rows())
            if( coeff.cols() != m0.cols() || m0.cols() != m1.cols())
                throw Message( "Cannot multiply coefficients! Sizes not equal!", ping);
        if( coeff.isVoid() || m0.isVoid() || m1.isVoid())
            throw Message( "Cannot work with void Matrices!\n", ping);
#endif
        const size_t rows = coeff.rows(), cols = coeff.cols();
        matrix_T temp;
        for( size_t i = 0; i<rows; i++)
            for( size_t j=0; j<cols; j++)
            {
                temp = m0(i, j);
                m0(i,j) = coeff(i,j)(0,0)*m0(i,j) + coeff(i,j)(0,1)*m1(i,j);
                m1(i,j) = coeff(i,j)(1,0)*temp    + coeff(i,j)(1,1)*m1(i,j);
            }
    }
    template< typename coeff_T, typename matrix_T>
    void multiply_coeff( const Matrix< QuadMat<coeff_T,3>, TL_NONE>& coeff, Matrix< matrix_T, TL_NONE>& m0, Matrix<matrix_T, TL_NONE>& m1, Matrix<matrix_T, TL_NONE>& m2)
    {
#ifdef TL_DEBUG
        if( coeff.rows() != m0.rows() || m0.rows() != m1.rows())
            if( coeff.cols() != m0.cols() || m0.cols() != m1.cols())
                throw Message( "Cannot multiply coefficients! Sizes not equal!", ping);
        if( coeff.isVoid() || m0.isVoid() || m1.isVoid() || m2.isVoid())
            throw Message( "Cannot work with void Matrices!\n", ping);
#endif
        const size_t rows = coeff.rows(), cols = coeff.cols();
        matrix_T temp0, temp1;
        for( size_t i = 0; i<rows; i++)
            for( size_t j=0; j<cols; j++)
            {
                temp0 = m0(i, j);
                temp1 = m1(i, j);
                m0(i,j) = coeff(i,j)(0,0)*m0(i,j) + coeff(i,j)(0,1)*m1(i,j) + coeff(i,j)(0,2)*m2(i,j);
                m1(i,j) = coeff(i,j)(1,0)*temp0   + coeff(i,j)(1,1)*m1(i,j) + coeff(i,j)(1,2)*m2(i,j);
                m2(i,j) = coeff(i,j)(2,0)*temp0   + coeff(i,j)(2,1)*temp1   + coeff(i,j)(2,2)*m2(i,j);
            }
        
    }

}

#endif //_COEFF_

