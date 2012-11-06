/*!
 * @file
 * @author Matthias Wiesenberger
 * @email Matthias.Wiesenberger@uibk.ac.at
 * 
 */
#ifndef _COEFF_
#define _COEFF_

#include "matrix.h"
#include "quadmat.h"
#include "exceptions.h"
#include <iostream>

namespace toefl{

    /*! @brief This function inverts the coefficient matrix and places them in a container  
     *
     * If in the system Lv = w of dimension n
     * you have the matrix of coefficients L then this functions computes L^-1.
     * @tparam T double or complex<double>
     * @tparam F the Functor class or function type (provides F( QuadMat<T, n>, size_t, size_t) 
     * @tparam n the size of the problem ( 2 or 3)
     * @param f_T will be called f_T( kx = 0..coeff_T.rows()-1, ky = 0..coeff_T.cols()-1)
     * @param coeff_T the Matrix of coefficient Matrices (the size of each has to be (n times n)
     * \note If you use a transposing fourier transform algorithm the first index of your coefficient matrices must be x in our convention
     * and the second y i.e. the transpose of your field matrices!
     */
    template< class F, class T, size_t n> //f( QuadMat< T, n>, size_t k, size_t l)
    void invert_coeff( F& f_T, Matrix< QuadMat<T,n>, TL_NONE>& coeff_T)
    {
        const size_t nkx = coeff.rows(), nky = coeff.cols();
        QuadMat<T,n> temp;
        for( size_t k = 0; k < nkx; k++)
            for( size_t q = 0; q < nky; q++)
            {
                f_T( temp, k, q);
                invert< T>( temp);
                coeff_T(k,q) = temp;
            }
    }

    /*! @brief Pointwise multiply a coefficient Matrix with a Matrix
     * 
     * Useful for multiplications in fourier space: m_T(kx, ky)*c_T(kx,ky).
     * @tparam T1 type of the coefficients i.e. double or std::complex<double>
     * @tparam T type of the matrix elements, default is std::complex<double>
     * @param c the coefficient matrix 
     * @param m0 contains solution on output
     */
    template< typename T1, typename T = std::complex<double> >
    void multiply_coeff( const Matrix< T1, TL_NONE>& c, Matrix< T, TL_NONE>& m0)
    {
#ifdef TL_DEBUG
        if( c.rows() != m0.rows() || c.cols() != m0.cols())
            throw Message( "Cannot multiply coefficients! Sizes not equal!", ping);
        if( c.isVoid() || m0.isVoid())
            throw Message( "Cannot work with void Matrices!\n", ping);
#endif
        for( size_t i = 0; i<c.rows(); i++)
            for( size_t j=0; j<c.cols(); j++)
                m0(i,j) *= c(i,j);
    }


    /*! @brief pointwise multiply the 2 x 2 Matrix of coefficients by a 2-vector of matrices  
     *
     * Compute the system m0 = c00*m0 + c01*m1, m1 = c11*m0 + c10*m1 where all
     * of the elements are matrices and matrix-matrix multiplications are done pointwise.
     * @tparam T1 type of the coefficients i.e. double or std::complex<double>
     * @tparam T type of the matrix elements, default is std::complex<double>
     * @param c the coefficient matrix 
     * @param m0 zeroth element of the vector, contains solution on output
     * @param m1 first element of the vector, contains solution on output
     */
    template< typename T1, typename T = std::complex<double> >
    void multiply_coeff( const Matrix< QuadMat< T1,2>, TL_NONE>& c, Matrix< T, TL_NONE>& m0, Matrix< T, TL_NONE>& m1)
    {
#ifdef TL_DEBUG
        if( c.rows() != m0.rows() || m0.rows() != m1.rows())
            if( c.cols() != m0.cols() || m0.cols() != m1.cols())
                throw Message( "Cannot multiply coefficients! Sizes not equal!", ping);
        if( c.isVoid() || m0.isVoid() || m1.isVoid())
            throw Message( "Cannot work with void Matrices!\n", ping);
#endif
        const size_t rows = c.rows(), cols = c.cols();
        T temp;
        for( size_t i = 0; i<rows; i++)
            for( size_t j=0; j<cols; j++)
            {
                temp = m0(i, j);
                m0(i,j) = c(i,j)(0,0)*m0(i,j) + c(i,j)(0,1)*m1(i,j);
                m1(i,j) = c(i,j)(1,0)*temp    + c(i,j)(1,1)*m1(i,j);
            }
    }
    /*! @brief pointwise multiply the 3 x 3 Matrix of coefficients by a 3-vector of matrices  
     *
     * Compute the system m0 = c00*m0 + c01*m1 + c02*m2, m1 = ... where all
     * of the elements are matrices and matrix-matrix multiplications are done pointwise.
     * @tparam T1 type of the coefficients i.e. double or std::complex<double>
     * @tparam T type of the matrix elements, i.e. double or std::complex<double>
     * @param c the coefficient matrix 
     * @param m0 zeroth element of the vector, contains solution on output
     * @param m1 first element of the vector, contains solution on output
     * @param m2 second element of the vector, contains solution on output
     */
    template< typename T1, typename T = std::complex<double> >
    void multiply_coeff( const Matrix< QuadMat<T1,3>, TL_NONE>& c, Matrix< T, TL_NONE>& m0, Matrix<T, TL_NONE>& m1, Matrix<T, TL_NONE>& m2)
    {
#ifdef TL_DEBUG
        if( c.rows() != m0.rows() || m0.rows() != m1.rows())
            if( c.cols() != m0.cols() || m0.cols() != m1.cols())
                throw Message( "Cannot multiply coefficients! T1izes not equal!", ping);
        if( c.isVoid() || m0.isVoid() || m1.isVoid() || m2.isVoid())
            throw Message( "Cannot work with void Matrices!\n", ping);
#endif
        const size_t rows = c.rows(), cols = c.cols();
        T temp0, temp1;
        for( size_t i = 0; i<rows; i++)
            for( size_t j=0; j<cols; j++)
            {
                temp0 = m0(i, j);
                temp1 = m1(i, j);
                m0(i,j) = c(i,j)(0,0)*m0(i,j) + c(i,j)(0,1)*m1(i,j) + c(i,j)(0,2)*m2(i,j);
                m1(i,j) = c(i,j)(1,0)*temp0   + c(i,j)(1,1)*m1(i,j) + c(i,j)(1,2)*m2(i,j);
                m2(i,j) = c(i,j)(2,0)*temp0   + c(i,j)(2,1)*temp1   + c(i,j)(2,2)*m2(i,j);
            }
        
    }

}

#endif //_COEFF_

