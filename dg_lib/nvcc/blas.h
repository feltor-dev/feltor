#ifndef _DG_BLAS_
#define _DG_BLAS_

namespace dg{

//CUDA relevant: BLAS routines must block until result is ready 
/*! @brief BLAS Level 1 routines 
 *
 * @ingroup blas
 * Only those routines that are actually called need to be implemented.
 * Don't forget to specialize in the dg namespace
 */
template < class VectorClass>
struct BLAS1
{
    /**
     * @brief The Vector class
     */
    typedef VectorClass Vector;
    /*! @brief Euclidean dot product between two Vectors
     *
     * This routine computes \f[ x^T y = \sum_{i=0}^{N-1} x_i y_i \f]
     * @param x Left Vector
     * @param y Right Vector might equal Left Vector
     * @return Scalar product
     */
    static double ddot( const Vector& x, const Vector& y)
    {
        return BLAS1<Vector>::ddot( x, y, Vector::vector_category);
    }
    /*! @brief Modified BLAS 1 routine daxpy
     *
     * This routine computes \f[ y =  \alpha x + \beta y \f] 
     * Q: Isn't it better to implement daxpy and daypx? \n
     * A: unlikely, because in all three cases all elements of x and y have to be loaded
     * and daxpy is memory bound. (Is there no original daxpby routine because 
     * the name is too long??)
     * @param alpha Scalar  
     * @param x Vector x migtht equal y 
     * @param beta Scalar
     * @param y Vector y contains solution on output
     * @note In an implementation you may want to check for alpha == 0 and beta == 1
     */
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y);
};

/*! @brief BLAS Level 2 routines 
 *
 * @ingroup blas
 * In an implementation Vector and Matrix should be typedefed.
 * Only those routines that are actually called need to be implemented.
 */
template < class MatrixClass, class VectorClass>
struct BLAS2
{

    /**
     * @brief The Vector class
     */
    typedef VectorClass Vector;
    typedef MatrixClass Matrix; //!< The Matrix class
    /*! @brief Symmetric Matrix Vector product
     *
     * This routine computes \f[ y = \alpha M x + \beta y \f]
     * where \f[ M\f] is a symmetric matrix. 
     * @param alpha A Scalar
     * @param m The Matrix
     * @param x A Vector different from y (except in the case where m is diagonal)
     * @param beta A Scalar
     * @param y contains solution on output
     * @note In an implementation you may want to check for alpha == 0 and beta == 1
     */
    static void dsymv( double alpha, const Matrix& m, const Vector& x, double beta, Vector& y);
    /*! @brief Symmetric Matrix Vector product
     *
     * This routine is equivalent to dsymv( 1., m, x, 0., y);
     * @param m The Matrix
     * @param x A Vector different from y (except in the case where m is diagonal)
     * @param y contains solution on output
     */
    static void dsymv( const Matrix& m, const Vector& x, Vector& y);
    /*! @brief General dot produt
     *
     * This routine computes the scalar product defined by the symmetric positive definit 
     * matrix P \f[ x^T P y = \sum_{i=0}^{N-1} x_i P_{ij} y_j \f]
     * where P is a diagonal matrix. (Otherwise it would be more efficient to 
     * precalculate \f[ Py\f] and then call the BLAS1::ddot routine!
     * @param x Left Vector
     * @param P The diagonal Matrix
     * @param y Right Vector might equal Left Vector
     * @return Generalized scalar product
     */
    static double ddot( const Vector& x, const Matrix& P, const Vector& y);
    /*! @brief General dot produt
     *
     * This routine is equivalent to the call ddot( x, P, x)
     * @param P The diagonal Matrix
     * @param x Right Vector
     * @return Generalized scalar product
     */
    static double ddot( const Matrix& P, const Vector& x);
};


} //namespace dg

#endif //_DG_BLAS_
