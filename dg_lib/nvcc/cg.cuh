#ifndef _DG_CG_
#define _DG_CG_

#include <cmath>

#include "blas.h"

namespace dg{

// TO DO: check for better stopping criteria using condition number estimates
/*!@brief Functor class for the conjugate gradient method

 @ingroup algorithms
 The Matrix and Vector class are assumed to be double valued
 @tparam Matrix The matrix class: no requirements except for the 
            BLAS routines
 @tparam Vector The Vector class: needs to model Assignable 

 The following 3 pseudo - BLAS routines need to be callable:
 \li double dot = BLAS1<Vector>::ddot( v1, v2);  
 \li BLAS1<Vector>::daxpby( alpha, x, beta, y);  
 \li BLAS2<Matrix, Vector> dsymv( m, x, y); 

 @note We don't use cusp because cusp allocates memory in every call to the solution method
*/
template< class Matrix, class Vector>
class CG
{
  public:
      //copy must be convertible to Vector
      /**
       * @brief Reserve memory for the cg method
       *
       * @param copy A Vector must be copy-constructible from copy. copy needs to have
       the same size as the vectors in the solution method.
       * @param max_iter Maximum number of iterations to be used
       */
    CG( const Vector& copy, unsigned max_iter):r(copy), p(r), ap(r), max_iter(max_iter){}
    /**
     * @brief Set the maximum number of iterations 
     *
     * @param new_max New maximum number
     */
    void set_max( unsigned new_max) {max_iter = new_max;}
    /**
     * @brief Get the current maximum number of iterations
     *
     * @return the current maximum
     */
    unsigned get_max() {return max_iter;}
    /**
     * @brief Solve the system A*x = b using a conjugate gradient method
     *
     * In every iteration the BLAS functions are called: \n
     *  dsymv 1x, ddot 2x, daxpy 3x
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output
     * @param b The right hand side vector. x and b may be the same vector.
     * @param eps The relative error to be respected
     *
     * @return Number of iterations used to achieve desired precision
     */
    unsigned operator()( const Matrix& A, Vector& x, const Vector& b, double eps = 1e-12);
  private:
    Vector r, p, ap; 
    unsigned max_iter;
};

template< class Matrix, class Vector>
unsigned CG<Matrix, Vector>::operator()( const Matrix& A, Vector& x, const Vector& b, double eps)
{
    double nrm2b = BLAS1<Vector>::ddot(b,b);
    //r = b; BLAS2<Matrix, Vector>::dsymv( -1., A, x, 1.,r); //compute r_0 
    //compute r <- -Ax+b
    BLAS2<Matrix, Vector>::dsymv( A,x,r);
    BLAS1<Vector>::axpby( 1., b, -1., r);
    p = r;
    double nrm2r_old = BLAS1<Vector>::ddot( r, r); //and store the norm of it
    double alpha, nrm2r_new;
    for( unsigned i=1; i<max_iter; i++)
    {
        BLAS2<Matrix, Vector>::dsymv( A, p, ap);
        alpha = nrm2r_old /BLAS1<Vector>::ddot( p, ap);
        BLAS1<Vector>::daxpby( alpha, p, 1.,x);
        BLAS1<Vector>::daxpby( -alpha, ap, 1., r);
        nrm2r_new = BLAS1<Vector>::ddot( r,r);
        if( sqrt( nrm2r_new/nrm2b) < eps) 
            return i;
        BLAS1<Vector>::daxpby(1., r, nrm2r_new/nrm2r_old, p );
        nrm2r_old=nrm2r_new;
    }
    return max_iter;
}

/**
* @brief Functor class for the preconditioned conjugate gradient method
*
 @ingroup algorithms
 The Matrix, Vector and Preconditioner classes are assumed to be double valued
 @tparam Matrix The matrix class: no requirements except for the 
            BLAS routines
 @tparam Vector The Vector class: needs to model Assignable 
 @tparam Preconditioner no requirements except for the blas routines

 The following 3 pseudo - BLAS routines need to be callable 
 \li double dot = BLAS1<Vector>::ddot( v1, v2); 
 \li BLAS1<Vector>::daxpby( alpha, x, beta, y);  
 \li BLAS2<Matrix, Vector> dsymv( m, x, y);     
 \li double dot = BLAS2< Preconditioner, Vector>::ddot( P, v); 
 \li BLAS2< Preconditioner, Vector>::dsymv( alpha, P, x, beta, y);
*/
template< class Matrix, class Vector, class Preconditioner>
class PCG
{
  public:
      /**
       * @brief Reserve memory for the pcg method
       *
       * @param copy A Vector must be copy-constructible from copy
       * @param max_iter Maximum number of iterations to be used
       */
    PCG( const Vector& copy, unsigned max_iter):r(copy), p(r), ap(r), max_iter(max_iter){}
    /**
     * @brief Set the maximum number of iterations 
     *
     * @param new_max New maximum number
     */
    void set_max( unsigned new_max) {max_iter = new_max;}
    /**
     * @brief Get the current maximum number of iterations
     *
     * @return the current maximum
     */
    unsigned get_max() {return max_iter;}
    /**
     * @brief Solve the system A*x = b using a preconditioned conjugate gradient method
     *
     * In every iteration the BLAS functions are called: \n
       dsymv 1x, ddot 1x, daxpy 2x, Prec. ddot 1x, Prec. dsymv 1x
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param eps The relative error to be respected
     *
     * @return Number of iterations used to achieve desired precision
     */
    unsigned operator()( const Matrix& A, Vector& x, const Vector& b, const Preconditioner& P, double eps = 1e-12);
  private:
    Vector r, p, ap; 
    unsigned max_iter;
};

/*
    compared to unpreconditioned compare
    ddot(r,r), axpby()
    to 
    ddot( r,P,r), dsymv(P)
    i.e. it will be slower, if P needs to be stored
    (but in our case P_{ii} can be computed directly
    compared to normal preconditioned compare
    ddot(r,P,r), dsymv(P)
    to
    ddot(r,z), dsymv(P), axpby(), (storage for z)
    i.e. it's surely faster if P contains no more elements than z 
    (which is the case for diagonal scaling)
    NOTE: the same comparison hold for A with the result that A contains 
    significantly more elements than z whence ddot(r,A,r) is far slower than ddot(r,z)
*/
template< class Matrix, class Vector, class Preconditioner>
unsigned PCG< Matrix, Vector, Preconditioner>::operator()( const Matrix& A, Vector& x, const Vector& b, const Preconditioner& P, double eps)
{
    double nrm2b = BLAS2<Preconditioner, Vector>::ddot( P,b);
    //r = b; BLAS2<Matrix, Vector>::dsymv( -1., A, x, 1.,r); //compute r_0 
    BLAS2<Matrix, Vector>::dsymv( A,x,r);
    BLAS1<Vector>::daxpby( 1., b, -1., r);
    BLAS2<Preconditioner, Vector>::dsymv( P, r, p );//<-- compute p_0
    double nrm2r_old = BLAS2<Preconditioner, Vector>::ddot( P,r); //and store the norm of it
    double alpha, nrm2r_new;
    for( unsigned i=1; i<max_iter; i++)
    {
        BLAS2<Matrix, Vector>::dsymv( A, p, ap);
        alpha = nrm2r_old /BLAS1<Vector>::ddot( p, ap);
        BLAS1<Vector>::daxpby( alpha, p, 1.,x);
        BLAS1<Vector>::daxpby( -alpha, ap, 1., r);
        nrm2r_new = BLAS2<Preconditioner, Vector>::ddot( P, r); //<--
        if( sqrt( nrm2r_new/nrm2b) < eps) 
            return i;
        BLAS2<Preconditioner, Vector>::dsymv(1.,P, r, nrm2r_new/nrm2r_old, p );//<--
        nrm2r_old=nrm2r_new;
    }
    return max_iter;
}


} //namespace dg



#endif //_DG_CG_
