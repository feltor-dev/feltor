#ifndef _DG_CG_
#define _DG_CG_

#include <cmath>

#include "blas.h"

namespace dg{

//// TO DO: check for better stopping criteria using condition number estimates

/**
* @brief Functor class for the preconditioned conjugate gradient method
*
 @ingroup algorithms
 @tparam Vector The Vector class: needs to model Assignable 

 The following 3 pseudo - BLAS routines need to be callable 
 \li double dot = blas1::dot( v1, v2); 
 \li blas1::axpby( alpha, x, beta, y);  
 \li blas2::symv( m, x, y);     
 \li double dot = blas2::dot( P, v); 
 \li blas2::symv( alpha, P, x, beta, y);

 @note Conjugate gradients might become unstable for positive semidefinite
 matrices arising e.g. in the discretization of the periodic laplacian
*/
template< class Vector>
class CG
{
  public:
    typedef typename Vector::value_type value_type;
      /**
       * @brief Reserve memory for the pcg method
       *
       * @param copy A Vector must be copy-constructible from copy
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
    unsigned get_max() const {return max_iter;}
    /**
     * @brief Solve the system A*x = b using a preconditioned conjugate gradient method
     *
     @tparam Matrix The matrix class: no requirements except for the 
            BLAS routines
     @tparam Preconditioner no requirements except for the blas routines. Thus far the dg library
        provides only diagonal preconditioners, which should be enough if the result is extrapolated from
        previous timesteps.
     * In every iteration the following BLAS functions are called: \n
       symv 1x, dot 1x, axpby 2x, Prec. dot 1x, Prec. symv 1x
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param eps The relative error to be respected
     *
     * @return Number of iterations used to achieve desired precision
     */
    template< class Matrix, class Preconditioner >
    unsigned operator()( const Matrix& A, Vector& x, const Vector& b, const Preconditioner& P , value_type eps = 1e-12);
    /**
     * @brief Solve the system A*x = b using unpreconditioned conjugate gradient method
     *
     @tparam Matrix The matrix class: no requirements except for the 
            BLAS routines
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param eps The relative error to be respected
     *
     * @return Number of iterations used to achieve desired precision
     */
    template< class Matrix >
    unsigned operator()( const Matrix& A, Vector& x, const Vector& b, value_type eps = 1e-12)
    {
        return this->operator()( A, x, b, Identity<value_type>(), eps);
    }
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
template< class Vector>
template< class Matrix, class Preconditioner>
unsigned CG< Vector>::operator()( const Matrix& A, Vector& x, const Vector& b, const Preconditioner& P, value_type eps)
{
    value_type nrmb = sqrt( blas2::dot( P, b));
#ifdef DG_DEBUG
    std::cout << "Norm of b "<<nrmb <<"\n";
    std::cout << "Residual errors: \n";
#endif //DG_DEBUG
    if( nrmb == 0)
    {
        blas1::axpby( 1., b, 0., x);
        return 0;
    }
    //r = b; blas2::symv( -1., A, x, 1.,r); //compute r_0 
    blas2::symv( A,x,r);
    cudaThreadSynchronize();
    blas1::axpby( 1., b, -1., r);
    cudaThreadSynchronize();
    blas2::symv( P, r, p );//<-- compute p_0
    value_type nrm2r_old = blas2::dot( P,r); //and store the norm of it
    value_type alpha, nrm2r_new;
    cudaThreadSynchronize();
    for( unsigned i=1; i<max_iter; i++)
    {
        blas2::symv( A, p, ap);
        cudaThreadSynchronize();
        alpha = nrm2r_old /blas1::dot( p, ap);
        blas1::axpby( alpha, p, 1.,x);
        blas1::axpby( -alpha, ap, 1., r);
        cudaThreadSynchronize();
        nrm2r_new = blas2::dot( P, r); 
#ifdef DG_DEBUG
        std::cout << "Absolute "<<sqrt( nrm2r_new) <<"\t ";
        std::cout << "Relative "<<sqrt( nrm2r_new)/nrmb << "\n";
#endif //DG_DEBUG
        if( sqrt( nrm2r_new) < eps*nrmb + eps) 
            return i;
        blas2::symv(1.,P, r, nrm2r_new/nrm2r_old, p );
        nrm2r_old=nrm2r_new;
        cudaThreadSynchronize();
    }
    return max_iter;
}


} //namespace dg



#endif //_DG_CG_
