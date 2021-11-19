#ifndef _DG_PCG_
#define _DG_PCG_

#include <cmath>

#include "blas.h"
#include "functors.h"
#include "extrapolation.h"
#include "backend/typedefs.h"

#include "backend/timer.h"

/*!@file
 * Conjugate gradient class and functions
 */

namespace dg{

//// TO DO: check for better stopping criteria using condition number estimates?

/**
* @brief Preconditioned conjugate gradient method to solve
* \f[ Ax=b\f]
*
* where \f$ A\f$ is positive definite and self-adjoint in the weighted scalar product (defined by the diagonal weights matrix \f$W\f$)
* \f[ A^\dagger := \frac{1}{W} A^T W \equiv A\f].
* Note that if \f$ A\f$ is self-adjoint then both \f$ (WA)^T = WA \f$ and \f$ \left(A \frac{1}{W}\right)^T = A\frac{1}{W}\f$ are symmetric.
* The positive definite, self-adjoint preconditioner \f$ P \approx A^{-1}\f$ that approximates the inverse
* of \f$ A\f$ and is fast to apply, is
* used to solve the left preconditioned system
* \f[ PAx=Pb\f]
*
* @ingroup invert
*
* @sa This implements the PCG algorithm (applied to \f$(WA)\f$ as given in
* https://en.wikipedia.org/wiki/Conjugate_gradient_method
* or the book
* <a href="https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf">Iterative Methods for Sparse Linear Systems" 2nd edition by Yousef Saad </a>
* @note Conjugate gradients might become unstable for positive semidefinite
* matrices arising e.g. in the discretization of the periodic laplacian
* @attention beware the sign: a negative definite matrix does @b not work in Conjugate gradient
*
* @snippet cg2d_t.cu doxygen
* @copydoc hide_ContainerType
*/
template< class ContainerType>
class PCG
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    PCG(){}
    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    PCG( const ContainerType& copyable, unsigned max_iterations):
        r(copyable), p(r), ap(r), max_iter(max_iterations){}
    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {max_iter = new_max;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return max_iter;}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return r;}

    ///@brief Set or unset debugging output during iterations
    ///@param verbose If true, additional output will be written to \c std::cout during solution
    void set_verbose( bool verbose){ m_verbose = verbose;}

    /**
    * @brief Perfect forward parameters to one of the constructors
    *
    * @tparam Params deduced by the compiler
    * @param ps parameters forwarded to constructors
    */
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = PCG( std::forward<Params>( ps)...);
    }
    /**
     * @brief Solve \f$ Ax = b\f$ using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||Ax-b||_W < \epsilon( ||b||_W + C) \f$ where \f$C\f$ is
     * the absolute error in units of \f$ \epsilon\f$ and \f$ S \f$ defines a square norm
     * @param A A self-adjoint positive definit matrix with respect to the weights \c W
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector.
     * @param P The preconditioner to be used (an approximation to the inverse of \c A that is fast to compute)
     * @param W Weights that define the scalar product in which \c A and \c P are
     * self-adjoint and in which the error norm is computed.
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * @param test_frequency if set to 1 then the norm of the error is computed
     * in every iteration to test if the loop can be terminated. Sometimes,
     * especially for small sizes the dot product is expensive to compute, then
     * it is beneficial to set this parameter to e.g. 10, which means that the
     * errror condition is only evaluated every 10th iteration.
     *
     * @return Number of iterations used to achieve desired precision
     * @note Required memops per iteration (\c P is assumed vector):
             - 15  reads + 4 writes
             - plus the number of memops for \c A;
     * @copydoc hide_matrix
     * @tparam ContainerTypes must be usable with \c MatrixType and \c ContainerType in \ref dispatch
     * @tparam Preconditioner A type for which the \c blas2::symv(Preconditioner&, ContainerType&, ContainerType&) function is callable.
     */
    template< class MatrixType, class ContainerType0, class ContainerType1, class Preconditioner, class ContainerType2 >
    unsigned solve( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, const ContainerType2& W, value_type eps = 1e-12, value_type nrmb_correction = 1, int test_frequency = 1);
  private:
    ContainerType r, p, ap;
    unsigned max_iter;
    bool m_verbose = false;
};

///@cond
template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner, class ContainerType2>
unsigned PCG< ContainerType>::solve( Matrix& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, const ContainerType2& W, value_type eps, value_type nrmb_correction, int save_on_dots )
{
    // self-adjoint: apply PCG algorithm to (P 1/W) (W A) x = (P 1/W) (W b) : P' A' x = P' b'
    // This effectively just replaces all scalar products with the weighted one
    value_type nrmb = sqrt( blas2::dot( W, b));
    value_type tol = eps*(nrmb + nrmb_correction);
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
    if( m_verbose)
    {
        DG_RANK0 std::cout << "# Norm of S b "<<nrmb <<"\n";
        DG_RANK0 std::cout << "# Residual errors: \n";
    }
    if( nrmb == 0)
    {
        blas1::copy( 0., x);
        return 0;
    }
    blas2::symv( A,x,r);
    blas1::axpby( 1., b, -1., r);
    if( sqrt( blas2::dot(W,r) ) < tol) //if x happens to be the solution
        return 0;
    blas2::symv( P, r, p );
    value_type nrmzr_old = blas2::dot( p,W,r); //and store the scalar product
    value_type alpha, nrmzr_new;
    for( unsigned i=1; i<max_iter; i++)
    {
        blas2::symv( A, p, ap);
        alpha =  nrmzr_old/blas2::dot( p, W, ap);
        blas1::axpby( alpha, p, 1.,x);
        blas1::axpby( -alpha, ap, 1., r);
        if( 0 == i%save_on_dots )
        {
            if( m_verbose)
            {
                DG_RANK0 std::cout << "# Absolute r*W*r "<<sqrt( blas2::dot(W,r)) <<"\t ";
                DG_RANK0 std::cout << "#  < Critical "<<tol <<"\t ";
                DG_RANK0 std::cout << "# (Relative "<<sqrt( blas2::dot(W,r) )/nrmb << ")\n";
            }
            if( sqrt( blas2::dot(W,r)) < tol)
                return i;
        }
        blas2::symv(P,r,ap);
        nrmzr_new = blas2::dot( ap, W, r);
        blas1::axpby(1.,ap, nrmzr_new/nrmzr_old, p );
        nrmzr_old=nrmzr_new;
    }
    return max_iter;
}
///@endcond

} //namespace dg



#endif //_DG_PCG_
