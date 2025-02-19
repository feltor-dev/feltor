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

// TODO check for better stopping criteria using condition number estimates?
/**
 * @brief Determine algorithm for complex matrices
 *
 * For a complex matrix it makes a difference if it is symmetric or Hermitian.
 * The difference is that for a complex symmetric matrix we use a product
 * \c dg::blas2::dot (that retuns a complex value) while for a Hermitian matrix we
 * can use the actual complex scalar product \c dg::blas1::vdot (that returns a
 * real value). In the literature the former is equivalent to the COCG method,
 * while the latter is the normal complex CG method.
 * @ingroup invert
 */
enum ComplexMode{
    complex_symmetric, //!< choose COCG algorithm
    complex_hermitian //!< choose Hermitian algorithm
};

// TODO test complex behaviour

/**
* @brief Preconditioned conjugate gradient method to solve \f$ Ax=b\f$
*
* where \f$ A\f$ is positive definite and self-adjoint in the weighted scalar
* product (defined by the diagonal weights matrix \f$W\f$)
* \f[ A^\dagger := \frac{1}{W} A^T W = A\f].
* Note that if \f$ A\f$ is self-adjoint then both \f$ (WA)^T = WA \f$ and \f$
* \left(A \frac{1}{W}\right)^T = A\frac{1}{W}\f$ are symmetric. The positive
* definite, self-adjoint preconditioner \f$ P \approx A^{-1}\f$ that
* approximates the inverse of \f$ A\f$ and is fast to apply, is used to solve
* the left preconditioned system \f[ PAx=Pb\f]
*
* @note For complex matrices the PCG algorithm works and is well-defined by
* replacing the transpose with the Hermitian transpose in the above
* definitions. If you have a complex matrix that is only symmetric and not
* Hermitian then the \c dg::complex_symmetric \c dg::ComplexMode may converge.
* This is equivalent to the COCG algorithm and we have an abbreviation \c dg::COCG
*
* @note Our implementation uses a stopping criterion based on the residual at
* iteration i \f$ || r_i ||_W = ||Ax_i-b||_W < \epsilon( ||b||_W + C) \f$.
* However, the real error is bound by \f[ \frac{ ||e_i||_W}{||x||_W} \leq
* \kappa(PA) \frac{||r_i||_W}{||b||_W} \f] Thus, if the condition number \f$
* \kappa\f$ is large the real error may still be large even if the residual
* error is small see <a href="https://doi.org/10.1023/A:1021961616621">Ashby
* et al., The Role of the Inner Product in Stopping Criteria for Conjugate
* Gradient Iterations (2001)</a>
*
* @ingroup invert
*
* @sa This implements the PCG algorithm (applied to \f$(WA)\f$ as given in
* https://en.wikipedia.org/wiki/Conjugate_gradient_method or the book
* <a href="https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf">"Iterative Methods for Sparse Linear Systems" 2nd edition by Yousef Saad</a>
* @note Conjugate gradients might become unstable for positive semidefinite
* matrices arising e.g. in the discretization of the periodic laplacian
* @attention beware the sign: a negative definite matrix does @b not work in
* Conjugate gradient
*
* @snippet cg2d_t.cpp doxygen
* @copydoc hide_ContainerType
* @tparam ComplexMode For complex value type you can choose between \c
* dg::complex_symmetric or \c dg::complex_hermitian matrices. In the former
* case the algorithm is the COCG algorithm in the latter the normal CG
* algorithm.
*/
template<class ContainerType, ComplexMode complex_mode = dg::complex_hermitian>
class PCG
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class

    ///@brief Allocate nothing, Call \c construct method before usage
    PCG() = default;
    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    PCG( const ContainerType& copyable, unsigned max_iterations):
        m_r(copyable), m_p(m_r), m_ap(m_r), m_max_iter(max_iterations){}
    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {m_max_iter = new_max;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return m_max_iter;}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_r;}

    ///@brief Set or unset debugging output during iterations
    ///@param verbose If true, additional output will be written to \c std::cout during solution
    void set_verbose( bool verbose){ m_verbose = verbose;}
    ///@brief Set or unset a throw on failure-to-converge
    ///@param throw_on_fail If true, the solve method will thow a dg::Fail if it is unable to converge
    ///@note the default value is true
    void set_throw_on_fail( bool throw_on_fail){
        m_throw_on_fail = throw_on_fail;
    }

    ///@copydoc hide_construct
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
     * the absolute error in units of \f$ \epsilon\f$ and \f$ W \f$ defines a square norm
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
     * @note The method will throw \c dg::Fail if the desired accuracy is not reached within \c max_iterations
     * You can unset this behaviour with the \c set_throw_on_fail member
     * @note Required memops per iteration (\c P is assumed vector):
             - 15  reads + 4 writes
             - plus the number of memops for \c A;
     * @copydoc hide_matrix
     * @copydoc hide_ContainerType
     */
    template< class MatrixType0, class ContainerType0, class ContainerType1, class MatrixType1, class ContainerType2 >
    unsigned solve( MatrixType0&& A, ContainerType0& x, const ContainerType1& b, MatrixType1&& P, const ContainerType2& W, double eps = 1e-12, double nrmb_correction = 1, int test_frequency = 1);
  private:
    struct NORM
    {
        template<class T, class Z>
        DG_DEVICE
        auto operator()( T w, Z z) { return w*norm(z);} // returns floating point
    };
    struct DOT
    {
        template<class Z0, class T, class Z1>
        DG_DEVICE
        auto operator()( Z0 z0, T w, Z1 z1) { // returns floating point
            return w*(z0.real()*z1.real() + z0.imag()*z1.imag());
        }
    };
    template<class ContainerType1, class ContainerType2>
    auto norm( const ContainerType1& w, const ContainerType2& x)
    {
        using value_type_x = dg::get_value_type<ContainerType2>;
        constexpr bool is_complex = dg::is_scalar_v<value_type_x, dg::ComplexTag>;
        if constexpr (is_complex)
            return sqrt( blas1::vdot( NORM(), w, x));
        else
            return sqrt( blas2::dot( w, x));
    }
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    auto dot( const ContainerType0& x0, const ContainerType1& w, const ContainerType2& x1)
    {
        using value_type_x1 = dg::get_value_type<ContainerType2>;
        constexpr bool is_complex = dg::is_scalar_v<value_type_x1, dg::ComplexTag>;
        if constexpr (not is_complex or complex_mode == dg::complex_symmetric)
            return blas2::dot( x0, w, x1); // this returns value_type
        else // For Hermitian matrices all dot products in CG are real
            return blas1::vdot( DOT(), x0, w, x1); // this returns floating point
    }
    ContainerType m_r, m_p, m_ap;
    unsigned m_max_iter;
    bool m_verbose = false, m_throw_on_fail = true, m_cocg = false;
};

/*!
 * @brief The Conjugate orthogonal conjugate gradient (COCG) algorithm
 * @ingroup invert
 */
template<class ContainerType>
using COCG = dg::PCG<ContainerType, dg::complex_symmetric>;

///@cond
template< class ContainerType, ComplexMode mode>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner, class ContainerType2>
unsigned PCG< ContainerType, mode>::solve( Matrix&& A, ContainerType0& x, const ContainerType1& b, Preconditioner&& P, const ContainerType2& W, double eps, double nrmb_correction, int save_on_dots )
{
    // self-adjoint: apply PCG algorithm to (P 1/W) (W A) x = (P 1/W) (W b) : P' A' x = P' b'
    // This effectively just replaces all scalar products with the weighted one
    auto nrmb = norm( W, b), nrmr(nrmb);
    double tol = eps*(nrmb + nrmb_correction);
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
    if( m_verbose)
    {
        DG_RANK0 std::cout << "# Norm of W b "<<nrmb <<"\n";
        DG_RANK0 std::cout << "# Residual errors: \n";
    }
    if( nrmb == 0)
    {
        blas1::copy( value_type(0), x);
        return 0;
    }
    blas2::symv( std::forward<Matrix>(A), x, m_r);
    blas1::axpby( 1., b, -1., m_r);
    nrmr = norm( W, m_r);

    if( nrmr < tol) //if x happens to be the solution
        return 0;
    blas2::symv( std::forward<Preconditioner>(P), m_r, m_p );
    auto nrmzr_old = dot( m_p, W, m_r), alpha(nrmzr_old), nrmzr_new(nrmzr_old);
    for( unsigned i=1; i<m_max_iter; i++)
    {
        blas2::symv( std::forward<Matrix>(A), m_p, m_ap);
        alpha =  nrmzr_old/dot( m_p, W, m_ap);
        blas1::axpby( alpha, m_p, 1., x);
        blas1::axpby( -alpha, m_ap, 1., m_r);
        if( 0 == i%save_on_dots )
        {
            nrmr = norm( W, m_r);
            if( m_verbose)
            {
                DG_RANK0 std::cout << "# Absolute r*W*r "<<nrmr <<"\t ";
                DG_RANK0 std::cout << "#  < Critical "<<tol <<"\t ";
                DG_RANK0 std::cout << "# (Relative "<<nrmr/nrmb << ")\n";
            }
            if( sqrt( nrmr) < tol)
                return i;
        }
        blas2::symv(std::forward<Preconditioner>(P),m_r,m_ap);
        nrmzr_new = dot( m_ap, W, m_r);
        blas1::axpby(1., m_ap, nrmzr_new/nrmzr_old, m_p );
        nrmzr_old=nrmzr_new;
    }
    if( m_throw_on_fail)
    {
        throw dg::Fail( tol, Message(_ping_)
            <<"After "<<m_max_iter<<" PCG iterations with rtol "<<eps<<" and atol "<<eps*nrmb_correction );
    }
    return m_max_iter;
}
///@endcond

} //namespace dg



#endif //_DG_PCG_
