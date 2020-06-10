#ifndef _DG_CHEB_
#define _DG_CHEB_

#include <cmath>

#include "blas.h"

/*!@file
 * Chebyshev solver
 */

namespace dg
{

/**
* @brief Three-term recursion of the Chebyshev iteration for solving
* \f[ Ax=b\f]
*
* Given the minimum and maximum Eigenvalue of the matrix A we define
* \f[ \theta = (\lambda_\min+\lambda_\max)/2 \quad \delta = (\lambda_\max - \lambda_\min)/2 \\
*     \rho_0 := \frac{\delta}{\theta},\ x_0 := x, \ x_{1} = x_0+\frac{1}{\theta} (b-Ax_0) \\
*     \rho_{k}:=\left(\frac{2\theta}{\delta}-\rho_{k-1}\right)^{-1} \\
*     x_{k+1} := x_k + \rho_k\left( \rho_{k-1}(x_k - x_{k-1})
*     + \frac{2}{\delta} ( b - Ax_k) \right)
* \f]
* For more information see the book "Iteratvie Methods for Sparse
* Linear Systems" 2nd edition by Yousef Saad
*
* @attention Chebyshev iteration may diverge if the elliptical bound of the Eigenvaleus is not accurate or if an ellipsis is not a good fit for the spectrum of the matrix
*
* @ingroup invert
*
* @copydoc hide_ContainerType
*/
template< class ContainerType>
class Chebyshev
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    Chebyshev(){}
    ///@copydoc construct()
    Chebyshev( const ContainerType& copyable):
        m_ax(copyable), m_xm1(m_ax){}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_ax;}

    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     */
    void construct( const ContainerType& copyable) {
        m_xm1 = m_ax = copyable;
    }
    /**
     * @brief Solve the system A*x = b using Chebyshev iteration
     *
     * The iteration stops when the maximum number of iterations is reached
     * @param A A symmetric, positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param min_ev the minimum Eigenvalue
     * @param max_ev the maximum Eigenvalue (must be larger than \c min_ev)
     * @param num_iter the number of iterations k (equals the number of times A is applied)
     *
     * @copydoc hide_matrix
     * @tparam ContainerTypes must be usable with \c MatrixType and \c ContainerType in \ref dispatch
     */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    void solve( MatrixType& A, ContainerType0& x, const ContainerType1& b,
        double min_ev, double max_ev, unsigned num_iter)
    {
        if( num_iter == 0)
            return;
        assert ( min_ev < max_ev);
        double theta = (min_ev+max_ev)/2., delta = (max_ev-min_ev)/2.;
        double rhokm1 = delta/theta, rhok=0;
        dg::blas1::copy( x, m_xm1); //x0
        dg::blas2::symv( A, x, m_ax);
        dg::blas1::axpbypgz( 1./theta, b, -1./theta, m_ax, 1., x); //x1
        for ( unsigned k=1; k<num_iter; k++)
        {
            rhok = 1./(2.*theta/delta - rhokm1);
            dg::blas2::symv( A, x, m_ax);
            dg::blas1::evaluate( m_xm1, dg::equals(), PairSum(),
                             1.+rhok*rhokm1, x,
                            -rhok*rhokm1,    m_xm1,
                             2.*rhok/delta,  b,
                            -2.*rhok/delta,  m_ax
                            );
            x.swap(m_xm1);
            rhokm1 = rhok;
        }
    }
  private:
    ContainerType m_ax, m_xm1;
};

} //namespace dg

#endif // _DG_CHEB_
