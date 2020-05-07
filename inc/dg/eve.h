#ifndef _DG_EVE_
#define _DG_EVE_

#include <cmath>
#include "blas.h"
#include "functors.h"

/*! @file
 * EVE adds an estimator for the largest Eigenvalue
 *  to not-yet-preconditioned CG.
 *  @author Eduard Reiter and Matthias Wiesenberger
 */


namespace dg
{

/*! @brief (EigenValueEstimator) estimate largest Eigenvalue using conjugate gradient method
* @copydoc hide_ContainerType
 * @ingroup invert
*/
template< class ContainerType>
class EVE
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    EVE() {}
    ///@copydoc construct()
    EVE( const ContainerType& copyable, unsigned max_iter = 100):r( copyable), p( r), ap( r), m_max_iter( max_iter) {}
    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iter Maximum number of iterations to be used
     */
    void construct( const ContainerType& copyable, unsigned max_iter = 100) {
        ap = p = r = copyable;
        m_max_iter = max_iter;
    }
    /// Set maximum number of iterations
    void set_max( unsigned new_max) {
        m_max_iter = new_max;
    }
    /// Get maximum number of iterations
    unsigned get_max() const {   return m_max_iter; }
    /**
     * @brief Unpreconditioned CG to estimate maximum Eigenvalue
     *
     * @param A A symmetric, positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param ev_max (output) maximum Eigenvalue on output
     * @param eps_ev The desired accuracy of the largest Eigenvalue
     *
     * @return Number of iterations used to achieve desired precision or max_iterations
     * @copydoc hide_matrix
     */
    template< class MatrixType>
    unsigned operator()( MatrixType& A, ContainerType& x, const ContainerType& b, value_type& ev_max, value_type eps_ev=1e-16);
  private:
    ContainerType r, p, ap;
    unsigned m_max_iter;
};

///@cond
template< class ContainerType>
template< class MatrixType>
unsigned EVE< ContainerType>::operator()( MatrixType& A, ContainerType& x, const ContainerType&
b, value_type& ev_max, value_type eps_ev)
{
    blas2::symv( A, x, r);
    blas1::axpby( 1., b, -1., r);
    value_type nrm2r_old = blas1::dot( r,r);
    blas1::copy( r, p);
    value_type nrm2r_new, nrmAp;
    value_type alpha = 1., alpha_inv = 1., delta = 0.;
    value_type evdash, gamma = 0., lambda, omega, beta = 0.;
    value_type ev_est = 0.;
    ev_max = 0.;
    for( unsigned i=1; i<m_max_iter; i++)
    {
        lambda = delta*alpha_inv;       // EVE!
        blas2::symv( A, p, ap);
        nrmAp = blas1::dot( p, ap);
        alpha = nrm2r_old /nrmAp;
        alpha_inv = nrmAp /nrm2r_old;   // EVE!
        lambda += alpha_inv;            // EVE!
        blas1::axpby( alpha, p, 1., x);
        blas1::axpby( -alpha, ap, 1., r);
        nrm2r_new = blas1::dot( r, r);
        delta = nrm2r_new /nrm2r_old;                  // EVE!
        evdash = ev_est -lambda;                       // EVE!
        omega = sqrt( evdash*evdash +4.*beta*gamma);   // EVE!
        gamma = 0.5 *(1. -evdash /omega);              // EVE!
        ev_max += omega*gamma;                         // EVE!
        if( fabs(ev_est-ev_max) < eps_ev*ev_max) {
            return i;
        }
        beta = delta*alpha_inv*alpha_inv;              // EVE!
        blas1::axpby(1., r, delta, p);
        nrm2r_old=nrm2r_new;
        ev_est = ev_max;
    }
    return m_max_iter;
};
///@endcond

} //namespace dg
#endif //_DG_EVE_
