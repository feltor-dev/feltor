/* EVE adds an estimator for the largest Eigenvalue
   to not-yet-preconditioned CG.

                           */

#ifndef _DG_EVE_
#define _DG_EVE_

#include <cmath>
#include "blas.h"
#include "functors.h"

namespace dg
{

//MW: please document
/* EVE (EigenValueEstimator) estimate largest EV using CG */
template< class Vector>
class EVE
{
public:
    using value_type  = get_value_type<Vector>;
    EVE() {}
    EVE( const Vector& copyable, unsigned max_iter):r( copyable), p( r), ap( r), max_iter( max_iter) {}
    void set_max( unsigned new_max)
    {   max_iter = new_max;
    }
    unsigned get_max() const
    {   return max_iter;
    }
    void construct( const Vector& copyable, unsigned max_iterations = 100)
    {   ap = p = r = copyable;
        max_iter = max_iterations;
    }
    template< class Matrix>
    unsigned operator()( Matrix& A, Vector& x, const Vector& b, value_type& ev_max, value_type eps_ev=1e-16);
private:
    Vector r, p, ap;
    unsigned max_iter;
};

template< class Vector>
template< class Matrix>
unsigned EVE< Vector>::operator()( Matrix& A, Vector& x, const Vector&
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
    for( unsigned i=1; i<max_iter; i++)
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
    return max_iter;
};

} //namespace dg
#endif //_DG_EVE_
