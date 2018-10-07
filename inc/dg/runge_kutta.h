#ifndef _DG_RK_
#define _DG_RK_

#include <cassert>
#include <array>

#include "backend/exceptions.h"
#include "tableau.h"
#include "blas1.h"
#include "implicit.h"

/*! @file
 * @brief Runge-Kutta explicit time-integrators
 */

namespace dg{

 /** @class hide_rhs
  * @tparam RHS The right hand side
        is a functor type with no return value (subroutine)
        of signature <tt> void operator()(value_type, const ContainerType&, ContainerType&)</tt>
        The first argument is the time, the second is the input vector, which the functor may \b not override, and the third is the output,
        i.e. y' = f(t, y) translates to f(t, y, y').
        The two ContainerType arguments never alias each other in calls to the functor.
  */

/**
* @brief Struct for Runge-Kutta explicit time-integration, classic formulation
* \f[
 \begin{align}
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j \\
    k_j = f\left( u^n + \Delta t \sum_{l=1}^j a_{jl} k_l\right)
 \end{align}
\f]

@snippet runge_kutta_t.cu function
@snippet runge_kutta_t.cu doxygen
* @ingroup time
*
* Uses only \c dg::blas1::axpby() routine to integrate one step.
* The coefficients are chosen in the classic form given by Runge and Kutta.
* @tparam s Order of the method (1, 2, 3, 4, 6, 17)
* @copydoc hide_ContainerType
*/
template<class ContainerType>
struct RKStep
{
    using real_type = get_value_type<ContainerType>;
    ///@brief No memory allocation, Call \c construct before using the object
    RKStep(){}
    ///@copydoc construct()
    RKStep( const ContainerType& copyable, ConvertsToButcherTableau<real_type> tableau)m_rk(tableau),
            m_k(m_rk.num_stages(), copyable),
            m_u( copyable)
        { }
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param copyable ContainerType of the size that is used in \c step
    * @param tableau Name of Butcher Tableau
    * @note it does not matter what values \c copyable contains, but its size is important
    */
    void construct( const ContainerType& copyable, ConvertsToButcherTableau<real_type> tableau){
        m_rk = tableau;
        m_k.assign(m_rk.num_stages(), copyable);
        m_u = copyable;
    }
    /**
    * @brief Advance one step
    *
    * @copydoc hide_rhs
    * @param rhs right hand side subroutine
    * @param t0 start time
    * @param u0 value at \c t0
    * @param t1 (write only) end time ( equals \c t0+dt on output, may alias \c t0)
    * @param u1 (write only) contains result on output (may alias u0)
    * @param dt timestep
    */
    template<class RHS>
    void step( RHS& rhs, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt);
    ///global order of the method
    size_t order() const {
        return m_rk.order();
    }
  private:
    ButcherTableau<real_type> m_rk;
    std::vector<ContainerType> m_k;
    ContainerType m_u;
};

template< class ContainerType>
template< class RHS>
void RKStep<ContainerType>::step( RHS& f, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt)
{
    f(t0, u0, m_k[0]); //compute k_0
    real_type tu = t0;
    unsigned s = m_rk.num_stages();
    for( unsigned i=1; i<s; i++) //compute k_i
    {
        blas1::axpby( 1., u0, dt*m_rk.a(i,0),m_k[0], m_u); //l=0
        tu = DG_FMA( dt,m_rk.c(i),t0); //l=0
        for( unsigned l=1; l<i; l++)
            blas1::axpby( dt*m_rk.a(i,l), m_k[l],1., m_u);
        f( tu, m_u, m_k[i]);

    }
    //Now add everything up to u1
    blas1::axpby( dt*m_rk.b(0), m_k[0],1.,u0, u1);
    for( unsigned i=1; i<s; i++)
        blas1::axpby( dt*m_rk.b(i), m_k[i],1., u1);
    t1 = t0 + dt;
}

/**
 * @brief Semi implicit Runge Kutta method after Yoh and Zhong (AIAA 42, 2004)
 *
The SIRK algorithm reads
\f[
	\vec v^{n+1} = \vec v^n + \sum_{i=0}^2 w_i \vec k_i \\
	\vec k_i = \Delta t\left[ \vec E\left( \vec v^n + \sum_{j=0}^{i-1} b_{ij}\vec k_j\right)
	+\vec I\left( \vec v^n + \sum_{j=0}^{i-1}c_{ij}\vec k_j + d_i \vec k_i\right) \right]
  \f]
with rational coefficients
\f[
	w_0 = \frac{1}{8} \quad b_{10} = \frac{8}{7} \quad d_0 = \frac{3}{4}  \quad c_{10} = \frac{5589}{6524}  \\
	w_1 = \frac{1}{8} \quad b_{20} = \frac{71}{252} \quad d_1 = \frac{75}{233}  \quad c_{20} = \frac{7691}{26096} \\
	w_2 = \frac{3}{4} \quad b_{21} = \frac{7}{36}   \quad d_2 = \frac{65}{168}  \quad c_{21} = -\frac{26335}{78288}
\f]
We solve the implicit substeps by a conjugate gradient method, which works as long
as the implicit part remains symmetric and linear.

The following code example demonstrates how to implement the method of manufactured solutions on a 2d partial differential equation with the dg library:
@snippet multistep_t.cu function
In the main function:
@snippet multistep_t.cu sirk
@note To our experience the implicit treatment of diffusive or hyperdiffusive
terms can significantly reduce the required number of time steps. This
far outweighs the increased computational cost of the additional matrix inversions.
 * @ingroup time
 * @copydoc hide_ContainerType
 */
template <class ContainerType>
struct SIRKStep
{
    using real_type = get_value_type<ContainerType>;
    ///@copydoc RK::RK()
    SIRKStep(){}
    ///@copydoc Karniadakis::construct()
    SIRKStep(const ContainerType& copyable, unsigned max_iter, real_type eps){
        construct( copyable, max_iter, eps);
    }
    ///@copydoc Karniadakis::construct()
    void construct(const ContainerType& copyable, unsigned max_iter, real_type eps)
    {
        k_.fill( copyable);
        temp_ = rhs_ = f_ = g_ = copyable;
        pcg.construct( copyable, max_iter);
        eps_ = eps;

        w[0] = 1./8., w[1] = 1./8., w[2] = 3./4.;
        b[1][0] = 8./7., b[2][0] = 71./252., b[2][1] = 7./36.;
        d[0] = 3./4., d[1] = 75./233., d[2] = 65./168.;
        c[1][0] = 5589./6524., c[2][0] = 7691./26096., c[2][1] = -26335./78288.;
    }
    /**
     * @brief integrate one step
     *
     * @copydoc hide_explicit_implicit
     * @param t0 start time
     * @param u0 start point at \c t0
     * @param t1 (write only) end time (equals \c t0+dt on output, may alias t0)
     * @param u1 (write only) contains result at \c t1 on output (may alias u0)
     * @param dt timestep
     */
    template <class Explicit, class Implicit>
    void step( Explicit& ex, Implicit& im, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt)
    {
        ex(t0, u0, f_);
        im(t0+d[0]*dt, u0, g_);
        dg::blas1::axpby( dt, f_, dt, g_, rhs_);
        detail::Implicit<Implicit, ContainerType> implicit( -dt*d[0], t0+d[0]*dt, im);
        implicit.alpha() = -dt*d[0];
        implicit.time()  = t0 + (d[0])*dt;
        blas2::symv( im.weights(), rhs_, rhs_);
        pcg( implicit, k_[0], rhs_, im.precond(), im.inv_weights(), eps_);

        dg::blas1::axpby( 1., u0, b[1][0], k_[0], rhs_);
        ex(t0+b[1][0]*dt, rhs_, f_);
        dg::blas1::axpby( 1., u0, c[1][0], k_[0], rhs_);
        im(t0+(c[1][0]+d[1])*dt, rhs_, g_);
        dg::blas1::axpby( dt, f_, dt, g_, rhs_);
        implicit.alpha() = -dt*d[1];
        implicit.time()  =  t0 + (c[1][0]+d[1])*dt;
        blas2::symv( im.weights(), rhs_, rhs_);
        pcg( implicit, k_[1], rhs_, im.precond(), im.inv_weights(), eps_);

        dg::blas1::axpby( 1., u0, b[2][0], k_[0], rhs_);
        dg::blas1::axpby( b[2][1], k_[1], 1., rhs_);
        ex(t0 + (b[2][1]+b[2][0])*dt, rhs_, f_);
        dg::blas1::axpby( 1., u0, c[2][0], k_[0], rhs_);
        dg::blas1::axpby( c[2][1], k_[1], 1., rhs_);
        im(t0 + (c[2][1]+c[2][0] + d[2])*dt, rhs_, g_);
        dg::blas1::axpby( dt, f_, dt, g_, rhs_);
        implicit.alpha() = -dt*d[2];
        implicit.time()  =  t0 + (c[2][1]+c[2][0] + d[2])*dt;
        blas2::symv( im.weights(), rhs_, rhs_);
        pcg( implicit, k_[2], rhs_, im.precond(), im.inv_weights(), eps_);
        //sum up results
        dg::blas1::copy( u0, u1);
        dg::blas1::axpby( 1., u1, w[0], k_[0], u1);
        dg::blas1::axpbypgz( w[1], k_[1], w[2], k_[2], 1., u1);
        t1 = t0 + dt;
    }
    private:
    std::array<ContainerType,3> k_;
    ContainerType f_, g_, rhs_, temp_;
    real_type w[3];
    real_type b[3][3];
    real_type d[3];
    real_type c[3][3];
    CG<ContainerType> pcg;
    real_type eps_;
};

/**
* @brief Struct for embedded Runge Kutta explicit time-step with error estimate
* \f[
 \begin{align}
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j \\
    \tilde u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s \tilde b_j k_j \\
    k_j = f\left( u^n + \Delta t \sum_{l=1}^j a_{jl} k_l\right)
 \end{align}
\f]

The coefficients for the Butcher tableau were taken from https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
The Prince Dormand method is an embedded Runge Kutta method, i.e. computes a solution together with an error estimate. It is effecitve due to its First Same as Last property.

* @ingroup time
*
* @copydoc hide_ContainerType
*/
template< class ContainerType>
struct ERKStep
{
    using real_type = get_value_type<ContainerType>;
    using container = ContainerType;
    ///@copydoc RK::RK()
    ERKStep(){
    }
    ///@copydoc RK::construct()
    ERKStep( const ContainerType& copyable, ConvertsToButcherTableau<real_type> tableau) m_rk(tableau), m_k(m_rk.num_stages(), copyable),
        m_u( copyable)
        { }
    ///@copydoc RK::construct()
    void construct( const ContainerType& copyable, ConvertsToButcherTableau<real_type> tableau){
        m_rk = tableau;
        m_k.assign(m_rk.num_stages(), copyable);
        m_u = copyable;
    }
    ///@copydoc RK::step(RHS&,real_type,const ContainerType&,real_type&,ContainerType&,real_type)
    ///@param delta Contains error estimate on output (must have equal sizeas u0)
    template<class RHS>
    void step( RHS& rhs, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta);
    ///global order of the algorithm
    int order() const {
        return m_rk.order();
    }
    int embedded_order() const {
        return m_rk.embedded_order();
    }
  private:
    ButcherTableau<real_type> m_rk;
    std::vector<ContainerType> m_k;
    ContainerType m_u;
    real_type m_last = 1e300;
};

template< class ContainerType>
template< class RHS>
void ERKStep<ContainerType>::step( RHS& f, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta)
{
    unsigned s = m_rk.num_stages();
    //0 stage: probe fsal
    real_type tu = t0;
    if( m_rk.isFsal() && t0 == m_last){
        blas1::copy( m_k[s-1], m_k[0]);
    }
    else
        f(t0, u0, m_k[0]); //compute k_0
    //1 stage
    tu = DG_FMA( m_rk.c(1),dt, t0);
    blas1::axpby( 1., u0, dt*m_rk.a(1,0), m_k[0], m_u);
    f( tu, m_u, m_k[1]);
    //2 stage
    if( s>2) {
        tu = DG_FMA( m_rk.c(2),dt, t0);
        blas1::subroutine( m_u, dg::equals(), PairSum(), 1., u0,
                            dt*m_rk.a(2,0),m_k[0],
                            dt*m_rk.a(2,1),m_k[1]);
        f( tu, m_u, m_k[2]);
    }
    //3 stage
    if( s> 3){
        tu = DG_FMA( m_rk.c(3),dt, t0);
        blas1::subroutine( m_u, dg::equals(), PairSum(), 1., u0,
                             dt*m_rk.a(3,0),m_k[0],
                             dt*m_rk.a(3,1),m_k[1],
                             dt*m_rk.a(3,2),m_k[2]);
        f( tu, m_u, m_k[3]);
    }
    //4 stage
    if( s>4){
        tu = DG_FMA( m_rk.c(4),dt, t0);
        blas1::subroutine( m_u, dg::equals(), PairSum(), 1.        , u0,
                             dt*m_rk.a(4,0),m_k[0],  dt*m_rk.a(4,1),m_k[1],
                             dt*m_rk.a(4,2),m_k[2],  dt*m_rk.a(4,3),m_k[3]);
        f( tu, m_u, m_k[4]);
    }
    //5 stage
    if( s>5) {
        tu = DG_FMA( m_rk.c(5),dt, t0);
        blas1::subroutine( m_u, dg::equals(), PairSum(), 1., u0,
                 dt*m_rk.a(5,0),m_k[0], dt*m_rk.a(5,1),m_k[1],
                 dt*m_rk.a(5,2),m_k[2], dt*m_rk.a(5,3),m_k[3],
                 dt*m_rk.a(5,4),m_k[4]);
        f( tu, m_u, m_k[5]);
    }
    //6 stage
    if( s>6)
    {
        tu = DG_FMA( m_rk.c(6),dt, t0);
        blas1::subroutine( m_u, dg::equals(), PairSum(), 1., u0,
                           dt*m_rk.a(6,0),m_k[0], dt*m_rk.a(6,1),m_k[2],
                           dt*m_rk.a(6,3),m_k[3], dt*m_rk.a(6,4),m_k[4],
                           dt*m_rk.a(6,5),m_k[5]);
        f( tu, m_u, m_k[6]);
        for ( unsigned i=7; i<s; i++)
        {
            blas1::axpby( 1.,u0, dt*m_rk.a(i,0),m_k[0], m_u); //l=0
            tu = DG_FMA( dt,m_rk.c(i),t0); //l=0
            for( unsigned l=1; l<i; l++)
                blas1::axpby( dt*m_rk.a(i,l), m_k[l],1., m_u);
            f( tu, m_u, m_k[i]);
        }
    }
    //Now add everything up to get solution and error estimate
    m_last = t1 = tu;
    switch( s)
    {
        case 2: blas1::evaluate( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1]); break;
        case 3: blas1::evaluate( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1],
                            dt*m_rk.b(2), dt*m_rk.d(2), m_k[2]); break;
        case 4: blas1::evaluate( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1],
                            dt*m_rk.b(2), dt*m_rk.d(2), m_k[2],
                            dt*m_rk.b(3), dt*m_rk.d(3), m_k[3]); break;
        case 5: blas1::evaluate( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1],
                            dt*m_rk.b(2), dt*m_rk.d(2), m_k[2],
                            dt*m_rk.b(3), dt*m_rk.d(3), m_k[3],
                            dt*m_rk.b(4), dt*m_rk.d(4), m_k[4]); break;
        case 6: blas1::evaluate( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1],
                            dt*m_rk.b(2), dt*m_rk.d(2), m_k[2],
                            dt*m_rk.b(3), dt*m_rk.d(3), m_k[3],
                            dt*m_rk.b(4), dt*m_rk.d(4), m_k[4],
                            dt*m_rk.b(5), dt*m_rk.d(5), m_k[5]); break;
        default: blas1::evaluate( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1],
                            dt*m_rk.b(2), dt*m_rk.d(2), m_k[2],
                            dt*m_rk.b(3), dt*m_rk.d(3), m_k[3],
                            dt*m_rk.b(4), dt*m_rk.d(4), m_k[4],
                            dt*m_rk.b(5), dt*m_rk.d(5), m_k[5],
                            dt*m_rk.b(6), dt*m_rk.d(6), m_k[6]);
            //sum the rest
            for( unsigned i=7; i<s; i++)
            {
                dg::blas1::axpby( dt*m_rk.b(i), m_k[i], 1., u1);
                dg::blas1::axpby( dt*m_rk.d(i), m_k[i], 1., delta);
            }
    }
}
template< class ContainerType>
struct ARKStep
{
    using real_type = get_value_type<ContainerType>;
    using container = ContainerType;
    ///@copydoc RK::RK()
    ARKStep(){ }
    ARKStep( const ContainerType& copyable,
             unsigned max_iter,
             real_type eps_cg,
             ConvertsToButcherTableau<real_type> ex_tableau,
             ConvertsToButcherTableau<real_type> im_tableau):
             m_kE(m_rk.num_stages(), copyable),
        m_u( copyable),
        m_rk(tableau)
        { }
    void construct( const ContainerType& copyable, ConvertsToButcherTableau<real_type> tableau){
    }
    void construct( const ContainerType& copyable,
             unsigned max_iter,
             real_type eps_cg,
             ConvertsToButcherTableau<real_type> ex_tableau,
             ConvertsToButcherTableau<real_type> im_tableau){
        m_rkE = ex_tableau;
        m_rkI = im_tableau;
        assert( m_rkE.num_stages() == m_rkI.num_stages());
        m_k.assign(m_rkE.num_stages(), copyable);
        m_u = copyable;
    }
    template< class Explicit, class Implicit>
    void step( Explicit& ex, Implicit& im, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta);
    private:
    ButcherTableau<real_type> m_rkE, m_rkI;
    std::vector<ContainerType> m_kE, m_kI;
    ContainerType m_u, m_rhs;
    real_type m_eps;
};

template< class ContainerType>
template< class Explicit, class Implicit>
void ARK<ContainerType>::step( Explicit& ex, Implicit& im, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta)
{
    //0 stage
    //a^E_00 = a^I_00 = 0
    ex(t0, u0, m_kE[0]);
    im(t0, u0, m_kI[0]);

    //1 stage
    blas1::subroutine( m_rhs, dg::equals(), PairSum(), 1., u0,
            dt*m_rkE.a(1,0), m_kE[0],
            dt*m_rkI.a(1,0), m_kI[0]);
    tu = DG_FMA( m_rkI.c(1),dt, t0);
    detail::Implicit<Implicit, ContainerType> implicit( -dt*m_rkI.a(1,1), tu, im);
    blas2::symv( im.weights(), m_rhs, m_rhs);
    //how to initialize m_u??
    m_pcg( implicit, m_u, m_rhs, im.precond(), im.inv_weights(), m_eps);
    ex(tu, m_u, m_kE[1]);
    im(tu, m_u, m_kI[1]);

    //2 stage
    blas1::subroutine( m_rhs, dg::equals(), PairSum(), 1., u0,
             dt*m_rkE.a(2,0), m_kE[0],
             dt*m_rkE.a(2,1), m_kE[1],
             dt*m_rkI.a(2,0), m_kI[0],
             dt*m_rkI.a(2,1), m_kI[1]);
    tu = DG_FMA( m_rkI.c(2),dt, t0);
    implicit.alpha() = -dt*m_rkI.a(2,2);
    implicit.time()  = tu;
    blas2::symv( im.weights(), m_rhs, m_rhs);
    m_pcg( implicit, m_u, m_rhs, im.precond(), im.inv_weights(), m_eps);
    ex(tu, m_u, m_kE[2]);
    im(tu, m_u, m_kI[2]);
    //3 stage
    blas1::subroutine( m_rhs, dg::equals(), PairSum(), 1., u0,
             dt*m_rkE.a(3,0), m_kE[0],
             dt*m_rkE.a(3,1), m_kE[1],
             dt*m_rkE.a(3,2), m_kE[2],
             dt*m_rkI.a(3,0), m_kI[0],
             dt*m_rkI.a(3,1), m_kI[1],
             dt*m_rkI.a(3,2), m_kI[2]);
    m_tlast = t1 = tu = t0 + dt;
    implicit.alpha() = -dt*m_rkI.a(3,3);
    implicit.time()  = tu;
    blas2::symv( im.weights(), m_rhs, m_rhs);
    m_pcg( implicit, m_u, m_rhs, im.precond(), im.inv_weights(), m_eps);
    ex(tu, m_u, m_kE[3]);
    im(tu, m_u, m_kI[3]);
    //Now compute result and error estimate
    blas1::evaluate( dg::EmbeddedPairSum(),
            u1, delta,
             1., 0. u0,
            dt*m_rkE.b(0), dt*m_rkE.d(0),m_kE[0],
            dt*m_rkE.b(1), dt*m_rkE.d(1),m_kE[1],
            dt*m_rkE.b(2), dt*m_rkE.d(2),m_kE[2],
            dt*m_rkE.b(3), dt*m_rkE.d(3),m_kE[3],
            dt*m_rkI.b(0), dt*m_rkI.d(0),m_kI[0],
            dt*m_rkI.b(1), dt*m_rkI.d(1),m_kI[1],
            dt*m_rkI.b(2), dt*m_rkI.d(2),m_kI[2],
            dt*m_rkI.b(3), dt*m_rkI.d(3),m_kI[3]);
    //sum the rest
    for( unsigned i=4; i<s; i++)
    {
        dg::blas1::axpbypgz( dt*m_rkE.b(i), m_kE[i],
                             dt*m_rkI.b(i), m_kI[i], 1., u1);
        dg::blas1::axpbypgz( dt*m_rkE.d(i), m_kE[i],
                             dt*m_rkI.d(i), m_kI[i], 1., delta);
    }
}


///@addtogroup time
///@{

/**
 * @brief Integrate differential equation with a stage s Runge-Kutta scheme and a fixed number of steps
 *
 * @copydoc hide_rhs
 * @copydoc hide_ContainerType
 * @param rhs The right-hand-side
 * @param t_begin initial time
 * @param begin initial condition
 * @param t_end final time
 * @param end (write-only) contains solution at \c t_end on output (may alias begin)
 * @param N number of steps
 */
template< class RHS, class ContainerType>
void stepperRK(ButcherTableau tableau, RHS& rhs, get_value_type<ContainerType>  t_begin, const ContainerType& begin, get_value_type<ContainerType> t_end, ContainerType& end, unsigned N )
{
    using real_type = get_value_type<ContainerType>;
    RKStep<ContainerType > rk( begin, tableau);
    if( t_end == t_begin){ end = begin; return;}
    real_type dt = (t_end-t_begin)/(real_type)N;
    end = begin;
    real_type t0 = t_begin;
    for( unsigned i=0; i<N; i++)
        rk.step( rhs, t0, end, t0, end, dt);
}


/**
 * @brief Integrates the differential equation using a Runge-Kutta scheme, a rudimentary stepsize-control and monitoring the sanity of integration
 *
 * Doubles the number of timesteps until the desired accuracy is reached
 *
 * @tparam s Order of the method (1, 2, 3, 4, 6, 17)
 * @copydoc hide_rhs
 * @tparam RHS
 * In addition, there must be the function \c bool \c monitor( const ContainerType& end);
 * available, which is called after every step.
 * Return \c true if everything is ok and \c false if the integrator certainly fails.
 * The other function is the \c real_type \c error( const ContainerType& end0, const ContainerType& end1); which computes the error norm in which the integrator should converge.
 * @copydoc hide_ContainerType
 * @param rhs The right-hand-side
 * @param t_begin initial time
 * @param begin initial condition
 * @param t_end final time
 * @param end (write-only) contains solution on output
 * @param eps_abs desired accuracy in the error function between \c end and \c end_old
 * @param NT_init initial number of steps
 * @return number of iterations if converged, -1 and a warning to \c std::cerr when \c isnan appears, -2 if failed to reach \c eps_abs
 */
template< class RHS, class ContainerType>
int integrateRK(
    const ConvertsToButcherTableau<get_value_type<ContainerType>& tableau,
    RHS& rhs,
    get_value_type<ContainerType> t_begin,
    const ContainerType& begin,
    get_value_type<ContainerType> t_end,
    ContainerType& end,
    get_value_type<ContainerType> eps_abs,
    unsigned NT_init = 2 )
{
    using real_type = get_value_type<ContainerType>;
    RKStep<ContainerType > rk( begin, tableau);
    ContainerType old_end(begin);
    blas1::copy( begin, end );
    if( t_end == t_begin) return 0;
    int NT = NT_init;
    real_type dt = (t_end-t_begin)/(real_type)NT;
    real_type error = 1e10;
    real_type t0 = t_begin;

    while( error > eps_abs && NT < pow( 2, 18) )
    {
        blas1::copy( begin, end );

        int i=0;
        while (i<NT)
        {
            rk.step( rhs, t0, end, t0, end, dt);
            if( !rhs.monitor( end ) )  //sanity check
            {
                #ifdef DG_DEBUG
                    std::cout << "---------Got sanity error -> choosing smaller step size and redo integration" << " NT "<<NT<<" dt "<<dt<< std::endl;
                #endif
                break;
            }
            i++;
        }
        error = rhs.error( end, old_end);
        blas1::copy( end, old_end);
        t0 = t_begin;
        dt /= 2.;
        NT *= 2;
    }
    if( std::isnan( error) )
    {
        std::cerr << "ATTENTION: Runge Kutta failed to converge. Error is NAN! "<<std::endl;
        return -1;
    }
    if( error > eps_abs )
    {
        std::cerr << "ATTENTION: Runge Kutta failed to converge. Error is "<<error<<" with "<<NT<<" steps"<<std::endl;
        return -2;
    }
    return NT;


}

///@}

} //namespace dg

#endif //_DG_RK_
