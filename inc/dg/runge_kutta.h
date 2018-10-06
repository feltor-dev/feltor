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
struct RK
{
    using real_type = get_value_type<ContainerType>;
    ///@brief No memory allocation, Call \c construct before using the object
    RK(){}
    ///@copydoc construct()
    RK( const ContainerType& copyable, std::string tableau): RK( copyable, dg::create::tableau( tableau)){
    }
    RK( const ContainerType& copyable, ButcherTableau tableau):m_rk(tableau){
        m_k.assign(tableau.num_stages(), copyable);
        m_u = copyable;
    }
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param copyable ContainerType of the size that is used in \c step
    * @param tableau Name of Butcher Tableau
    * @note it does not matter what values \c copyable contains, but its size is important
    */
    void construct( const ContainerType& copyable, std::string tableau){
        construct( copyable, dg::create::tableau(tableau));
    }
    void construct( const ContainerType& copyable, ButcherTableau tableau){
        m_rk = tableau;
        m_k.assign(tableau.num_stages(), copyable);
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
    std::array<ContainerType,s> m_k;
    ContainerType m_u;
    ButcherTableau<real_type> m_rk;
};

template< class ContainerType>
template< class RHS>
void RK<ContainerType>::step( RHS& f, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt)
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
struct SIRK
{
    using real_type = get_value_type<ContainerType>;
    ///@copydoc RK_opt::RK_opt()
    SIRK(){}
    ///@copydoc Karniadakis::construct()
    SIRK(const ContainerType& copyable, unsigned max_iter, real_type eps){
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
    RK<ContainerType > rk( begin, tableau);
    if( t_end == t_begin){ end = begin; return;}
    real_type dt = (t_end-t_begin)/(real_type)N;
    end = begin;
    real_type t0 = t_begin;
    for( unsigned i=0; i<N; i++)
        rk.step( rhs, t0, end, t0, end, dt);
}


/**
 * @brief Integrates the differential equation using a stage s Runge-Kutta scheme, a rudimentary stepsize-control and monitoring the sanity of integration
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
 * @attention This function is superseded by the better \c dg::integrateHRK Halfstep function
 */
template< class RHS, class ContainerType>
int integrateRK(ButcherTableau tableau, RHS& rhs, get_value_type<ContainerType> t_begin, const ContainerType& begin, get_value_type<ContainerType> t_end, ContainerType& end, get_value_type<ContainerType> eps_abs, unsigned NT_init = 2 )
{
    using real_type = get_value_type<ContainerType>;
    RK<ContainerType > rk( begin, tableau);
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
