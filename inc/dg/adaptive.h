#pragma once

#include "implicit.h"

namespace dg
{

template <class ContainerType>
get_value_type<ContainerType> l2norm( const ContainerType& x)
{
    return sqrt( dg::blas1::dot( x,x));
}
template<class real_type>
real_type pid_control( real_type dt_old, real_type eps_0, real_type eps_1, real_type eps_2, int embedded_order, int order)
{
    real_type m_k1 = -0.58, m_k2 = 0.21, m_k3 = -0.1;
    real_type factor = pow( eps_0  m_k1/(real_type)order)
                     * pow( eps_1, m_k2/(real_type)order)
                     * pow( eps_2, m_k3/(real_type)order);
    return dt_old*factor;
}
template<class real_type>
real_type pi_control( real_type dt_old, real_type eps_0, real_type eps_1, real_type eps_2, int embedded_order, int order)
{
    real_type m_k1 = -0.8, m_k2 = 0.31;
    real_type factor = pow( eps_0, m_k1/(real_type)order)
                     * pow( eps_1, m_k2/(real_type)order)
    return dt_old*factor;
}
template<class real_type>
real_type i_control( real_type dt_old, real_type eps_0, real_type eps_1, real_type eps_2, int embedded_order, int order)
{
    real_type m_k1 = -1.;
    real_type factor = pow( eps_0, m_k1/(real_type)order);
    return dt_old*factor;
}

template<class real_type>
struct PIDController
{
    PIDController( ){}
    real_type operator()( real_type dt_old, real_type eps_n, real_type eps_n1, real_type eps_n2, int embedded_order, int order)const
    {
        real_type factor = pow( eps_n,  m_k1/(real_type)order)
                         * pow( eps_n1, m_k2/(real_type)order)
                         * pow( eps_n2, m_k3/(real_type)order);
        real_type dt_new = dt_old*std::max( m_lower_limit, std::min( m_upper_limit, factor) );
        return dt_new;
    }
    void set_lower_limit( real_type lower_limit) {
        m_lower_limit = lower_limit;
    }
    void set_upper_limit( real_type upper_limit) {
        m_upper_limit = upper_limit;
    }
    private:
    real_type m_k1 = -0.58, m_k2 = 0.21, m_k3 = -0.1;
    real_type m_lower_limit = 0, m_upper_limit = 1e300;
};

template<class real_type>
struct Tolerance
{
    Tolerance( real_type rtol, real_type atol, real_type size):m_rtol(rtol*sqrt(size)), m_atol( atol*sqrt(size)){}
    DG_DEVICE
    void operator()( real_type previous, real_type& delta) const{
        delta = delta/ ( m_rtol*fabs(previous) + m_atol);
    }
    private:
    real_type m_rtol, m_atol;
};

//%%%%%%%%%%%%%%%%%%%Adaptive%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
template<class Stepper>
class Adaptive
{
    using container = typename Stepper::container;
    using real_type = typename Stepper::real_type;
    template<class ...StepperParams>
    Adapative( const container& copyable, StepperParams&& ...ps): m_stepper(copyable, std::forward<Ts>(xs)...) , m_next(copyable), m_delta(copyable)
    {
        dg::blas1::transform( copyable, m_next, dg::ONE<real_type>());
        m_size = dg::blas1::dot( m_next, 1.);
    }
    template<class Explicit, class ErrorNorm>
    real_type initial_guess( Explicit& ex, real_type t0, container& u0, ErrorNorm& norm, enum direction = dg::forward) const;

    template<class RHS, class ControlFunction, class ErrorNorm>
    int step( RHS& rhs,
              real_type t_begin,
              const container& begin,
              real_type& t_end,
              container& end,
              real_type& dt,
              ControlFunction& control,
              ErrorNorm& norm,
              real_time rtol,
              real_time atol
              )
    {
        real_type t_next;
        m_stepper.step( rhs, t_begin, begin, t_next, m_next, dt, m_delta);
        dg::blas1::subroutine( Tolerance( rtol, atol, m_size), begin, m_delta);
        real_type eps0 = norm(m_delta);
        dt = control( dt, eps0, m_eps1, m_eps2, m_stepper.embedded_order(), m_stepper.order());
        if( eps0 > m_reject_limit)
            m_failed = true;
        else
        {
            std::swap( m_eps2, m_eps1);
            std::swap( eps0, m_eps1);
            dg::blas1::copy( m_next, end);
            t_end = t_next;
            m_failed = false;
        }
        return m_stepper.num_stages();
    }
    template<class Explicit, class Implicit, class ControlFunction, class ErrorNorm>
    int step( Explicit& ex,
              Implicit& im,
              real_type t_begin,
              const container& begin,
              real_type t_end,
              container& end,
              real_type& dt,
              ControlFunction& control,
              ErrorNorm& norm,
              real_time rtol,
              real_time atol
    {
        real_type t_end;
        m_stepper.step( ex, im, t_current, current, t_end, m_next, dt, m_delta);
        dg::blas1::subroutine( Tolerance( rtol, atol, m_size), begin, m_delta);
        real_type eps0 = norm(m_delta);
        dt = control( dt, eps0, m_eps1, m_eps2, m_stepper.embedded_order(), m_stepper.order());
        if( eps0 > m_reject_limit)
            m_failed = true;
        else
        {
            std::swap( m_eps2, m_eps1);
            std::swap( eps0, m_eps1);
            dg::blas1::copy( m_next, end);
            t_end = t_next;
            m_failed = false;
        }
        return m_stepper.num_stages();
    }
    private:
    bool m_failed = false;
    Stepper m_stepper;
    container m_next, m_delta;
    real_type m_reject_limit = 2;
    real_type m_size, m_eps0=1., m_eps1=1., m_eps2=1.;
};
template<class Stepper>
template<class Explicit, class ErrorNorm>
real_type Apaptive<Stepper>::initial_guess( Explicit& ex, real_type t_begin, const ContainerType& begin, ErrorNorm& tol, enum direction)
{
    ex( t_begin, begin, m_next);
    real_type norm = tol(m_next), dt;
    if( forward)
        dt = 1./norm;
    else
        dt = -1./norm;
    return dt;
}

///@addtogroup time
///@{

/**
 * @brief Integrates a differential equation using a one-step explicit Timestepper, with adaptive stepsize-control and monitoring the sanity of integration
 *
 The adaptivity is given by: \c dt_current*=0.9*pow(desired_accuracy/error, 1./(real_type)stepper.order()) ;
 If the error lies above the desired accuracy, a step is rejected and subsequently recomputed.
 * @tparam Stepper must have the \c step member function and an \c order member function that returns the global order of the Stepper
 * @copydoc hide_rhs
 * @copydoc hide_ContainerType
 * @tparam Monitor
 * Must have a member function \c real_type \c norm( const ContainerType& ); which computes the norm in which the integrator should converge. The DefaultMonitor is usually sufficient.
 * @param rhs The right-hand-side
 * @param t_begin initial time
 * @param begin initial condition
 * @param t_end final time. The integrator will truncate the last stepsize such that the final time is exactly \c t_end
 * @param end (write-only) contains solution on output
 * @param dt on input: initial guess for the timestep (can be 0, then the algorithm will come up with an initial guess on its own)
 * on output: contains last (untruncated) stepsize that can be used to continue the integration
 * @param eps_rel the desired accuracy is given by
         \c eps_rel*norm+eps_abs, where \c norm is the norm of the current step
 * @param eps_abs the desired accuracy is given by
         \c eps_rel*norm+eps_abs, where \c norm is the norm of the current step
 * @param epus error per unit step ( if \c true, the desired accuracy is multiplied with \c dt to make the global error small, instead of only the local one)
 * @param monitor instance of the \c Monitor class
 * @return number of steps
 */
template<class Stepper, class RHS, class ContainerType>
int integrateAdaptive(Stepper& stepper,
                      RHS& rhs,
                      get_value_type<ContainerType> t_begin,
                      const ContainerType& begin,
                      get_value_type<ContainerType> t_end,
                      ContainerType& end,
                      get_value_type<ContainerType>& dt,
                      get_value_type<ContainerType> eps_rel,
                      get_value_type<ContainerType> eps_abs=1e-10,
                      )
{
    using  real_type = get_value_type<ContainerType>;
    real_type t_current = t_begin, dt_current = dt, t_next;
    ContainerType next(begin), delta(begin);
    blas1::copy( begin, end );
    ContainerType& current(end);
    if( t_end == t_begin)
        return 0;
    bool forward = (t_end - t_begin > 0);

    //1. Find initial test step
    if( dt == 0)
    {
        real_type desired_accuracy = eps_rel*monitor.norm(current) + eps_abs;
        rhs( t_current, current, next);
        dt_current = std::min( fabs( t_end - t_begin), pow(desired_accuracy, 1./(real_type)stepper.order())/monitor.norm(next));
        if( !forward) dt_current*=-1.;
        //std::cout << t_current << " "<<dt_current<<"\n";
    }
    int counter =0;
    while( (forward && t_current < t_end) || (!forward && t_current > t_end))
    {
        dt = dt_current;
        if( (forward && t_current+dt_current > t_end) || (!forward && t_current + dt_current < t_end) )
            dt_current = t_end-t_current;
        // Compute a step and error
        stepper.step( rhs, t_current, current, t_next, next, dt_current, delta);
        counter++;
        real_type norm = monitor.norm( next);
        real_type error = monitor.norm( delta);
        real_type desired_accuracy = eps_rel*norm + eps_abs;
        //std::cout << eps_abs << " " <<desired_accuracy<<std::endl;
        if( epus)
            desired_accuracy*=fabs(dt_current);
        dt_current *= std::max( 0.1, std::min( 10., 0.9*pow(desired_accuracy/error, 1./(real_type)stepper.order()) ) );  //DON'T DO THIS; MAKES dt FUNCTION NONLINEAR
        //std::cout << t_current << " "<<t_next<<" "<<dt_current<<" acc "<<error<<" "<<desired_accuracy<<"\n";
        if( error>desired_accuracy)
            continue;
        else
        {
            dg::blas1::copy( next, current);
            t_current = t_next;
        }
    }
    return counter;
}

///Shortcut for \c dg::integrateAdaptive with the \c dg::PrinceDormand class as timestepper
///@snippet adaptive_t.cu function
///@snippet adaptive_t.cu doxygen
template< class RHS, class ContainerType, class Monitor = DefaultMonitor>
int integrateRK45( RHS& rhs,
                   get_value_type<ContainerType> t_begin,
                   const ContainerType& begin,
                   get_value_type<ContainerType> t_end,
                   ContainerType& end,
                   get_value_type<ContainerType>& dt,
                   get_value_type<ContainerType> eps_rel,
                   get_value_type<ContainerType> eps_abs=1e-10,
                   bool epus=false,
                   Monitor monitor=Monitor() )
{
    dg::PrinceDormand<ContainerType> pd( begin);
    return integrateAdaptive( pd, rhs, t_begin, begin, t_end, end, dt, eps_rel, eps_abs, epus, monitor);
}
///Shortcut for \c dg::integrateAdaptive with a half-step Runge Kutta (\c dg::RK) of stage \c s scheme as timestepper (recompute each step with half the stepsize to get the error estimate)
template< size_t s, class RHS, class ContainerType, class Monitor = DefaultMonitor>
int integrateHRK( RHS& rhs,
                  get_value_type<ContainerType> t_begin,
                  const ContainerType& begin,
                  get_value_type<ContainerType> t_end,
                  ContainerType& end,
                  get_value_type<ContainerType>& dt,
                  get_value_type<ContainerType> eps_rel,
                  get_value_type<ContainerType> eps_abs=1e-10,
                  bool epus=false,
                  Monitor monitor=Monitor() )
{
    dg::HalfStep<dg::RK<s, ContainerType>> rk( begin);
    return integrateAdaptive( rk, rhs, t_begin, begin, t_end, end, dt, eps_rel, eps_abs, epus, monitor);
}
///Shortcut for \c dg::integrateAdaptive with a half-step Runge Kutta (\c dg::RK) of stage \c s scheme as timestepper (recompute each step with half the stepsize to get the error estimate)
template< class Explicit, class Implicit, class ContainerType, class Monitor = DefaultMonitor>
int integrateHSIRK( Explicit& exp, Implicit& imp,
                  get_value_type<ContainerType> t_begin,
                  const ContainerType& begin,
                  get_value_type<ContainerType> t_end,
                  ContainerType& end,
                  get_value_type<ContainerType>& dt,
                  get_value_type<ContainerType> eps_rel,
                  get_value_type<ContainerType> eps_abs=1e-10,
                  bool epus=false,
                  Monitor monitor=Monitor() )
{
    dg::ExImHalfStep<dg::SIRK<ContainerType>> sirk( begin);
    return integrateAdaptive( sirk, std::make_pair(std::ref(exp),std::ref(imp)), t_begin, begin, t_end, end, dt, eps_rel, eps_abs, epus, monitor);
}
///@}
}//namespace dg
