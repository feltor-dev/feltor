#pragma once

#include "implicit.h"
#include "runge_kutta.h"

namespace dg
{

template <class ContainerType>
get_value_type<ContainerType> l2norm( const ContainerType& x)
{
    return sqrt( dg::blas1::dot( x,x));
}
template<class real_type>
real_type pid_control( real_type dt_old, real_type eps_0, real_type eps_1, real_type eps_2, unsigned embedded_order, unsigned order)
{
    real_type m_k1 = -0.58, m_k2 = 0.21, m_k3 = -0.1;
    real_type factor = pow( eps_0, m_k1/(real_type)order)
                     * pow( eps_1, m_k2/(real_type)order)
                     * pow( eps_2, m_k3/(real_type)order);
    return dt_old*factor;
}
template<class real_type>
real_type pi_control( real_type dt_old, real_type eps_0, real_type eps_1, real_type eps_2, unsigned embedded_order, unsigned order)
{
    real_type m_k1 = -0.8, m_k2 = 0.31;
    real_type factor = pow( eps_0, m_k1/(real_type)order)
                     * pow( eps_1, m_k2/(real_type)order);
    return dt_old*factor;
}
template<class real_type>
real_type i_control( real_type dt_old, real_type eps_0, real_type eps_1, real_type eps_2, unsigned embedded_order, unsigned order)
{
    real_type m_k1 = -1.;
    real_type factor = pow( eps_0, m_k1/(real_type)order);
    return dt_old*factor;
}

template<class real_type>
struct PIDController
{
    PIDController( ){}
    real_type operator()( real_type dt_old, real_type eps_n, real_type eps_n1, real_type eps_n2, unsigned embedded_order, unsigned order)const
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
namespace detail{
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
}

//%%%%%%%%%%%%%%%%%%%Adaptive%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
template<class Stepper>
struct Adaptive
{
    using container = typename Stepper::container;
    using real_type = typename Stepper::real_type;
    template<class ...StepperParams>
    Adaptive( const container& copyable, StepperParams&& ...ps): m_stepper(copyable, std::forward<StepperParams>(ps)...) , m_next(copyable), m_delta(copyable)
    {
        dg::blas1::copy( 1., m_next);
        m_size = dg::blas1::dot( m_next, 1.);
    }
    template<class Explicit, class ErrorNorm = real_type(const container&)>
    real_type guess_stepsize( Explicit& ex, real_type t0, const container& u0, enum direction dir, ErrorNorm& norm, real_type rtol, real_type atol);

    template< class RHS,
              class ControlFunction = real_type (real_type, real_type, real_type, real_type, unsigned, unsigned),
              class ErrorNorm = real_type( const container&)>
    void step( RHS& rhs,
              real_type t_begin,
              const container& begin,
              real_type& t_end,
              container& end,
              real_type& dt,
              ControlFunction& control,
              ErrorNorm& norm,
              real_type rtol,
              real_type atol
              )
    {
        m_stepper.step( rhs, t_begin, begin, m_t_next, m_next, dt, m_delta);
        return update( t_begin, begin, t_end, end, dt, control, norm , rtol, atol);
    }
    template< class Explicit,
              class Implicit,
              class ControlFunction = real_type (real_type, real_type, real_type, real_type, unsigned, unsigned),
              class ErrorNorm = real_type( const container&)>
    void step( Explicit& ex,
              Implicit& im,
              real_type t_begin,
              const container& begin,
              real_type& t_end,
              container& end,
              real_type& dt,
              ControlFunction& control,
              ErrorNorm& norm,
              real_type rtol,
              real_type atol)
    {
        m_stepper.step( ex, im, t_begin, begin, m_t_next, m_next, dt, m_delta);
        return update( t_begin, begin, t_end, end, dt, control, norm , rtol, atol);
    }
    bool hasFailed() const {
        return m_failed;
    }
    private:
    template<   class ControlFunction = real_type (real_type, real_type, real_type, real_type, unsigned, unsigned),
                class ErrorNorm = real_type( const container&)>
    void update( real_type t_begin,
                const container& begin,
                real_type& t_end,
                container& end,
                real_type& dt,
                ControlFunction& control,
                ErrorNorm& norm,
                real_type rtol,
                real_type atol
              )
    {
        //std::cout << "Try stepsize "<<dt;
        dg::blas1::evaluate( detail::Tolerance<real_type>( rtol, atol, m_size), begin, m_delta);
        real_type eps0 = norm(m_delta);
        //std::cout << " error "<<eps0;
        dt = control( dt, eps0, m_eps1, m_eps2, m_stepper.embedded_order(), m_stepper.order());
        //std::cout << " new stepsize "<<dt;
        if( eps0 > m_reject_limit || std::isnan( eps0) )
        {
            m_failed = true;
            dg::blas1::copy( begin, end);
            t_end = t_begin;
            //std::cout << " Failed ";
        }
        else
        {
            m_eps2 = m_eps1;
            m_eps1 = eps0;
            dg::blas1::copy( m_next, end);
            t_end = m_t_next;
            m_failed = false;
            //std::cout << " Success " << t_end<<" "<<end[0]<<" "<<end[1];
        }
        //std::cout << std::endl;
    }
    bool m_failed = false;
    Stepper m_stepper;
    container m_next, m_delta;
    real_type m_reject_limit = 2;
    real_type m_size, m_eps1=1, m_eps2=1;
    real_type m_t_next = 0;
};
template<class Stepper>
template<class Explicit, class ErrorNorm>
typename Adaptive<Stepper>::real_type Adaptive<Stepper>::guess_stepsize( Explicit& ex, real_type t_begin, const container& begin, enum direction dir, ErrorNorm& tol, real_type rtol, real_type atol)
{
    real_type desired_accuracy = rtol*tol(begin) + atol;
    ex( t_begin, begin, m_next);
    real_type dt = pow(desired_accuracy, 1./(real_type)m_stepper.order())/tol(m_next);
    if( dir != forward)
        dt*=-1.;
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
template<class Adaptive, class RHS, class ContainerType, class ErrorNorm = get_value_type<ContainerType>( const ContainerType&),
             class ControlFunction = get_value_type<ContainerType> (get_value_type<ContainerType>, get_value_type<ContainerType>, get_value_type<ContainerType>, get_value_type<ContainerType>, unsigned, unsigned)>
int integrateAdaptive(Adaptive& adaptive,
                      RHS& rhs,
                      get_value_type<ContainerType> t_begin,
                      const ContainerType& begin,
                      get_value_type<ContainerType> t_end,
                      ContainerType& end,
                      get_value_type<ContainerType>& dt,
                      ControlFunction control,
                      ErrorNorm norm,
                      get_value_type<ContainerType> rtol,
                      get_value_type<ContainerType> atol=1e-10
                      )
{
    using  real_type = get_value_type<ContainerType>;
    real_type t_current = t_begin, dt_current = dt;
    ContainerType next(begin), delta(begin);
    blas1::copy( begin, end );
    ContainerType& current(end);
    if( t_end == t_begin)
        return 0;
    bool forward = (t_end - t_begin > 0);
    if( dt == 0)
        dt_current = adaptive.guess_stepsize( rhs, t_begin, begin, forward ? dg::forward:dg::backward, norm, rtol, atol);

    int counter =0;
    while( (forward && t_current < t_end) || (!forward && t_current > t_end))
    {
        dt = dt_current;
        if( (forward && t_current+dt_current > t_end) || (!forward && t_current + dt_current < t_end) )
            dt_current = t_end-t_current;
        // Compute a step and error
        adaptive.step( rhs, t_current, current, t_current, current, dt_current, control, norm, rtol, atol);
        counter++;
    }
    return counter;
}

///Shortcut for \c dg::integrateAdaptive with an embedded ERK class as timestepper
///@snippet adaptive_t.cu function
///@snippet adaptive_t.cu doxygen
template<class RHS, class ContainerType, class ErrorNorm = get_value_type<ContainerType>( const ContainerType&),
             class ControlFunction = get_value_type<ContainerType> (get_value_type<ContainerType>, get_value_type<ContainerType>, get_value_type<ContainerType>, get_value_type<ContainerType>, unsigned, unsigned)>
int integrateERK( std::string name,
                  RHS& rhs,
                  get_value_type<ContainerType> t_begin,
                  const ContainerType& begin,
                  get_value_type<ContainerType> t_end,
                  ContainerType& end,
                  get_value_type<ContainerType>& dt,
                  ControlFunction control,
                  ErrorNorm norm,
                  get_value_type<ContainerType> rtol,
                  get_value_type<ContainerType> atol=1e-10
              )
{
    dg::Adaptive<dg::ERKStep<ContainerType>> pd( begin, name);
    return integrateAdaptive( pd, rhs, t_begin, begin, t_end, end, dt, control, norm, rtol, atol);
}
///@}
}//namespace dg
