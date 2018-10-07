#pragma once

#include "implicit.h"

namespace dg
{

//template<class Stepper>
//class Adaptive
//{
//    using container = typename Stepper::container;
//    using real_type = typename Stepper::real_type;
//    template<class ...Ts>
//    Adapative( const container& copyable, Ts&& ...xs): m_stepper(copyable, std::forward<Ts>(xs)...) , m_next(copyable), m_delta(copyable){}
//
//    template<class RHS, class Tolerance, class Controller>
//    int step( RHS& rhs, 
//                  real_type t_begin,
//                  const container& begin,
//                  real_type t_end,
//                  container& end,
//                  real_type& dt,
//                  Tolerance& tol,
//                  Controller& control)
//    {
//        real_type t_end;
//        m_stepper.step( rhs, t_current, current, t_end, m_next, dt, m_delta);
//        update( )
//    }
//    template<class Explicit, class Implicit, class Tolerance, class Controller>
//    int step( Explicit& ex, Implicit& im,
//                  real_type t_begin,
//                  const container& begin,
//                  real_type t_end,
//                  container& end,
//                  real_type& dt,
//                  Tolerance& tol,
//                  Controller& control)
//    {
//        real_type t_end;
//        m_stepper.step( ex, im, t_current, current, t_end, m_next, dt, m_delta);
//        update( )
//    }
//    private:
//    void update()
//    bool m_failed = false;
//    Stepper m_stepper;
//    container m_next, m_delta;
//    real_type m_reject_limit = 2;
//};
//template<class Stepper>
//template<class Explicit>
//void Apaptive<Stepper>::initial_guess( Explicit& ex, real_type t_begin, const ContainerType& begin, real_type& dt, Tolerance& tol, bool forward)
//{
//    ex( t_begin, begin, m_next);
//    real_type norm = tol(m_next);
//    if( forward)
//        dt = 1./norm;
//    else
//        dt = -1./norm;
//}
//
//template<class Stepper>
//template<class RHS, class Tolerance, class Controller>
//void Adaptive<Stepper>::step( RHS& rhs, 
//                  get_value_type<ContainerType> t_current,
//                  const ContainerType& current,
//                  get_value_type<ContainerType>& t_next,
//                  ContainerType& next,
//                  get_value_type<ContainerType>& dt,
//                  Tolerance& tol,
//                  Controller& control)
//{
//    real_type t_end;
//    real_type norm = tol( current, delta);
//    if( norm > m_reject_limit)
//        m_failed = true;
//    else
//    {
//        std::swap( m_norm2, m_norm1);
//        std::swap( norm, m_norm1);
//        dg::blas1::copy( m_next, next);
//        t_next = t_end;
//        m_failed = false;
//    }
//    dt = control( dt, norm, m_norm1, m_norm2, m_stepper.order());
//    return;
//}

template<class real_type>
struct DefaultTolerance
{
    DefaultTolerance( real_type rtol, real_type atol):m_rtol(rtol), m_atol( atol){}
    template<class ContainerType>
    real_type operator()( const ContainerType& current, const ContainerType& delta) const{
        real_type tol = m_rtol*sqrt(dg::blas1::dot( current, current))+m_atol;
        real_type err = sqrt( dg::blas1::dot( delta, delta));
        return err/tol;
    }
    private:
    real_type m_rtol, m_atol;
};
template<class real_type>
struct PIController
{
    real_type operator()( real_type dt_old, real_type eps_n, real_type eps_n1, real_type eps_n2, int order)const
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



///@cond
template<class Stepper>
struct HalfStep
{
    HalfStep(){}
    HalfStep( const Stepper& copyable): m_stepper(copyable){}
    template <class Explicit, class ContainerType>
    void step( Explicit& exp,
               get_value_type<ContainerType> t0,
               const ContainerType& u0,
               get_value_type<ContainerType>&  t1,
               ContainerType& u1,
               get_value_type<ContainerType> dt,
               ContainerType& delta)
    {
        m_stepper.step( exp, t0, u0, t1, delta, dt); //one full step
        m_stepper.step( exp, t0, u0, t1, u1, dt/2.);
        m_stepper.step( exp, t1, u1, t1, u1, dt/2.);
        dg::blas1::axpby( 1., u1, -1., delta);
        t1 = t0 + dt;
    }
    int order() const{
        return m_stepper.order();
    }
    private:
    Stepper m_stepper;
};
template<class ExImStepper>
struct ExImHalfStep
{
    ExImHalfStep(){}
    ExImHalfStep( const ExImStepper& copyable): m_stepper(copyable){}
    template <class Explicit, class Implicit, class ContainerType>
    void step( Explicit& ex, Implicit& im,
               get_value_type<ContainerType> t0,
               const ContainerType& u0,
               get_value_type<ContainerType>&  t1,
               ContainerType& u1,
               get_value_type<ContainerType> dt,
               ContainerType& delta)
    {
        m_stepper.step( ex, im, t0, u0, t1, delta, dt); //one full step
        m_stepper.step( ex, im, t0, u0, t1, u1, dt/2.);
        m_stepper.step( ex, im, t1, u1, t1, u1, dt/2.);
        dg::blas1::axpby( 1., u1, -1., delta);
        t1 = t0 + dt;
    }
    int order() const{
        return m_stepper.order();
    }
    private:
    ExImStepper m_stepper;
};
///@endcond

///Default Monitor for \c dg::integrateAdaptive function
struct DefaultMonitor
{
    ///same as \c sqrt(dg::blas1::dot(x,x))
    template<class ContainerType>
    get_value_type<ContainerType> norm( const ContainerType& x)const
    {
        return sqrt(dg::blas1::dot( x,x));
    }
};

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
template<class Stepper, class RHS, class ContainerType, class Monitor = DefaultMonitor>
int integrateAdaptive(Stepper& stepper,
                      RHS& rhs,
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
