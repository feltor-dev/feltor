#pragma once

#include "runge_kutta.h"

namespace dg
{

///@addtogroup time_utils
///@{
/*! @brief Compute \f$ \sqrt{\sum_i x_i^2}\f$ using \c dg::blas1::dot
 *
 * The intention of this function is to used in the \c Adaptive timestepping class.
 * @param x Vector to take the norm of
 * @return \c sqrt(dg::blas1::dot(x,x))
* @copydoc hide_ContainerType
 */
template <class ContainerType>
get_value_type<ContainerType> l2norm( const ContainerType& x)
{
    return sqrt( dg::blas1::dot( x,x));
}
///\f$ h_{n+1}= h_n \epsilon_n^{-1/p}\f$
template<class value_type>
value_type i_control( std::array<value_type,3> dt, std::array<value_type,3> eps, unsigned embedded_order, unsigned order)
{
    return dt[0]*pow( eps[0], -1./(value_type)embedded_order);
}
///\f$ h_{n+1}= h_n \epsilon_n^{-0.8/p}\epsilon_{n-1}^{0.31/p}\f$
template<class value_type>
value_type pi_control( std::array<value_type,3> dt, std::array<value_type,3> eps, unsigned embedded_order, unsigned order)
{
    if( dt[1] == 0)
        return i_control( dt, eps, embedded_order, order);
    value_type m_k1 = -0.8, m_k2 = 0.31;
    value_type factor = pow( eps[0], m_k1/(value_type)embedded_order)
                     * pow( eps[1], m_k2/(value_type)embedded_order);
    return dt[0]*factor;
}
/**
 * @brief \f$ h_{n+1}= h_n \epsilon_n^{-0.58/p}\epsilon_{n-1}^{0.21/p}\epsilon_{n-2}^{-0.1/p}\f$
 *
 * PID stands for "Proportional" (the present error), "Integral" (the past error), "Derivative" (the future error). See a good tutorial here https://www.youtube.com/watch?v=UR0hOmjaHp0
 * and further information in
 * <a href="http://runge.math.smu.edu/arkode_dev/doc/guide/build/html/Mathematics.html#">the mathematical primer</a> in the ARKode library.
 * The PID controller is a good controller to start with, it does not overshoot
 * too much, is smooth, has no systematic over- or under-estimation and
 * converges very quickly to the desired timestep. In fact Kennedy and Carpenter, Appl. num. Math., (2003) report
 * that it outperformed other controllers in practical problems
 * @tparam value_type
 * @param dt the present (old, 0), previous [1] and second previous [2] timestep h_n
 * @param eps the error relative to the tolerance of the present [0], previous [1] and second previous [2] timestep
 * @param embedded_order order \c q of the embedded timestep
 * @param order order \c p  of the timestep
 *
 * @return the new timestep
 */
template<class value_type>
value_type pid_control( std::array<value_type,3> dt, std::array<value_type,3> eps, unsigned embedded_order, unsigned order)
{
    if( dt[1] == 0)
        return i_control( dt, eps, embedded_order, order);
    if( dt[2] == 0)
        return pi_control( dt, eps, embedded_order, order);
    value_type m_k1 = -0.58, m_k2 = 0.21, m_k3 = -0.1;
    //value_type m_k1 = -0.37, m_k2 = 0.27, m_k3 = -0.1;
    value_type factor = pow( eps[0], m_k1/(value_type)embedded_order)
                     * pow( eps[1], m_k2/(value_type)embedded_order)
                     * pow( eps[2], m_k3/(value_type)embedded_order);
    return dt[0]*factor;
}

/// \f$ h_{n+1} = h_n \epsilon_n^{-0.367/p}(\epsilon_n/\epsilon_{n-1})^{0.268/p} \f$
template<class value_type>
value_type ex_control( std::array<value_type,3> dt, std::array<value_type,3> eps, unsigned embedded_order, unsigned order)
{
    if( dt[1] == 0)
        return i_control( dt, eps, embedded_order, order);
    value_type m_k1 = -0.367, m_k2 = 0.268;
    value_type factor = pow( eps[0], m_k1/(value_type)embedded_order)
                      * pow( eps[0]/eps[1], m_k2/(value_type)embedded_order);
    return dt[0]*factor;
}
/// \f$ h_{n+1} = h_n (h_n/h_{n-1}) \epsilon_n^{-0.98/p}(\epsilon_n/\epsilon_{n-1})^{-0.95/p} \f$
template<class value_type>
value_type im_control( std::array<value_type,3> dt, std::array<value_type,3> eps, unsigned embedded_order, unsigned order)
{
    if( dt[1] == 0)
        return i_control( dt, eps, embedded_order, order);
    value_type m_k1 = -0.98, m_k2 = -0.95;
    value_type factor = pow( eps[0], m_k1/(value_type)embedded_order)
                     *  pow( eps[0]/eps[1], m_k2/(value_type)embedded_order);
    return dt[0]*dt[0]/dt[1]*factor;
}
/// h_{n+1} = |ex_control| < |im_control| ? ex_control : im_control
template<class value_type>
value_type imex_control( std::array<value_type,3> dt, std::array<value_type,3> eps, unsigned embedded_order, unsigned order)
{
    value_type h1 = ex_control( dt, eps, embedded_order, order);
    value_type h2 = im_control( dt, eps, embedded_order, order);
    // find value closest to zero
    return fabs( h1) < fabs( h2) ? h1 : h2;
}

///@}

///@cond
namespace detail{
template<class value_type>
struct Tolerance
{
    // sqrt(size) is norm( 1)
    Tolerance( value_type rtol, value_type atol, value_type size) :
        m_rtol(rtol*sqrt(size)), m_atol( atol*sqrt(size)){}
    DG_DEVICE
    void operator()( value_type u0, value_type& delta) const{
        delta = delta/ ( m_rtol*fabs(u0) + m_atol);
    }
    private:
    value_type m_rtol, m_atol;
};
} //namespace detail
///@endcond

/*!@class hide_stepper
 *
 * @tparam Stepper A timestepper class that computes the actual timestep
 * and an error estimate, for example an embedded Runge Kutta method
 * \c dg::ERKStep or the additive method \c dg::ARKStep. But really,
 * you can also choose to use your own timestepper class. The requirement
 * is that there is a \c step member function that is called as
 * \b stepper.step( rhs, t0, u0, t1, u1, dt, delta)
 * or \b stepper.step( ex, im, t0, u0, t1, u1, dt, delta)
 * depending on whether a purely explicit/implicit or a semi-implicit stepper
 * is used.
 * Here, t0, t1 and dt are of type \b Stepper::value_type, u0,u1 and delta
 * are vector types of type \b Stepper::container_type& and rhs, ex and im are
 * functors implementing the equations that are forwarded from the caller.
 * The parameters t1, u1 and delta are output parameters and must be updated by
 * the stepper.
 * The \c Stepper must have the \c order() and \c embedded_order() member functions that
 * return the (global) order of the method and its error estimate.
  The <tt> const ContainerType& copyable()const; </tt> member must return
  a container of the size that is later used in \c step
  (it does not matter what values \c copyable contains, but its size is important;
  the \c step method can only be called with vectors of this size)
 */

//%%%%%%%%%%%%%%%%%%%Adaptive%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/*!@brief Driver class for adaptive timestep integration
 *
 * In order to build an adaptive Time integrator you basically need three
 * ingredients: a \c Stepper, a \c ControlFunction and an \c ErrorNorm.
 * The \c Stepper does the actual computation and advances the solution one
 * step further with a given timestep \c dt. Furthermore, it has to come up
 * with an estimate of the error of the solution \f$\delta_{n+1}\f$ and indicate the order of that
 * error.  With the \c ErrorNorm the error estimate can be converted to a
 * scalar that can be compared to given relative and absolute error tolerances
 * \c rtol and \c atol.
 * \f[ \epsilon_{n+1} = || \frac{\delta_{n+1}}{(\epsilon_{rtol} |u_{n}| + \epsilon_{atol})\sqrt{N}}|| \f]
 * where N is the array size, \c n is the solution at the previous timestep and
 * the fraction is to be understood as a pointwise division of the vector elements.
 * The \c ControlFunction will try to keep \f$ \epsilon_{n+1}\f$ close to 1 and
 * comes up with an adapted
 * suggestion for the timestep in the next step.
 *
 * This form of error control entails that on average every point in the solution vector on its
 * own fulfills \f$ |\delta_{n+1,i}| \approx \epsilon_{rtol}|u_{n,i} + \epsilon_{atol}\f$, (which places
 * emphasize on \c atol in regions where the solution is close to zero).
 *
 * However, if \f$\epsilon_{n+1} > r\f$
 * where \c r=2 by default is the user-adaptable reject-limit, the step is
 * rejected and the step will be recomputed and the controller restarted.
 * For more
 * information on these concepts we recommend
 * <a href="http://runge.math.smu.edu/arkode_dev/doc/guide/build/html/Mathematics.html#">the mathematical primer</a> of the ARKode library.
 *
 * For an example on how to use this class in a practical example consider the
 * following code snippet:
 * @snippet multistep_t.cu adaptive
 * @copydoc hide_stepper
 * @note On step rejection, choosing timesteps and introducing restrictions on
 * the controller: here is a quote from professor G. Söderlind (the master of
 * control functions) from a private e-mail conversation:
 *
@note "The issue is that most controllers are best left to do their work
without interference. Each time one interferes with the control loop, one
creates a transient that has to be dealt with by the controller.

@note It is indeed necessary to reject steps every once in a while, but I
usually try to avoid it to the greatest possible extent. In my opinion, the
fear of having a too large error on some steps is vastly exaggerated. You see,
in some steps the error is too large, and in others it is too small, and all
controllers I constructed are designed to be “expectation value correct” in the
sense that if the errors are random, the too large and too small errors
basically cancel in the long run.

@note Even so, there are times when the error is way out of proportion. But I
usually accept an error that is up to, say 2*TOL, which typically won’t cause
any problems. Of course, if one hits a sharp change in the solution, the error
may be much larger, and the step recomputed. But then one must “reset" the
controller, i.e., the controller keeps back information, and it is better to
restart the controller to avoid back information to affect the next step."
@attention Should you use this class instead of a fixed stepsize Multistep say?
The thing to consider especially when solving partial differential equations,
is that the right hand side might be very costly to evaluate. An adaptive
stepper (especially one-step methods) usually calls this right hand side more
often than a multistep (only one call per step). The additional computation of
the error norm in the Adaptive step might also be important since the norm
requires global communication in a parallel program (bad for scaling to many
nodes).  So does the Adaptive timestepper make up for its increased cost
throught the adaption of the stepsize? In some cases the answer might be a
sound Yes.  Especially when there are velocity peaks in the solution the
multistep timestep might be restricted too much. In other cases when the
timestep does not need to be adapted much, a multistep method can be faster.
In any case the big advantage of Adaptive is that it usually just works (even
though it is not fool-proof) and you do not have to spend time finding a
suitable timestep like in the multistep method.
@ingroup time
 */
template<class Stepper>
struct Adaptive
{
    using stepper_type = Stepper;
    using container_type = typename Stepper::container_type; //!< the type of the vector class in use by \c Stepper
    using value_type = typename Stepper::value_type; //!< the value type of the time variable defined by \c Stepper (float or double)
    Adaptive(){}
    /*!@brief Allocate workspace and construct stepper
     * @param ps All parameters are forwarded to the constructor of \c Stepper
     * @tparam StepperParams Type of parameters (deduced by the compiler)
     * @note The workspace for Adaptive is constructed from the \c copyable
     * member of Stepper
     */
    template<class ...StepperParams>
    Adaptive(StepperParams&& ...ps): m_stepper(std::forward<StepperParams>(ps)...),
        m_next(m_stepper.copyable()), m_delta(m_stepper.copyable())
    {
        dg::blas1::copy( 1., m_next);
        m_size = dg::blas1::dot( m_next, 1.);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct(Params&& ...ps)
    {
        //construct and swap
        *this = Adaptive(  std::forward<Params>(ps)...);
    }
    /*!@brief Guess an initial stepsize
     *
     * If you have wondered what stepsize you should choose in the beginning,
     * don't freak out about it. Really, the initial stepsize is not that
     * important, the stepper does not even have to succeed. Usually the
     * control function will very(!) quickly adapt the stepsize in just one or
     * two steps (even if it's several orders of magnitude off in the beginning).
     *
     * Currently, this function won't do much better than if you just choose a
     * smallish number yourself, but it's there for future improvements.
     */
    template<class Explicit, class ErrorNorm = value_type(const container_type&)>
    value_type guess_stepsize( Explicit& ex, value_type t0, const
            container_type& u0, enum direction dir, ErrorNorm& norm, value_type
            rtol, value_type atol);

    ///@brief Allow write access to internal stepper
    ///
    ///Maybe useful to set options in the stepper
    stepper_type& stepper() { return m_stepper;}
    ///@brief Read access to internal stepper
    const stepper_type& stepper() const { return m_stepper;}

    /*!@brief Explicit adaptive step
     *
     * @param rhs The right hand side of the equation to integrate
     * @copydoc hide_adaptive_params
     * @copydoc hide_rhs
     * @copydoc hide_control_error
     */
    template< class RHS,
              class ControlFunction = value_type (std::array<value_type,3>,
                      std::array<value_type,3>, unsigned, unsigned),
              class ErrorNorm = value_type( const container_type&)>
    void step( RHS& rhs,
              value_type t0,
              const container_type& u0,
              value_type& t1,
              container_type& u1,
              value_type& dt,
              ControlFunction control,
              ErrorNorm norm,
              value_type rtol,
              value_type atol,
              value_type reject_limit = 2)
    {
        m_stepper.step( rhs, t0, u0, m_t_next, m_next, dt, m_delta);
        return update( t0, u0, t1, u1, dt, control, norm , rtol, atol, reject_limit);
    }
    /*!@brief Semi-implicit adaptive step
     *
     * @copydoc hide_adaptive_params
     * @copydoc hide_explicit_implicit
     * @copydoc hide_control_error
     */
    template< class Explicit,
              class Implicit,
              class Solver,
              class ControlFunction = value_type (std::array<value_type,3>,
                      std::array< value_type,3>, unsigned , unsigned),
              class ErrorNorm = value_type( const container_type&)>
    void step( Explicit& ex,
              Implicit& im,
              Solver& solve,
              value_type t0,
              const container_type& u0,
              value_type& t1,
              container_type& u1,
              value_type& dt,
              ControlFunction control,
              ErrorNorm norm,
              value_type rtol,
              value_type atol,
              value_type reject_limit = 2
              )
    {
        m_stepper.step( ex, im, solve, t0, u0, m_t_next, m_next, dt, m_delta);
        return update( t0, u0, t1, u1, dt, control, norm , rtol, atol, reject_limit);
    }
    /*!@brief implicit adaptive step
     *
     * @copydoc hide_adaptive_params
     * @copydoc hide_rhs_solve
     * @copydoc hide_control_error
     */
    template< class RHS,
              class Solver,
              class ControlFunction = value_type (std::array<value_type,3>,
                      std::array<value_type,3>, unsigned, unsigned),
              class ErrorNorm = value_type( const container_type&)>
    void step( RHS& rhs,
              Solver& solve,
              value_type t0,
              const container_type& u0,
              value_type& t1,
              container_type& u1,
              value_type& dt,
              ControlFunction& control,
              ErrorNorm& norm,
              value_type rtol,
              value_type atol,
              value_type reject_limit = 2
              )
    {
        m_stepper.step( rhs, solve, t0, u0, m_t_next, m_next, dt, m_delta);
        return update( t0, u0, t1, u1, dt, control, norm , rtol, atol, reject_limit);
    }
    ///Return true if the last stepsize in step was rejected
    bool failed() const {
        return m_failed;
    }

    /**
     * @brief Get the latest error norm relative to solution vector
     *
     * The error of the latest call to \c step
     * @return eps_{n+1}
     */
    value_type get_error( ) const{
        return m_eps0;
    }
    private:
    template< class ControlFunction = value_type (std::array<value_type,3>,
                      std::array<value_type,3>, unsigned, unsigned),
                class ErrorNorm = value_type( const container_type&)>
    void update( value_type t0,
                const container_type& u0,
                value_type& t1,
                container_type& u1,
                value_type& dt,
                ControlFunction& control,
                ErrorNorm& norm,
                value_type rtol,
                value_type atol,
                value_type reject_limit
              )
    {
        dg::blas1::subroutine( detail::Tolerance<value_type>( rtol, atol,
                    m_size), u0, m_delta);
        m_eps0 = norm(m_delta);
        value_type dt0 = dt;
        if( m_eps0 > reject_limit || std::isnan( m_eps0) )
        {
            // if stepper fails, restart controller
            dt = control( {dt0, 0., m_dt2}, {m_eps0, m_eps1, m_eps2},
                    m_stepper.embedded_order(),
                    m_stepper.order());
            if( fabs( dt) > 0.9*fabs(dt0))
                dt = 0.9*dt0;
            //0.9*dt0 is a safety limit
            //that prevents an increase of the timestep in case the stepper fails
            m_failed = true;
            dg::blas1::copy( u0, u1);
            t1 = t0;
        }
        else
        {
            if( m_eps0 < 1e-20) // small or zero
            {
                dt = 1e14*dt0; // a very large number
                m_eps0 = 1e-20; // prevent storing zero
            }
            else
            {
                dt = control( {dt0, m_dt1, m_dt2}, {m_eps0, m_eps1, m_eps2},
                        m_stepper.embedded_order(),
                        m_stepper.order());
            }
            m_eps2 = m_eps1;
            m_eps1 = m_eps0;
            m_dt2 = m_dt1;
            m_dt1 = dt0;
            dg::blas1::copy( m_next, u1);
            t1 = m_t_next;
            m_failed = false;
        }
    }
    bool m_failed = false;
    Stepper m_stepper;
    container_type m_next, m_delta;
    value_type m_size, m_eps0 = 1, m_eps1=1, m_eps2=1;
    value_type m_t_next = 0;
    value_type m_dt1 = 0., m_dt2 = 0.;
};
template<class Stepper>
template<class Explicit, class ErrorNorm>
typename Adaptive<Stepper>::value_type Adaptive<Stepper>::guess_stepsize(
        Explicit& ex, value_type t0, const container_type& u0, enum direction
        dir, ErrorNorm& tol, value_type rtol, value_type atol)
{
    value_type desired_accuracy = rtol*tol(u0) + atol;
    ex( t0, u0, m_next);
    value_type dt = pow(desired_accuracy,
            1./(value_type)m_stepper.order())/tol(m_next);
    if( dir != forward)
        dt*=-1.;
    return dt;
}
/*!@class hide_adaptive_params
 * @param t0 initial time
 * @param u0 initial value at \c t0
 * @param t1 (write only) end time ( equals \c t0+dt on output if the step was
 * accepted, otherwise equals \c t0, may alias \c t0)
 * @param u1 (write only) contains the updated result on output if the step was
 * accepted, otherwise a copy of \c u0 (may alias \c u0)
 * @param dt on input: timestep (see dg::Adaptive::guess_stepsize() for an
 * initial stepsize).
 * On output: stepsize proposed by the controller that can be used to continue
 * the integration in the next step.
 * @param control The control function.
 * For explicit and imex methods, \c dg::pid_control
 * is a good choice with \c dg::ex_control or \c dg::imex_control
 * as an alternative if too many steps fail.
 * For implicit methods use the \c dg::im_control.
 * The task of the control function is to compute a new timestep size
 * based on the old timestep size, the order of the method and the past
 * error(s). The behaviour of the controller is also changed by the
 * \c set_reject_limit function
 * @param norm The error norm. Usually \c dg::l2norm is a good choice, but for
 * very small vector sizes the time for the binary reproducible dot product
 * might become a performance bottleneck. Then it's time for your own
 * implementation.
 * @param rtol the desired relative accuracy. Usually 1e-5 is a good choice.
 * @param atol the desired absolute accuracy. Usually 1e-7 is a good choice.
 * @note The error tolerance is computed such that on average every point in
 * the solution vector fulfills \f$ |\delta_i| \approx r|u_i| + a\f$
 * @note Try not to mess with dt. The controller is best left alone and it does
 * a very good job choosing timesteps. But how do I output my solution at
 * certain (equidistant) timesteps? First, think about if you really, really
 * need that. Why is it so bad to have output at non-equidistant timesteps? If
 * you are still firm, then consider using an interpolation scheme (cf.
 * \c dg::Extrapolation). Let choosing the timestep yourself be the very last
 * option if the others are not viable
 * @note From Kennedy and Carpenter, Appl. num. Math., (2003):
 * "Step-size control is a means by which accuracy, iteration, and to a lesser extent stability are controlled.
The choice of (t) may be chosen from many criteria, among those are the (t) from the accuracy
based step controller, the (t)_inviscid and (t)_viscous associated with the inviscid and viscous stability
limits of the ERK, and the (t)_iter associated with iteration convergence."
 *
 * Our control method is an error based control method.
 * However, due to the CFL (or other stability) condition there might be a sharp
 * barrier in the range of possible stepsizes, i.e the stability limited timestep
 * where the error sharply increses leading to a large number of rejected steps.
 * The pid-controller usually does a good job keeping the timestep "just right"
 * but in future work we may want to consider stability based control methods as well.
 * Furthermore, for implicit or semi-implicit methods the chosen timestep
 * influences how fast the iterative solver converges. If the timestep is too
 * large the solver might take too long to converge and a smaller timestep might
 * be preferable. Even for explicit methods where in each timestep
 * an elliptic equation has to be solved the timestep may influence the
 * convergence rate. Currently, there is no communication implemented between these
 * solvers and the controller, but we may consider it in future releases.
 * @attention When you use the ARKStep in combination with the Adaptive time
 * step algorithm pay attention to solve the implicit part with sufficient
 * accuracy. Else, the error propagates into the time controller, which will
 * then choose the timestep as if the implicit part was explicit i.e. far too
 * small. This might have to do with stiffness-leakage [Kennedy and Carpenter, Appl. num. Math., (2003)]:
"An essential requirement for the viability of stiff/nonstiff IMEX schemes is that the stiffness remains
truely separable. If this were not the case then stiffness would leak out of the stiff terms and stiffen the
nonstiff terms. It would manifest itself as a loss in stability or a forced reduction in stepsize of the nonstiff
terms. A more expensive fully implicit approach might then be required, and hence, methods that leak
substantial stiffness might best be avoided".
 * @note The exact values of \c rtol and \c atol might not be too important.
 * However, don't
 * make \c rtol too large, \c 1e-1 say, since then the controller might get too
 * close to the CFL barrier. The timestepper is still able to crash, mind, even
 * though the chances of that happening are somewhat lower than in a fixed
 * stepsize method.
 */

/*!@class hide_control_error
 *
 * @tparam ControlFunction function or Functor called as dt' = \c control( {dt0, dt1, dt2},
 * {eps0, eps1, eps2}, embedded_order, order), where all parameters are of type
 * value_type except the last two, which are unsigned.
 * @tparam ErrorNorm function or Functor of type value_type( const
 * ContainerType&)
 * @param reject_limit the default value is 2.
 * Sometimes even 2 is not enough, for example
 * when the stepsize fluctuates very much and the stepper fails often
 * then it may help to increase the reject_limit further (to 10 say).
 *
 */

/**
 * @brief The domain that contains all points
 * @ingroup time_utils
 */
struct EntireDomain
{
    ///@brief always true
    template<class T>
    bool contains( T& t) const { return true;}
};

///@addtogroup time
///@{
// On the problem of capturing perfect forwarding in a lambda
//https://stackoverflow.com/questions/26831382/capturing-perfectly-forwarded-variable-in-lambda
//https://vittorioromeo.info/index/blog/capturing_perfectly_forwarded_objects_in_lambdas.html

template<template<class> class Stepper, class ContainerType, class RHS, class
value_type, class ErrorNorm = value_type( const ContainerType&), class
ControlFunction = value_type(std::array<value_type,3>,
        std::array<value_type,3>, unsigned, unsigned)>
std::function<void( value_type&, ContainerType&, value_type&)>
    bind_step( dg::Adaptive<Stepper<ContainerType>>&& adapt,
          RHS&& rhs,
          ControlFunction control,
          ErrorNorm norm,
          value_type rtol,
          value_type atol)
{
    using Adapt = dg::Adaptive<Stepper<ContainerType>>;
    return [=, cap = std::tuple<Adapt, RHS>(std::forward<Adapt>(adapt),
            std::forward<RHS>(rhs))  ]( value_type& t,
            ContainerType& y, value_type& dt) mutable
    {
        std::get<0>(cap).step( std::get<1>(cap), t, y, t, y, dt, control, norm,
                rtol, atol);
    };
}

template<template<class> class Stepper, class ContainerType, class RHS, class Solver,
class value_type, class ErrorNorm = value_type( const ContainerType&), class
ControlFunction = value_type(std::array<value_type,3>,
        std::array<value_type,3>, unsigned, unsigned)>
std::function<void( value_type&, ContainerType&, value_type&)>
    bind_step( dg::Adaptive<Stepper<ContainerType>>&& adapt,
          RHS&& rhs,
          Solver&& solve,
          ControlFunction control,
          ErrorNorm norm,
          value_type rtol,
          value_type atol,
          value_type reject_limit = 2)
{
    using Adapt = dg::Adaptive<Stepper<ContainerType>>;
    adapt.set_reject_limit( reject_limit);
    return [=, cap = std::tuple<Adapt, RHS, Solver>(std::forward<Adapt>(adapt),
            std::forward<RHS>(rhs), std::forward<Solver>(solve)) ](
            value_type& t, ContainerType& y, value_type& dt) mutable
    {
        std::get<0>(cap).step( std::get<1>(cap), std::get<2>(cap), t, y, t, y,
                dt, control, norm, rtol, atol);
    };
}
template<template<class> class Stepper, class ContainerType, class Explicit, class Implicit, class Solver,
class value_type, class ErrorNorm = value_type( const ContainerType&), class
ControlFunction = value_type(std::array<value_type,3>,
        std::array<value_type,3>, unsigned, unsigned)>
std::function<void( value_type&, ContainerType&, value_type&)>
    bind_step( dg::Adaptive<Stepper<ContainerType>>&& adapt,
                  Explicit&& ex,
                  Implicit&& im,
                  Solver&& solve,
                  ControlFunction control,
                  ErrorNorm norm,
                  value_type rtol,
                  value_type atol,
                  value_type reject_limit = 2)
{
    using Adapt = dg::Adaptive<Stepper<ContainerType>>;
    adapt.set_reject_limit( reject_limit);
    return [=, cap = std::tuple<Adapt, Explicit, Implicit,
           Solver>(std::forward<Adapt>(adapt), std::forward<Explicit>(ex),
                   std::forward<Implicit>(im), std::forward<Solver>(solve))  ](
                   value_type& t, ContainerType& y, value_type& dt) mutable
    {
        std::get<0>(cap).step( std::get<1>(cap), std::get<2>(cap),
                std::get<3>(cap), t, y, t, y, dt, control, norm, rtol, atol);
    };
}

/**
 * @brief Integrates a differential equation
 *  monitoring the sanity of integration
 *
 * @param stepper can be called like \c stepper( t0, u0, dt)
 * @param t0 initial time
 * @param u0 initial value at \c t0
 * @param t1 end time
 * @param u1 (write only) contains the result corresponding to t1 on output
 * @param dt The initial timestep guess (if 0 the function chooses something
 * for you). The exact value is not really
 * important, the stepper does not even have to succeed. Usually the
 * control function will very(!) quickly adapt the stepsize in just one or
 * two steps (even if it's several orders of magnitude off in the beginning).
 * @attention The integrator may throw if it detects too small timesteps, too
 * many failures, NaN, Inf, or other non-sanitary behaviour
 * @copydoc hide_ContainerType
 */
template< class Stepper, class ContainerType>
int integrate( Stepper& stepper,
              get_value_type<ContainerType> t0,
              const ContainerType& u0,
              get_value_type<ContainerType> t1,
              ContainerType& u1,
              get_value_type<ContainerType> dt = 1e-6
              )
{
    using  value_type = get_value_type<ContainerType>;
    value_type t_current = t0, dt_current = dt;
    blas1::copy( u0, u1 );
    ContainerType& current(u1);
    if( t1 == t0)
        return 0;
    bool forward = (t1 - t0 > 0);
    if( dt == 0)
        dt_current = 1e-6; // a good a guess as any

    int counter =0;
    while( (forward && t_current < t1) || (!forward && t_current > t1))
    {
        //remember last step
        t0 = t_current;
        if( (forward && t_current+dt_current > t1) || (!forward && t_current +
                    dt_current < t1) )
            dt_current = t1-t_current;
        // Compute a step and error
        stepper( t_current, current, dt_current);
        if( !std::isfinite(dt_current) || fabs(dt_current) < 1e-9*fabs(t1-t0))
            throw dg::Error(dg::Message(_ping_)<<"integrate failed to converge! dt = "<<std::scientific<<dt_current);
        counter++;
    }
    return counter;
}

/**
 * @brief Integrates a differential equation
 *  monitoring the sanity of integration and the integration domain
 *
 * @param stepper can be called like \c stepper( t0, u0, dt) Cannot be
 * a multistep stepper
 * @param t0 initial time
 * @param u0 initial value at \c t0
 * @param t1 (read / write) end time; if the solution leaves the domain
 * contains the last time where the solution still lies within the domain
 * on output
 * @param u1 (write only) contains the result corresponding to t1 on output
 * @param dt The initial timestep guess (if 0 the function chooses something
 * for you). The exact value is not really
 * important, the stepper does not even have to succeed. Usually the
 * control function will very(!) quickly adapt the stepsize in just one or
 * two steps (even if it's several orders of magnitude off in the beginning).
 * @param domain (optional) a restriction of the solution space. The integrator
 * checks after every step if the solution is still within the given domain
 * \c domain.contains(u1). If not, the integrator will bisect the exact domain
 * boundary (up to the given tolerances) and return (t1, u1) that lies closest
 * (but within) the domain boundary.
 * @param eps_root Relative error of root finding algorithm \c dt < eps( t1 + 1)
 * @tparam Domain Must have the \c contains(const ContainerType&) const member
 * function returning true if the given solution is part of the domain,
 * false else (can for example be \c dg::aRealTopology2d)
 * @copydoc hide_ContainerType
 */
template< class Stepper, class ContainerType, class Domain>
int integrate_in_domain( Stepper& stepper,
              get_value_type<ContainerType> t0,
              const ContainerType& u0,
              get_value_type<ContainerType>& t1,
              ContainerType& u1,
              get_value_type<ContainerType> dt,
              Domain&& domain,
              get_value_type<ContainerType> eps_root
              )
{
    using  value_type = get_value_type<ContainerType>;
    value_type t_current = t0, dt_current = dt;
    blas1::copy( u0, u1 );
    ContainerType& current(u1);
    if( t1 == t0)
        return 0;
    bool forward = (t1 - t0 > 0);
    if( dt == 0)
        dt_current = 1e-6; // a good a guess as any

    int counter =0;
    ContainerType last( u0);
    while( (forward && t_current < t1) || (!forward && t_current > t1))
    {
        //remember last step
        t0 = t_current;
        dg::blas1::copy( current, last);
        if( (forward && t_current+dt_current > t1) || (!forward && t_current +
                    dt_current < t1) )
            dt_current = t1-t_current;
        // Compute a step and error
        stepper( t_current, current, dt_current);
        if( !std::isfinite(dt_current) || fabs(dt_current) < 1e-9*fabs(t1-t0))
            throw dg::Error(dg::Message(_ping_)<<"integrate_in_domain failed to converge! dt = "<<std::scientific<<dt_current);
        counter++;
        if( !domain.contains( current) )
        {
            //start bisection between t0 and t1
            t1 = t_current;

            // t0 and last are inside
            dg::blas1::copy( last, current);
            int j_max = 50;
            for(int j=0; j<j_max; j++)
            {
                if( fabs(t1-t0) < eps_root*fabs(t1) + eps_root)
                {
                    return counter;
                }
                dt_current = (t1-t0)/2.;
                t_current = t0;
                // Here we can assume that the timestepper always succeeds
                // because dt_current is always smaller than the previous timestep
                // t_current = t0, current = last (inside)
                stepper( t_current, current, dt_current);
                //stepper( t0, last, t_middle, u1, (t1-t0)/2.);
                counter++;
                if( domain.contains( current) )
                {
                    t0 = t_current;
                    dg::blas1::copy( current, last);
                }
                else
                {
                    t1 = t_current;
                    dg::blas1::copy( last, current);
                }
            }
            throw dg::Error( dg::Message(_ping_)<<"Too many steps in root finding!");
        }
    }
    return counter;
}


/*!
 * @brief Shortcut for \c dg::integrate with an embedded ERK class as timestepper
 *
 * @snippet adaptive_t.cu function
 * @snippet adaptive_t.cu doxygen
 * @param name name of an embedded method that \c ConvertsToButcherTableau
 * @param rhs The right-hand-side
 * @param t0 initial time
 * @param u0 initial value at \c t0
 * @param t1 end time
 * @param u1 (write only) contains the result corresponding to t1 on output
 * @param dt The initial timestep guess (if 0 the function chooses something
 * for you). The exact value is not really
 * important, the stepper does not even have to succeed. Usually the
 * control function will very(!) quickly adapt the stepsize in just one or
 * two steps (even if it's several orders of magnitude off in the beginning).
 * @param control The control function.
 * For explicit and imex methods, \c dg::pid_control
 * is a good choice with \c dg::ex_control or \c dg::imex_control
 * as an alternative if too many steps fail.
 * For implicit methods use the \c dg::im_control.
 * The task of the control function is to compute a new timestep size
 * based on the old timestep size, the order of the method and the past
 * error(s). The behaviour of the controller is also changed by the
 * \c set_reject_limit function
 * @param norm The error norm. Usually \c dg::l2norm is a good choice, but for
 * very small vector sizes the time for the binary reproducible dot product
 * might become a performance bottleneck. Then it's time for your own
 * implementation.
 * @param rtol the desired relative accuracy. Usually 1e-5 is a good choice.
 * @param atol the desired absolute accuracy. Usually 1e-7 is a good choice.
 * @return number of steps
 * @attention The integrator may throw if it detects too small timesteps, too many failures, NaN, Inf, or other non-sanitary behaviour
 * @copydoc hide_rhs
 * @copydoc hide_control_error
 * @copydoc hide_ContainerType
 */
template<class RHS, class ContainerType, class value_type, class ErrorNorm =
    value_type( const ContainerType&), class ControlFunction = value_type
    (std::array<value_type,3>, std::array<value_type,3>, unsigned, unsigned)>
int integrateERK( std::string name,
                  RHS& rhs,
                  value_type t0,
                  const ContainerType& u0,
                  value_type t1,
                  ContainerType& u1,
                  value_type dt,
                  ControlFunction control,
                  ErrorNorm norm,
                  value_type rtol,
                  value_type atol=1e-10,
                  value_type reject_limit = 2
              )
{
    dg::Adaptive<dg::ERKStep<ContainerType>> pd( name,u0);
    auto adapt = [&](double& t, ContainerType& u, double& dt)
    {
        pd.step( rhs, t, u, t, u, dt, control, norm, rtol, atol, reject_limit);
    };
    return integrate( adapt, t0, u0, t1, u1, dt);
}
///@}
}//namespace dg
