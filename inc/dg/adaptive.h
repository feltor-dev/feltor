#pragma once

#include "implicit.h"

namespace dg
{
/**
* @brief Struct for Prince-Dormand explicit time-step with error estimate
* \f[
 \begin{align}
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j \\
    k_j = f\left( u^n + \Delta t \sum_{l=1}^j a_{jl} k_l\right)
 \end{align}
\f]

The coefficients for the Butcher tableau were taken from https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
The Prince Dormand method is an embedded Runge Kutta method, i.e. computes a solution together with an error estimate. It is effecitve due to its First Same as Last property.

* @ingroup time
*
* @copydoc hide_ContainerType
* @attention Call the \c reset() member before using the \c step() method on
* a different Equation
*/
template< class ContainerType>
struct PrinceDormand
{
    using real_type = get_value_type<ContainerType>;
    ///@copydoc RK_opt::RK_opt()
    PrinceDormand(){
        m_init=true;
    }
    ///@copydoc RK_opt::construct(const ContainerType&)
    PrinceDormand( const ContainerType& copyable){
        construct( copyable);
    }
    ///@copydoc RK_opt::construct(const ContainerType&)
    void construct( const ContainerType& copyable){
        m_k.fill( copyable);
        m_u = copyable;
        m_init = true;
    }
    /*!@brief Call this function before using step on a
     * different ODE
     *
     * The reason behind this is that the implementation uses the last function evaluation as the initial value for the integration
    */
    void reset() {
        m_init=true;
    }
    ///@copydoc RK_opt::step(RHS&,real_type,const ContainerType&,real_type&,ContainerType&,real_type)
    ///@param delta Contains error estimate on output (must have equal sizeas u0)
    template<class RHS>
    void step( RHS& rhs, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta);
    ///global order of the algorithm
    int order() const {return 4;}
  private:
    std::array<ContainerType,7> m_k;
    ContainerType m_u;
    bool m_init;
};

template< class ContainerType>
template< class RHS>
void PrinceDormand<ContainerType>::step( RHS& f, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta)
{
    //0 stage
    if( m_init){
        f(t0, u0, m_k[0]); //compute k_0
        m_init = false;
    }
    else
        blas1::copy( m_k[6], m_k[0]);
    real_type tu;
    //1 stage
    blas1::axpby( 1., u0, 0.2*dt, m_k[0], m_u);
    tu = t0 + 0.2*dt;
    f( tu, m_u, m_k[1]);
    //2 stage
    blas1::subroutine( m_u, dg::equals(), PairSum(), 1., u0, dt*3./40., m_k[0], dt*9./40., m_k[1]);
    //blas1::evaluate( U2(), m_u, u0, m_k[0],m_k[1], dt);
    tu = t0 + 0.3*dt;
    f( tu, m_u, m_k[2]);
    //3 stage
    blas1::subroutine( m_u, dg::equals(), PairSum(), 1.        , u0,
                         dt*44./45., m_k[0], -dt*56./15., m_k[1], dt*32./9., m_k[2]);
    tu = t0 + 0.8*dt;
    f( tu, m_u, m_k[3]);
    //4 stage
    blas1::subroutine( m_u, dg::equals(), PairSum(), 1.             , u0,
                         dt*19372./6561.,m_k[0], -dt*25360./2187.,m_k[1],
                        +dt*64448./6561.,m_k[2], -dt*212./729.   ,m_k[3]);
    tu = t0 + 8./9.*dt;
    f( tu, m_u, m_k[4]);
    //5 stage
    blas1::subroutine( m_u, dg::equals(), PairSum(), 1., u0,
            dt*9017./3168.,m_k[0], -dt*355./33.,m_k[1], dt*46732./5247.,m_k[2],
            dt*49./176.,   m_k[3], -dt*5103./18656., m_k[4]);
    tu = t0 + dt;
    f( tu, m_u, m_k[5]);
    //6 stage
    blas1::subroutine( u1, dg::equals(), PairSum(), 1., u0,
                       dt*35./384. ,m_k[0],  dt*500./1113., m_k[2],
                       dt*125./192.,m_k[3], -dt*2187./6784.,m_k[4], dt*11./84., m_k[5]);
    t1 = t0 + dt;
    f( t1, u1, m_k[6]);
    //Now add everything up to get error estimate
    blas1::subroutine( delta, dg::equals(), PairSum(),
            dt*(35./384.-5179./57600.),m_k[0], 
            dt*(500./1113.-7571./16695.),m_k[2],
            dt*(125./192.-393./640.),m_k[3],
            -dt*(2187./6784.-92097/339200.),m_k[4],
            dt*(11./84.-187./2100.),m_k[5],
            -dt/40.,m_k[6]);
}
template< class ContainerType>
struct ARK423
{
    using real_type = get_value_type<ContainerType>;
    template< class Explicit, class Implicit>
    void step( Explicit& exp, Implicit& imp, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta);
    private:
    std::array<ContainerType, 4> m_kE, m_kI;
    ContainerType m_u, m_rhs;
    real_type m_eps;
};
template< class ContainerType>
template< class Explicit, class Implicit>
void ARK423<ContainerType>::step( Explicit& ex, Implicit& im, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta)
{
    //1. stage
    //a^E_00 = a^I_00 = 0
    ex(t0, u0, m_kE[0]);
    im(t0, u0, m_kI[0]);

    //2. stage
    blas1::subroutine( m_rhs, dg::equals(), PairSum(), 1., u0, 
            dt*1767732205903./ 2027836641118., m_kE[0], 
            dt*1767732205903./4055673282236., m_kI[0]);
    real_type tu = t0 + dt * 1767732205903./2027836641118.;
    detail::Implicit<Implicit, ContainerType> implicit( -dt*1767732205903./4055673282236., tu, im);
    blas2::symv( im.weights(), m_rhs, m_rhs);
    m_pcg( implicit, m_u, m_rhs, im.precond(), im.inv_weights(), m_eps);
    ex(tu, m_u, m_kE[1]);
    im(tu, m_u, m_kI[1]);

    //3. stage
    blas1::subroutine( m_rhs, dg::equals(), PairSum(), 1., u0, 
             dt*5535828885825./10492691773637., m_kE[0], 
             dt*788022342437./10882634858940., m_kE[1],
             dt*2746238789719./10658868560708., m_kI[0], 
            -dt*640167445237./6845629431997., m_kI[1]);
    tu = t0 + 3./5.*dt;
    implicit.alpha() = -dt*1767732205903./4055673282236.;
    implicit.time()  = tu;
    blas2::symv( im.weights(), m_rhs, m_rhs);
    m_pcg( implicit, m_u, m_rhs, im.precond(), im.inv_weights(), m_eps);
    ex(tu, m_u, m_kE[2]);
    im(tu, m_u, m_kI[2]);
    //4. stage
    blas1::subroutine( m_rhs, dg::equals(), PairSum(), 1., u0, 
             dt*6485989280629./16251701735622., m_kE[0], 
            -dt*4246266847089./9704473918619.,  m_kE[1], 
             dt*10755448449292./10357097424841., m_kE[2],
             dt*1471266399579./7840856788654., m_kI[0],
            -dt*4482444167858./7529755066697., m_kI[1], 
             dt*11266239266428./11593286722821., m_kI[2]);
    t1 = tu = t0 + dt;
    implicit.alpha() = -dt*1767732205903./4055673282236.;
    implicit.time()  = tu;
    blas2::symv( im.weights(), m_rhs, m_rhs);
    m_pcg( implicit, m_u, m_rhs, im.precond(), im.inv_weights(), m_eps);
    ex(tu, m_u, m_kE[3]);
    im(tu, m_u, m_kI[3]);
    //Now compute result and error estimate
    blas1::subroutine( u1, dg::equals(), PairSum(), 1., u0,
             dt*1471266399579./7840856788654., m_kE[0],
            -dt*4482444167858./7529755066697., m_kE[1], 
             dt*11266239266428./11593286722821., m_kE[2],
             dt*1767732205903./4055673282236., m_kE[3],
             dt*1471266399579./7840856788654., m_kI[0],
            -dt*4482444167858./7529755066697., m_kI[1], 
             dt*11266239266428./11593286722821., m_kI[2],
             dt*1767732205903./4055673282236., m_kI[3]);
    blas1::subroutine( delta, dg::equals(), PairSum(), +1., u1,
            -dt*2756255671327./12835298489170., m_kE[0],
             dt*10771552573575./22201958757719., m_kE[1], 
            -dt*9247589265047./10645013368117., m_kE[2],
            -dt*2193209047091./5459859503100., m_kE[3],
            -dt*2756255671327./12835298489170., m_kI[0],
             dt*10771552573575./22201958757719., m_kI[1], 
            -dt*9247589265047./10645013368117., m_kI[2],
            -dt*2193209047091./5459859503100., m_kI[3]);
}

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
template<class ExpImpStepper>
struct ExpImpHalfStep
{
    ExpImpHalfStep(){}
    ExpImpHalfStep( const ExpImpStepper& copyable): m_stepper(copyable){}
    template <class Explicit, class Implicit, class ContainerType>
    void step( std::pair<Explicit&, Implicit&> rhs,
               get_value_type<ContainerType> t0,
               const ContainerType& u0,
               get_value_type<ContainerType>&  t1,
               ContainerType& u1,
               get_value_type<ContainerType> dt,
               ContainerType& delta)
    {
        m_stepper.step( std::get<0>(rhs), std::get<1>(rhs), t0, u0, t1, delta, dt); //one full step
        m_stepper.step( std::get<0>(rhs), std::get<1>(rhs), t0, u0, t1, u1, dt/2.);
        m_stepper.step( std::get<0>(rhs), std::get<1>(rhs), t1, u1, t1, u1, dt/2.);
        dg::blas1::axpby( 1., u1, -1., delta);
        t1 = t0 + dt;
    }
    int order() const{
        return m_stepper.order();
    }
    private:
    ExpImpStepper m_stepper;
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
    dg::ExpImpHalfStep<dg::SIRK<ContainerType>> sirk( begin);
    return integrateAdaptive( sirk, std::make_pair(std::ref(exp),std::ref(imp)), t_begin, begin, t_end, end, dt, eps_rel, eps_abs, epus, monitor);
}
///@}
}//namespace dg
