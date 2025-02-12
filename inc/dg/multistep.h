#pragma once

#include <map>
#include <tuple>
#include "ode.h"
#include "runge_kutta.h"
#include "multistep_tableau.h"

/*! @file
  @brief contains multistep explicit& implicit ODE-integrators
  */
namespace dg{


/*!@class hide_note_multistep
* @note Uses only \c blas1::axpby routines to integrate one step.
* @note The difference between a multistep and a single step method like RungeKutta
* is that the multistep only takes one right-hand-side evaluation per step.
* This is advantageous if the right hand side is expensive to evaluate.
* Even though Runge Kutta methods can have a larger absolute timestep, if
* the effective timestep per rhs evaluation is compared, multistep methods
* generally win.
* @note a disadvantage of multistep is that timestep adaption is not easily done.
*/
///@cond
template<class ContainerType>
struct FilteredExplicitMultistep;
///@endcond
//
///@addtogroup time
///@{

/**
* @brief General explicit linear multistep ODE integrator
* \f$
* \begin{align}
    v^{n+1} = \sum_{j=0}^{s-1} a_j v^{n-j} + \Delta t\left(\sum_{j=0}^{s-1}b_j  \hat f\left(t^{n}-j\Delta t, v^{n-j}\right)\right)
    \end{align}
    \f$

    which discretizes
    \f[
    \frac{\partial v}{\partial t} = \hat f(t,v)
    \f]
    where \f$ f \f$ contains the equations.
    The coefficients for an order 3 "eBDF" scheme are given as an example:
    \f[
    a_0 = \frac{18}{11}\ a_1 = -\frac{9}{11}\ a_2 = \frac{2}{11} \\
    b_0 = \frac{18}{11}\ b_1 = -\frac{18}{11}\ b_2 = \frac{6}{11}
\f]
    You can use your own coefficients defined as a \c dg::MultistepTableau
    or use one of the predefined coefficients in
    @copydoc hide_explicit_multistep_tableaus
*
* @copydoc hide_note_multistep
* @copydoc hide_ContainerType
*/
template<class ContainerType>
struct ExplicitMultistep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc ERKStep::ERKStep()
    ExplicitMultistep() = default;
    ///@copydoc FilteredExplicitMultistep::FilteredExplicitMultistep(ConvertsToMultistepTableau<value_type>,const ContainerType&)
    ExplicitMultistep( ConvertsToMultistepTableau<value_type> tableau, const ContainerType& copyable): m_fem( tableau, copyable){ }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct(Params&& ...ps)
    {
        //construct and swap
        *this = ExplicitMultistep(  std::forward<Params>(ps)...);
    }
    ///@copydoc hide_copyable
    const ContainerType& copyable()const{ return m_fem.copyable();}

    /**
     * @brief Initialize timestepper. Call before using the step function.
     *
     * This routine has to be called before the first timestep is made.
     * @copydoc hide_explicit_rhs
     * @param rhs The rhs functor
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep saved for later use
     * @note the implementation is such that on return the last call to the explicit part \c ex is at \c (t0,u0).
     * This might be interesting if the call to \c ex changes its state.
     */
    template< class ExplicitRHS>
    void init( ExplicitRHS& rhs, value_type t0, const ContainerType& u0, value_type dt){
        dg::IdentityFilter id;
        m_fem.init( std::tie(rhs, id), t0, u0, dt);
    }

    /**
    * @brief Advance one timestep
    *
    * @copydoc hide_explicit_rhs
    * @param rhs The rhs functor
    * @param t (write-only), contains timestep corresponding to \c u on return
    * @param u (write-only), contains next step of time-integration on return
    * @note the implementation is such that on return the last call to the explicit part \c ex is at the new \c (t,u).
    * This might be interesting if the call to \c ex changes its state.
    * @attention The first few steps after the call to the init function are performed with a Runge-Kutta method (of the same order) to initialize the multistepper
    */
    template< class ExplicitRHS>
    void step( ExplicitRHS& rhs, value_type& t, ContainerType& u){
        dg::IdentityFilter id;
        m_fem.step( std::tie(rhs, id), t, u);
    }

  private:
    FilteredExplicitMultistep<ContainerType> m_fem;
};


/**
 * @brief Semi-implicit multistep ODE integrator
 * \f$
 * \begin{align}
     v^{n+1} = \sum_{q=0}^{s-1} a_q v^{n-q} + \Delta t\left[\left(\sum_{q=0}^{s-1}b_q  \hat E(t^{n}-q\Delta t, v^{n-q}) + \sum_{q=1}^{s} c_q \hat I( t^n - q\Delta t, v^{n-q})\right) + c_0\hat I(t^{n}+\Delta t, v^{n+1})\right]
     \end{align}
     \f$

     which discretizes
     \f[
     \frac{\partial v}{\partial t} = \hat E(t,v) + \hat I(t,v)
     \f]
    where \f$ \hat E \f$ contains the explicit and \f$ \hat I \f$ the implicit part of the equations.
    As an example, the coefficients for the 3-step, 3rd order "Karniadakis" scheme are
    \f[
    a_0 = \frac{18}{11}\ a_1 = -\frac{9}{11}\  a_2 = \frac{2}{11} \\
    b_0 = \frac{18}{11}\ b_1 = -\frac{18}{11}\ b_2 = \frac{6}{11} \\
    c_0 = \frac{6}{11}\quad c_1 = c_2 = c_3 = 0 \\
\f]
    You can use your own coefficients defined as a \c dg::MultistepTableau
    or use one of the predefined coefficients in
    @copydoc hide_imex_multistep_tableaus
 *
 * The necessary Inversion in the implicit part is provided by the \c Implicit class.
 *
 * The following code example demonstrates how to implement the method of manufactured solutions on a 2d partial differential equation with the dg library:
 * @snippet multistep_t.cpp function
 * In the main function:
 * @snippet multistep_t.cpp karniadakis
 * @note In our experience the implicit treatment of diffusive or hyperdiffusive
terms may significantly reduce the required number of time steps. This
outweighs the increased computational cost of the additional matrix inversions.
However, each PDE is different and general statements like this one should be
treated with care.
* @copydoc hide_note_multistep
* @copydoc hide_ContainerType
*/
template<class ContainerType>
struct ImExMultistep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc ERKStep::ERKStep()
    ImExMultistep() = default;

    /*! @brief Reserve memory for integration and construct Solver
     *
     * @param tableau Tableau, name or identifier that \c ConvertsToMultistepTableau
     * @param copyable vector of the size that is later used in \c step (
      it does not matter what values \c copyable contains, but its size is important;
      the \c step method can only be called with vectors of the same size)
     */
    ImExMultistep( ConvertsToMultistepTableau<value_type> tableau,
            const ContainerType& copyable):
        m_t(tableau)
    {
        //only store implicit part if needed
        unsigned size_f = 0;
        for( unsigned i=0; i<m_t.steps(); i++ )
        {
            if( m_t.im( i+1) != 0 )
                size_f = i+1;
        }
        m_im.assign( size_f, copyable);
        m_u.assign( m_t.steps(), copyable);
        m_ex.assign( m_t.steps(), copyable);
        m_tmp = copyable;
        m_counter = 0;
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = ImExMultistep( std::forward<Params>( ps)...);
    }
    ///@copydoc hide_copyable
    const ContainerType& copyable()const{ return m_tmp;}

    /**
     * @brief Initialize timestepper. Call before using the step function.
     *
     * This routine has to be called before the first timestep is made.
     * @copydoc hide_explicit_rhs
     * @copydoc hide_implicit_rhs
     * @copydoc hide_solver
     * @param ode the <explicit rhs, implicit rhs, solver for the rhs> functor.
     * Typically \c std::tie(explicit_rhs, implicit_rhs, solver)
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep saved for later use
     * @note the implementation is such that on return the last call is the
     * explicit part \c explicit_part at \c (t0,u0).  This is useful if
     * \c explicit_part holds state, which is then updated to that timestep
     * and/or if \c implicit_rhs changes the state of \c explicit_rhs
     */
    template< class ExplicitRHS, class ImplicitRHS, class Solver>
    void init( const std::tuple<ExplicitRHS, ImplicitRHS, Solver>& ode,
            value_type t0, const ContainerType& u0, value_type dt);

    /**
    * @brief Advance one timestep
    *
    * @copydoc hide_explicit_rhs
    * @copydoc hide_implicit_rhs
    * @copydoc hide_solver
    * @param ode the <explicit rhs, implicit rhs, solver for the rhs> functor.
    * Typically \c std::tie(explicit_rhs, implicit_rhs, solver)
    * @param t (write-only), contains timestep corresponding to \c u on return
    * @param u (write-only), contains next step of time-integration on return
    * @note After a \c solve, we call \c explicit_rhs at the new (t,u)
    * which is the last call before return.
    * This is useful if \c explicit_rhs holds state, which is then updated
    * to the new timestep and/or if \c solver changes the state of \c explicit_rhs
    * @attention The first few steps after the call to the init function are
    * performed with a semi-implicit Runge-Kutta method to initialize the
    * multistepper
    * @note The only time the \c implicit_rhs functor is called is during the
    * initialization phase (the first few steps after the call to the init
    * function)
    */
    template< class ExplicitRHS, class ImplicitRHS, class Solver>
    void step( const std::tuple<ExplicitRHS, ImplicitRHS, Solver>& ode,
            value_type& t, ContainerType& u);

  private:
    dg::MultistepTableau<value_type> m_t;
    std::vector<ContainerType> m_u, m_ex, m_im;
    ContainerType m_tmp;
    value_type m_tu, m_dt;
    unsigned m_counter; //counts how often step has been called after init
};

///@cond
template< class ContainerType>
template< class RHS, class Diffusion, class Solver>
void ImExMultistep<ContainerType>::init( const std::tuple<RHS, Diffusion, Solver>& ode, value_type t0, const ContainerType& u0, value_type dt)
{
    m_tu = t0, m_dt = dt;
    unsigned s = m_t.steps();
    blas1::copy(  u0, m_u[s-1]);
    m_counter = 0;
    if( s-1-m_counter < m_im.size())
        std::get<1>(ode)( m_tu, m_u[s-1-m_counter], m_im[s-1-m_counter]);
    std::get<0>(ode)( t0, u0, m_ex[s-1]); //f may not destroy u0
}

template<class ContainerType>
template< class RHS, class Diffusion, class Solver>
void ImExMultistep<ContainerType>::step( const std::tuple<RHS, Diffusion, Solver>& ode, value_type& t, ContainerType& u)
{
    unsigned s = m_t.steps();
    if( m_counter < s - 1)
    {
        std::map<unsigned, std::string> order2method{
            {1, "ARK-4-2-3"},
            {2, "ARK-4-2-3"},
            {3, "ARK-4-2-3"},
            {4, "ARK-6-3-4"},
            {5, "ARK-8-4-5"},
            {6, "ARK-8-4-5"},
            {7, "ARK-8-4-5"}
        };
        ARKStep<ContainerType> ark( order2method.at( m_t.order()), u);
        ContainerType tmp ( u);
        ark.step( ode, t, u, t, u, m_dt, tmp);
        m_counter++;
        m_tu = t;
        dg::blas1::copy( u, m_u[s-1-m_counter]);
        if( s-1-m_counter < m_im.size())
            std::get<1>(ode)( m_tu, m_u[s-1-m_counter], m_im[s-1-m_counter]);
        std::get<0>(ode)( m_tu, m_u[s-1-m_counter], m_ex[s-1-m_counter]);
        return;
    }
    //compute right hand side of inversion equation
    dg::blas1::axpbypgz( m_t.a(0), m_u[0], m_dt*m_t.ex(0), m_ex[0], 0., m_tmp);
    for (unsigned i = 1; i < s; i++)
        dg::blas1::axpbypgz( m_t.a(i), m_u[i], m_dt*m_t.ex(i), m_ex[i], 1., m_tmp);
    for (unsigned i = 0; i < m_im.size(); i++)
        dg::blas1::axpby( m_dt*m_t.im(i+1), m_im[i], 1., m_tmp);
    t = m_tu = m_tu + m_dt;

    //Rotate 1 to the right (note the reverse iterator here!)
    std::rotate( m_u.rbegin(), m_u.rbegin() + 1, m_u.rend());
    std::rotate( m_ex.rbegin(), m_ex.rbegin() + 1, m_ex.rend());
    if( !m_im.empty())
        std::rotate( m_im.rbegin(), m_im.rbegin() + 1, m_im.rend());
    //compute implicit part
    value_type alpha = m_dt*m_t.im(0);
    std::get<2>(ode)( alpha, t, u, m_tmp);

    blas1::copy( u, m_u[0]); //store result
    if( 0 < m_im.size())
        dg::blas1::axpby( 1./alpha, u, -1./alpha, m_tmp, m_im[0]);
    std::get<0>(ode)(m_tu, m_u[0], m_ex[0]); //call f on new point
}
///@endcond

/**
* @brief Implicit multistep ODE integrator
* \f$
* \begin{align}
    v^{n+1} &= \sum_{i=0}^{s-1} a_i v^{n-i} + \Delta t \sum_{i=1}^{s} c_i\hat I(t^{n+1-i}, v^{n+1-i}) + \Delta t c_{0} \hat I (t + \Delta t, v^{n+1}) \\
    \end{align}
    \f$

    which discretizes
    \f[
    \frac{\partial v}{\partial t} = \hat I(t,v)
    \f]
    where \f$ \hat I \f$ represents the right hand side of the equations.
    You can use your own coefficients defined as a \c dg::MultistepTableau
    or use one of the predefined coefficients in
    @copydoc hide_implicit_multistep_tableaus
    and (any imex tableau can be used in an implicit scheme, disregarding the explicit coefficients)
    @copydoc hide_imex_multistep_tableaus
*
* The necessary Inversion in the implicit part must be provided by the
* \c Implicit class.
*
* @note In our experience the implicit treatment of diffusive or hyperdiffusive
terms can significantly reduce the required number of time steps. This
outweighs the increased computational cost of the additional inversions.
However, each PDE is different and general statements like this one should be
treated with care.
* @copydoc hide_note_multistep
*/
template<class ContainerType>
struct ImplicitMultistep
{

    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc ERKStep::ERKStep()
    ImplicitMultistep() = default;

    /*! @brief Reserve memory for integration
     *
     * @param tableau Tableau, name or identifier that \c ConvertsToMultistepTableau
     * @param copyable vector of the size that is later used in \c step (
     it does not matter what values \c copyable contains, but its size is important;
     the \c step method can only be called with vectors of the same size)
     */
    ImplicitMultistep( ConvertsToMultistepTableau<value_type> tableau, const
            ContainerType& copyable): m_t( tableau)
    {
        unsigned size_f = 0;
        for( unsigned i=0; i<m_t.steps(); i++ )
        {
            if( m_t.im( i+1) != 0 )
                size_f = i+1;
        }
        m_f.assign( size_f, copyable);
        m_u.assign( m_t.steps(), copyable);
        m_tmp = copyable;
        m_counter = 0;
    }

    ///@copydoc hide_construct
    template<class ...Params>
    void construct(Params&& ...ps)
    {
        //construct and swap
        *this = ImplicitMultistep(  std::forward<Params>(ps)...);
    }
    ///@copydoc hide_copyable
    const ContainerType& copyable()const{ return m_tmp;}

    /**
     * @brief Initialize timestepper. Call before using the step function.
     *
     * This routine has to be called before the first timestep is made.
     * @copydoc hide_explicit_rhs
     * @copydoc hide_solver
     * @param ode the <right hand side, solver for the rhs> functors.
     * Typically \c std::tie(implicit_rhs, solver)
     * @attention \c solver is not actually called only \c implicit_rhs
     * and only if the rhs actually needs to be stored
     * (The \c dg::BDF_X_X tableaus in fact completely avoid calling
     * \c implicit_rhs)
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep saved for later use
     * This might be interesting if the call to \c ex changes its state.
     */
    template<class ImplicitRHS, class Solver>
    void init(const std::tuple<ImplicitRHS, Solver>& ode, value_type t0, const ContainerType& u0, value_type dt);

    /**
    * @brief Advance one timestep
    *
    * @copydoc hide_explicit_rhs
    * @copydoc hide_solver
    * @param ode the <right hand side, solver for the rhs> functors.
    * Typically \c std::tie(implicit_rhs, solver)
    * @attention the \c implicit_rhs functor is only called during the
    * initialization phase (the first few steps after the call to the init
    * function) but only if the rhs actually needs to be stored
    * (The \c dg::BDF_X_X tableaus in fact completely avoid calling
    * \c implicit_rhs)
    * @param t (write-only), contains timestep corresponding to \c u on return
    * @param u (write-only), contains next step of time-integration on return
    * This might be interesting if the call to \c ex changes its state.
    * @note The first few steps after the call to the init function are
    * performed with a DIRK method (of the same order) to initialize the
    * multistepper
    */
    template<class ImplicitRHS, class Solver>
    void step(const std::tuple<ImplicitRHS, Solver>& ode, value_type& t, container_type& u);
    private:
    dg::MultistepTableau<value_type> m_t;
    value_type m_tu, m_dt;
    std::vector<ContainerType> m_u;
    std::vector<ContainerType> m_f;
    ContainerType m_tmp;
    unsigned m_counter = 0; //counts how often step has been called after init
};
///@cond
template< class ContainerType>
template<class ImplicitRHS, class Solver>
void ImplicitMultistep<ContainerType>::init(const std::tuple<ImplicitRHS, Solver>& ode,
        value_type t0, const ContainerType& u0, value_type dt)
{
    m_tu = t0, m_dt = dt;
    dg::blas1::copy( u0, m_u[m_u.size()-1]);
    m_counter = 0;
    //only assign to f if we actually need to store it
    unsigned s = m_t.steps();
    if( s-1-m_counter < m_f.size())
        std::get<0>(ode)( m_tu, m_u[s-1-m_counter], m_f[s-1-m_counter]);
}

template< class ContainerType>
template<class ImplicitRHS, class Solver>
void ImplicitMultistep<ContainerType>::step(const std::tuple<ImplicitRHS, Solver>& ode,
        value_type& t, container_type& u)
{
    unsigned s = m_t.steps();
    if( m_counter < s - 1)
    {
        std::map<unsigned, enum tableau_identifier> order2method{
            {1, IMPLICIT_EULER_1_1},
            {2, SDIRK_2_1_2},
            {3, SDIRK_4_2_3},
            {4, SDIRK_5_3_4},
            {5, SANCHEZ_6_5},
            {6, SANCHEZ_7_6},
            {7, SANCHEZ_7_6}
        };
        ImplicitRungeKutta<ContainerType> dirk(
                order2method.at(m_t.order()), u);
        dirk.step( ode, t, u, t, u, m_dt);
        m_counter++;
        m_tu = t;
        dg::blas1::copy( u, m_u[s-1-m_counter]);
        //only assign to f if we actually need to store it
        if( s-1-m_counter < m_f.size())
            std::get<0>(ode)( m_tu, m_u[s-1-m_counter], m_f[s-1-m_counter]);
        return;
    }
    //compute right hand side of inversion equation
    dg::blas1::axpby( m_t.a(0), m_u[0], 0., m_tmp);
    for (unsigned i = 1; i < s; i++)
        dg::blas1::axpby( m_t.a(i), m_u[i], 1., m_tmp);
    for (unsigned i = 0; i < m_f.size(); i++)
        dg::blas1::axpby( m_dt*m_t.im(i+1), m_f[i], 1., m_tmp);
    t = m_tu = m_tu + m_dt;

    //Rotate 1 to the right (note the reverse iterator here!)
    std::rotate(m_u.rbegin(), m_u.rbegin() + 1, m_u.rend());
    if( !m_f.empty())
        std::rotate(m_f.rbegin(), m_f.rbegin() + 1, m_f.rend());
    value_type alpha = m_dt*m_t.im(0);
    std::get<1>(ode)( alpha, t, u, m_tmp);

    dg::blas1::copy( u, m_u[0]);
    if( 0 < m_f.size())
        dg::blas1::axpby( 1./alpha, u, -1./alpha, m_tmp, m_f[0]);
}
///@endcond



/**
* @brief EXPERIMENTAL: General explicit linear multistep ODE integrator with Limiter / Filter
* \f$
* \begin{align}
    \tilde v &= \sum_{j=0}^{s-1} a_j v^{n-j} + \Delta t\left(\sum_{j=0}^{s-1}b_j  \hat f\left(t^{n}-j\Delta t, v^{n-j}\right)\right) \\
    v^{n+1} &= \Lambda\Pi \left( \tilde v\right)
    \end{align}
    \f$
    @copydoc ExplicitMultistep
* @attention The filter function inside the Explicit Multistep method is a
* somewhat experimental feature, so use this class over
* \c dg::ExplicitMultistep at your own risk
*/
template<class ContainerType>
struct FilteredExplicitMultistep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc ERKStep::ERKStep()
    FilteredExplicitMultistep(){ m_u.resize(1); //this makes the copyable function work
    }

    /**
     * @brief Reserve memory for the integration
     *
     * Set the coefficients \f$ a_i,\ b_i\f$
     * @param tableau Tableau, name or identifier that \c ConvertsToMultistepTableau
     * @param copyable ContainerType of the size that is used in \c step
     * @note it does not matter what values \c copyable contains, but its size is important
     */
    FilteredExplicitMultistep( ConvertsToMultistepTableau<value_type> tableau,
            const ContainerType& copyable): m_t(tableau)
    {
        m_f.assign( m_t.steps(), copyable);
        m_u.assign( m_t.steps(), copyable);
        m_counter = 0;
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = FilteredExplicitMultistep( std::forward<Params>( ps)...);
    }
    ///@copydoc hide_copyable
    const ContainerType& copyable()const{ return m_u[0];}

    /**
     * @brief Initialize timestepper. Call before using the step function.
     *
     * This routine has to be called before the first timestep is made.
     * @copydoc hide_explicit_rhs
     * @copydoc hide_limiter
     * @param ode The <rhs, limiter or filter> functor.
     * Typically \c std::tie( explicit_rhs, limiter)
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep saved for later use
     * @note the implementation is such that on return the last call to the
     * explicit part is at \c (t0,u0).
     * This might be interesting if the call to \c ex changes its state.
     */
    template< class ExplicitRHS, class Limiter>
    void init( const std::tuple<ExplicitRHS, Limiter>& ode, value_type t0, const ContainerType& u0, value_type dt);

    /**
    * @brief Advance one timestep
    *
    * @copydoc hide_explicit_rhs
    * @copydoc hide_limiter
    * @param ode The <rhs, limiter or filter> functor.
    * Typically \c std::tie( explicit_rhs, limiter)
    * @param t (write-only), contains timestep corresponding to \c u on return
    * @param u (write-only), contains next step of time-integration on return
    * @note the implementation is such that on return the last call to the
    * explicit part is at the new \c (t,u).
    * This might be interesting if the call to \c ex changes its state.
    * @attention The first few steps after the call to the init function are
    * performed with a Runge-Kutta method
    */
    template< class ExplicitRHS, class Limiter>
    void step( const std::tuple<ExplicitRHS, Limiter>& ode, value_type& t, ContainerType& u);

  private:
    dg::MultistepTableau<value_type> m_t;
    std::vector<ContainerType> m_u, m_f;
    value_type m_tu, m_dt;
    unsigned m_counter; //counts how often step has been called after init
};
///@cond
template< class ContainerType>
template< class ExplicitRHS, class Limiter>
void FilteredExplicitMultistep<ContainerType>::init( const std::tuple<ExplicitRHS, Limiter>& ode, value_type t0, const ContainerType& u0, value_type dt)
{
    m_tu = t0, m_dt = dt;
    unsigned s = m_t.steps();
    dg::blas1::copy( u0, m_u[s-1]);
    std::get<1>(ode)( m_u[s-1]);
    std::get<0>(ode)(m_tu, m_u[s-1], m_f[s-1]); //call f on new point
    m_counter = 0;
}

template<class ContainerType>
template<class ExplicitRHS, class Limiter>
void FilteredExplicitMultistep<ContainerType>::step(const std::tuple<ExplicitRHS, Limiter>& ode, value_type& t, ContainerType& u)
{
    unsigned s = m_t.steps();
    if( m_counter < s-1)
    {
        std::map<unsigned, enum tableau_identifier> order2method{
            {1, SSPRK_2_2},
            {2, SSPRK_2_2},
            {3, SSPRK_3_3},
            {4, SSPRK_5_4},
            {5, SSPRK_5_4},
            {6, SSPRK_5_4},
            {7, SSPRK_5_4}
        };
        ShuOsher<ContainerType> rk( order2method.at(m_t.order()), u);
        rk.step( ode, t, u, t, u, m_dt);
        m_counter++;
        m_tu = t;
        blas1::copy(  u, m_u[s-1-m_counter]);
        std::get<0>(ode)( m_tu, m_u[s-1-m_counter], m_f[s-1-m_counter]);
        return;
    }
    //compute new t,u
    t = m_tu = m_tu + m_dt;
    dg::blas1::axpby( m_t.a(0), m_u[0], m_dt*m_t.ex(0), m_f[0], u);
    for (unsigned i = 1; i < s; i++){
        dg::blas1::axpbypgz( m_t.a(i), m_u[i], m_dt*m_t.ex(i), m_f[i], 1., u);
    }
    //apply limiter
    std::get<1>(ode)( u);
    //permute m_f[s-1], m_u[s-1]  to be the new m_f[0], m_u[0]
    std::rotate( m_f.rbegin(), m_f.rbegin()+1, m_f.rend());
    std::rotate( m_u.rbegin(), m_u.rbegin()+1, m_u.rend());
    blas1::copy( u, m_u[0]); //store result
    std::get<0>(ode)(m_tu, m_u[0], m_f[0]); //call f on new point
}
///@endcond
//


/*! @brief Integrate using a for loop and a fixed non-changeable time-step
 *
 * The implementation (of integrate) is equivalent to
 * @code{.cpp}
  dg::blas1::copy( u0, u1);
  unsigned N = round((t1 - t0)/dt);
  for( unsigned i=0; i<N; i++)
      step( t0, u1);
 * @endcode
 * where \c dt is a given constant. \c t1 can only be matched exactly if the timestep
 * evenly divides the given interval.
 * @note the difference to dg::SinglestepTimeloop is the way the step function
 * is called and the fact that \c dt cannot be changed
 * @ingroup time_utils
 * @sa AdaptiveTimeloop, SinglestepTimeloop
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
struct MultistepTimeloop : public aTimeloop<ContainerType>
{
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>;
    /// no allocation
    MultistepTimeloop( ) = default;
    // We cannot reset dt because that would require access to the Stepper to re-init
    /**
     * @brief Construct using a \c std::function
     *
     * @param step Called in the timeloop as <tt> step( t0, u1) </tt>. Has to advance the ode in-place by \c dt
     * @param dt The constant timestep. Can be set later with \c set_dt. Can be negative.
     */
    MultistepTimeloop( std::function<void ( value_type&, ContainerType&)>
            step, value_type dt ) : m_step(step), m_dt(dt){}
    /**
     * @brief Initialize and bind the step function of a Multistep stepper
     *
     * First call \c stepper.init().
     * Then construct a lambda function that calls the step function of \c stepper
     * with given parameters and stores it internally in a \c std::function
     * @tparam Stepper possible steppers are for example dg::ExplicitMultistep,
     * dg::ImplicitMultistep or dg::ImExMultistep
     * @param stepper If constructed in-place (rvalue), will be copied into the
     * lambda. If an lvalue, then the lambda stores a reference
     * @attention If stepper is an lvalue then you need to make sure
     * that stepper remains valid to avoid a "dangling reference"
     * @copydoc hide_ode
     * @param t0 The initial time (forwarded to <tt> stepper.init( ode, t0, u0, dt) </tt>)
     * @param u0 The initial condition (forwarded to <tt> stepper.init( ode, t0, u0, dt) </tt>)
     * @param dt The constant timestep. Can be negative. Cannot be changed as
     * changing it would require to re-init the multistep stepper (which is
     * hidden in the lambda). (forwarded to <tt> stepper.init( ode, t0, u0, dt) </tt>)
     */
    template<class Stepper, class ODE>
    MultistepTimeloop(
            Stepper&& stepper, ODE&& ode, value_type t0, const
            ContainerType& u0, value_type dt )
    {
        stepper.init( ode, t0, u0, dt);
        m_step = [=, cap = std::tuple<Stepper, ODE>(std::forward<Stepper>(stepper),
                std::forward<ODE>(ode))  ]( auto& t, auto& y) mutable
        {
            std::get<0>(cap).step( std::get<1>(cap), t, y);
        };
        m_dt = dt;
    }

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = MultistepTimeloop( std::forward<Params>( ps)...);
    }

    virtual MultistepTimeloop* clone() const{return new
        MultistepTimeloop(*this);}
    private:
    virtual void do_integrate(value_type& t0, const container_type& u0,
            value_type t1, container_type& u1, enum to mode) const;
    std::function<void ( value_type&, ContainerType&)> m_step;
    virtual value_type do_dt( ) const { return m_dt;}
    value_type m_dt;
};

///@cond
template< class ContainerType>
void MultistepTimeloop<ContainerType>::do_integrate(
        value_type&  t_begin, const ContainerType&
        begin, value_type t_end, ContainerType& end,
        enum to mode ) const
{
    bool forward = (t_end - t_begin > 0);
    if( (m_dt < 0 && forward) || ( m_dt > 0 && !forward) )
        throw dg::Error( dg::Message(_ping_)<<"Timestep has wrong sign! dt "<<m_dt);
    if( m_dt == 0)
        throw dg::Error( dg::Message(_ping_)<<"Timestep may not be zero in MultistepTimeloop!");
    dg::blas1::copy( begin, end);
    if( is_divisable( t_end-t_begin, m_dt))
    {
        unsigned N = (unsigned)round((t_end - t_begin)/m_dt);
        for( unsigned i=0; i<N; i++)
            m_step( t_begin, end);
        return;
    }
    else
    {
        if( dg::to::exact == mode)
            throw dg::Error( dg::Message(_ping_) << "In a multistep integrator dt "
                    <<m_dt<<" has to integer divide the interval "<<t_end-t_begin);
        unsigned N = (unsigned)floor( (t_end-t_begin)/m_dt);
        for( unsigned i=0; i<N+1; i++)
            m_step( t_begin, end);
    }
}
///@endcond

///@}

} //namespace dg
