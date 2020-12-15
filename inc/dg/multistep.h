#pragma once

#include "implicit.h"
#include "runge_kutta.h"


/*! @file
  @brief contains multistep explicit& implicit time-integrators
  */
namespace dg{

//MW: if ever we want to change the SolverType at runtime (with an input parameter e.g.) make it a new parameter in the solve method (either abstract type or template like RHS)

/*! @class hide_explicit_implicit
 * @tparam Explicit The explicit part of the right hand side
        is a functor type with no return value (subroutine)
        of signature <tt> void operator()(value_type, const ContainerType&, ContainerType&)</tt>
        The first argument is the time, the second is the input vector, which the functor may \b not override, and the third is the output,
        i.e. y' = f(t, y) translates to f(t, y, y').
        The two ContainerType arguments never alias each other in calls to the functor.
 * @tparam Implicit The implicit part of the right hand side
        is a functor type with no return value (subroutine)
        of signature <tt> void operator()(value_type, const ContainerType&, ContainerType&)</tt>
        The first argument is the time, the second is the input vector, which the functor may \b not override, and the third is the output,
        i.e. y' = f(t, y) translates to f(t, y, y').
        The two ContainerType arguments never alias each other in calls to the functor.
    Furthermore, if the \c DefaultSolver is used, the routines %weights(), %inv_weights() and %precond() must be callable
    and return diagonal weights, inverse weights and the preconditioner for the conjugate gradient method.
    The return type of these member functions must be useable in blas2 functions together with the ContainerType type.
 * @param ex explic part
 * @param im implicit part ( must be linear in its second argument and symmetric up to weights)
 */
/*!@class hide_note_multistep
* @note Uses only \c blas1::axpby routines to integrate one step.
* @note The difference between a multistep and a single step method like RungeKutta
* is that the multistep only takes one right-hand-side evaluation per step.
* This might be advantageous if the right hand side is expensive to evaluate like in
* partial differential equations. However, it might also happen that the stability
* region of the one-step method is larger so that a larger timestep can be taken there
* and on average they take just the same rhs evaluations.
* @note a disadvantage of multistep is that timestep adaption is not easily done.
*/

/**
* @brief Struct for Adams-Bashforth explicit multistep time-integration
* \f[ u^{n+1} = u^n + \Delta t\sum_{j=0}^{s-1} b_j f\left(t^n - j \Delta t, u^{n-j}\right) \f]
*
* with coefficients taken from https://en.wikipedia.org/wiki/Linear_multistep_method
* @note This scheme has a smaller region of absolute stability than some of the \c ExplicitMultistep method
* @copydoc hide_note_multistep
* @copydoc hide_ContainerType
* @ingroup time
*/
template<class ContainerType>
struct AdamsBashforth
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///copydoc RungeKutta::RungeKutta()
    AdamsBashforth(){}
    ///@copydoc AdamsBashforth::construct()
    AdamsBashforth( unsigned order, const ContainerType& copyable){
        construct( order, copyable);
    }
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param order (global) order (= number of steps in the multistep) of the method (Currently, one of 1, 2, 3, 4 or 5)
    * @param copyable ContainerType of the size that is used in \c step
    * @note it does not matter what values \c copyable contains, but its size is important
    */
    void construct(unsigned order, const ContainerType& copyable){
        m_k = order;
        m_f.assign( order, copyable);
        m_u = copyable;
        m_ab.resize( order);
        switch (order){
            case 1: m_ab = {1}; break;
            case 2: m_ab = {1.5, -0.5}; break;
            case 3: m_ab = { 23./12., -4./3., 5./12.}; break;
            case 4: m_ab = {55./24., -59./24., 37./24., -3./8.}; break;
            case 5: m_ab = { 1901./720., -1387./360., 109./30., -637./360., 251./720.}; break;
            default: throw dg::Error(dg::Message()<<"Order not implemented in AdamsBashforth!");
        }
    }
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_u;}

    /**
     * @brief Initialize timestepper. Call before using the step function.
     *
     * This routine has to be called before the first timestep is made.
     * @copydoc hide_rhs
     * @param rhs The rhs functor
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep
     * @note the implementation is such that on output the last call to the rhs is at (t0,u0). This might be interesting if the call to the rhs changes its state.
     */
    template< class RHS>
    void init( RHS& rhs, value_type t0, const ContainerType& u0, value_type dt);
    /**
    * @brief Advance u0 one timestep
    *
    * @copydoc hide_rhs
    * @param rhs right hand side function or functor
    * @param t (write-only) contains timestep corresponding to \c u on output
    * @param u (write-only) contains next step of the integration on output
    * @note the implementation is such that on output the last call to the rhs is at the new (t,u). This might be interesting if the call to the rhs changes its state.
    */
    template< class RHS>
    void step( RHS& rhs, value_type& t, ContainerType& u);
  private:
    value_type m_tu, m_dt;
    std::vector<ContainerType> m_f;
    ContainerType m_u;
    std::vector<value_type> m_ab;
    unsigned m_k, m_counter;
};

template< class ContainerType>
template< class RHS>
void AdamsBashforth<ContainerType>::init( RHS& f, value_type t0, const ContainerType& u0, value_type dt)
{
    m_tu = t0, m_dt = dt;
    f( t0, u0, m_f[m_k-1]); //f may not destroy u0
    blas1::copy(  u0, m_u);
    m_counter = 0;
}

template<class ContainerType>
template< class RHS>
void AdamsBashforth<ContainerType>::step( RHS& f, value_type& t, ContainerType& u)
{
    if( m_counter < m_k-1)
    {
        ERKStep<ContainerType> erk( "ARK-4-2-3 (explicit)", u);
        ContainerType tmp ( u);
        erk.step( f, t, u, t, u, m_dt, tmp);
        m_counter++;
        m_tu = t;
        blas1::copy(  u, m_u);
        f( m_tu, m_u, m_f[m_k - 1 - m_counter]);
        return;
    }
    for( unsigned i=0; i<m_k; i++)
        blas1::axpby( m_dt*m_ab[i], m_f[i], 1., m_u);
    //permute m_f[k-1]  to be the new m_f[0]
    std::rotate( m_f.rbegin(), m_f.rbegin()+1, m_f.rend());
    blas1::copy( m_u, u);
    t = m_tu = m_tu + m_dt;
    f( m_tu, m_u, m_f[0]); //evaluate f at new point
}


/**
* @brief Struct for Karniadakis semi-implicit multistep time-integration
* \f[
* \begin{align}
    v^{n+1} = \sum_{q=0}^2 \alpha_q v^{n-q} + \Delta t\left[\left(\sum_{q=0}^2\beta_q  \hat E(t^{n}-q\Delta t, v^{n-q})\right) + \gamma_0\hat I(t^{n}+\Delta t, v^{n+1})\right]
    \end{align}
    \f]

    which discretizes
    \f[
    \frac{\partial v}{\partial t} = \hat E(t,v) + \hat I(t,v)
    \f]
    where \f$ \hat E \f$ contains the explicit and \f$ \hat I \f$ the implicit part of the equations.
    The coefficients are
    \f[
    \alpha_0 = \frac{18}{11}\ \alpha_1 = -\frac{9}{11}\ \alpha_2 = \frac{2}{11} \\
    \beta_0 = \frac{18}{11}\ \beta_1 = -\frac{18}{11}\ \beta_2 = \frac{6}{11} \\
    \gamma_0 = \frac{6}{11}
\f]
    for the default third order method and
    \f[
    \alpha_0 = \frac{4}{3}\ \alpha_1 = -\frac{1}{3}\ \alpha_2 = 0 \\
    \beta_0  = \frac{4}{3}\ \beta_1 = -\frac{2}{3}\ \beta_2 = 0 \\
    \gamma_0 = \frac{2}{3}
\f]
    for a second order method and
    \f[
    \alpha_0 = 1\ \alpha_1 = 0\ \alpha_2 = 0\\
    \beta_0  = 1\ \beta_1 = 0\ \beta_2 = 0\\
    \gamma_0 = 1
\f]
for a semi-implicit (first order) Euler method
*
* The necessary Inversion in the imlicit part is provided by the \c SolverType class.
* Per Default, a conjugate gradient method is used (therefore \f$ \hat I(t,v)\f$ must be linear in \f$ v\f$).
* @note This scheme implements <a href = "https://dx.doi.org/10.1016/0021-9991(91)90007-8"> Karniadakis, et al. J. Comput. Phys. 97 (1991)</a>
* @note The implicit part equals a third order backward differentiation formula (BDF) https://en.wikipedia.org/wiki/Backward_differentiation_formula
* while the explicit part equals the Minimal Projecting method by
<a href = "https://www.ams.org/journals/mcom/1979-33-148/S0025-5718-1979-0537965-0/S0025-5718-1979-0537965-0.pdf"> Alfeld, P., Math. Comput. 33.148 1195-1212 (1979)</a>
* or **extrapolated BDF** in <a href = "https://doi.org/10.1137/S0036142902406326"> Hundsdorfer, W., Ruuth, S. J., & Spiteri, R. J. (2003). Monotonicity-preserving linear multistep methods. SIAM Journal on Numerical Analysis, 41(2), 605-623 </a>
*
The following code example demonstrates how to implement the method of manufactured solutions on a 2d partial differential equation with the dg library:
* @snippet multistep_t.cu function
* In the main function:
* @snippet multistep_t.cu karniadakis
* @note In our experience the implicit treatment of diffusive or hyperdiffusive
terms may significantly reduce the required number of time steps. This
outweighs the increased computational cost of the additional matrix inversions.
However, each PDE is different and general statements like this one should be treated with care.
* @copydoc hide_note_multistep
* @copydoc hide_SolverType
* @copydoc hide_ContainerType
* @ingroup time
*/
template<class ContainerType, class SolverType = dg::DefaultSolver<ContainerType>>
struct Karniadakis
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc RungeKutta::RungeKutta()
    Karniadakis(){}

    ///@copydoc construct()
    template<class ...SolverParams>
    Karniadakis( SolverParams&& ...ps):m_solver( std::forward<SolverParams>(ps)...){
        m_f.fill(m_solver.copyable()), m_u.fill(m_solver.copyable());
        set_order(3);
        m_counter = 0;
    }
    /**
     * @brief Reserve memory for the integration
     *
     * @param ps Parameters that are forwarded to the constructor of \c SolverType
     * @tparam SolverParams Type of parameters (deduced by the compiler)
    */
    template<class ...SolverParams>
    void construct( SolverParams&& ...ps){
        m_solver = Solver( std::forward<SolverParams>(ps)...);
        m_f.fill(m_solver.copyable()), m_u.fill(m_solver.copyable());
        set_order(3);
        m_counter = 0;
    }
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_u[0];}

    ///Write access to the internal solver for the implicit part
    SolverType& solver() { return m_solver;}
    ///Read access to the internal solver for the implicit part
    const SolverType& solver() const { return m_solver;}

    /**
     * @brief Initialize timestepper. Call before using the step function.
     *
     * This routine has to be called before the first timestep is made.
     * @copydoc hide_explicit_implicit
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep saved for later use
     * @note the implementation is such that on output the last call to the explicit part \c ex is at \c (t0,u0). This might be interesting if the call to \c ex changes its state.
     */
    template< class Explicit, class Implicit>
    void init( Explicit& ex, Implicit& im, value_type t0, const ContainerType& u0, value_type dt);

    /**
    * @brief Advance one timestep
    *
    * @copydoc hide_explicit_implicit
    * @param t (write-only), contains timestep corresponding to \c u on output
    * @param u (write-only), contains next step of time-integration on output
    * @note the implementation is such that on output the last call to the explicit part \c ex is at the new \c (t,u). This might be interesting if the call to \c ex changes its state.
    * @attention The first two steps after the call to the init function are performed with a semi-implicit Runge-Kutta method
    */
    template< class Explicit, class Implicit>
    void step( Explicit& ex, Implicit& im, value_type& t, ContainerType& u);

    /**
     * @brief Set the order of the method
     *
     * Change the coefficients \f$ \alpha_i,\ \beta_i,\ \gamma_0\f$ to a first,
     * second or third order set.
     * @param order 1 (Euler), 2 or 3 (default)
     */
    void set_order(unsigned order){
        switch( order){
            case 1:
                a[0] = 1.;  b[0] = 1.;
                a[1] = 0.;  b[1] = 0.;
                a[2] = 0.;  b[2] = 0.;   //Euler
                g0 = 1.;
                break;
            case 2:
                a[0] =  4./3.;  b[0] = 4./3.;
                a[1] = -1./3.;  b[1] = -2./3.;
                a[2] = 0.;      b[2] = 0.;   //2nd Karniadakis
                g0 = 2./3.;
                break;
            default:
                a[0] =  18./11.;    b[0] =  18./11.;
                a[1] = -9./11.;     b[1] = -18./11.;
                a[2] = 2./11.;      b[2] = 6./11.;   //Karniadakis
                g0 = 6./11.;
                break;
        }
    }
  private:
    SolverType m_solver;
    std::array<ContainerType,3> m_u, m_f;
    value_type m_t, m_dt;
    value_type a[3];
    value_type b[3], g0 = 6./11.;
    unsigned m_counter; //counts how often step has been called after init
};

///@cond
template< class ContainerType, class SolverType>
template< class RHS, class Diffusion>
void Karniadakis<ContainerType, SolverType>::init( RHS& f, Diffusion& diff, value_type t0, const ContainerType& u0, value_type dt)
{
    m_t = t0, m_dt = dt;
    blas1::copy(  u0, m_u[2]);
    f( t0, u0, m_f[2]); //f may not destroy u0
    m_counter = 0;
}

template<class ContainerType, class SolverType>
template< class RHS, class Diffusion>
void Karniadakis<ContainerType, SolverType>::step( RHS& f, Diffusion& diff, value_type& t, ContainerType& u)
{
    if( m_counter < 2)
    {
        ARKStep<ContainerType, SolverType> ark( "ARK-4-2-3", m_solver);
        ContainerType tmp ( u);
        ark.step( f, diff, t, u, t, u, m_dt, tmp);
        m_counter++;
        m_t = t;
        if( m_counter == 1)
        {
            blas1::copy(  u, m_u[1]);
            f( m_t, m_u[1], m_f[1]);
        }
        else
        {
            blas1::copy(  u, m_u[0]);
            f( m_t, m_u[0], m_f[0]);
        }
        m_solver = ark.solver();
        return;
    }

    blas1::axpbypgz( m_dt*b[0], m_f[0], m_dt*b[1], m_f[1], m_dt*b[2], m_f[2]);
    blas1::axpbypgz( a[0], m_u[0], a[1], m_u[1], a[2], m_u[2]);
    //permute m_f[2], m_u[2]  to be the new m_f[0], m_u[0]
    std::rotate( m_f.rbegin(), m_f.rbegin()+1, m_f.rend());
    std::rotate( m_u.rbegin(), m_u.rbegin()+1, m_u.rend());
    blas1::axpby( 1., m_f[0], 1., m_u[0]);
    //compute implicit part
    value_type alpha[2] = {2., -1.};
    //value_type alpha[2] = {1., 0.};
    blas1::axpby( alpha[0], m_u[1], alpha[1],  m_u[2], u); //extrapolate previous solutions
    t = m_t = m_t + m_dt;
    m_solver.solve( -m_dt*g0, diff, t, u, m_u[0]);
    blas1::copy( u, m_u[0]); //store result
    f(m_t, m_u[0], m_f[0]); //call f on new point
}
///@endcond

/**
* @brief Struct for Backward differentiation formula implicit multistep time-integration
* \f[
* \begin{align}
    v^{n+1} = \sum_{q=0}^{s-1} \alpha_q v^{n-q} + \Delta t\beta\hat I(t^{n}+\Delta t, v^{n+1})
    \end{align}
    \f]

    which discretizes
    \f[
    \frac{\partial v}{\partial t} = \hat I(t,v)
    \f]
    where \f$ \hat I \f$ represents the right hand side of the equations.
    The coefficients for up to order 6 can be found at
    https://en.wikipedia.org/wiki/Backward_differentiation_formula
*
* The necessary Inversion in the imlicit part is provided by the \c SolverType class.
* Per Default, a conjugate gradient method is used (therefore \f$ \hat I(t,v)\f$ must be linear in \f$ v\f$). For nonlinear right hand side we recommend the AndersonSolver
*
* @note In our experience the implicit treatment of diffusive or hyperdiffusive
terms can significantly reduce the required number of time steps. This
outweighs the increased computational cost of the additional inversions.
* @copydoc hide_note_multistep
* @copydoc hide_SolverType
* @copydoc hide_ContainerType
* @ingroup time
*/
template<class ContainerType, class SolverType = dg::DefaultSolver<ContainerType>>
struct BDF
{

    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc RungeKutta::RungeKutta()
    BDF(){}

    ///@copydoc construct()
    template<class ...SolverParams>
    BDF( unsigned order, SolverParams&& ...ps):m_solver( std::forward<SolverParams>(ps)...), m_u( order, m_solver.copyable()), m_f(m_solver.copyable()) {
        init_coeffs(order);
        m_k = order;
    }

    /*! @brief Reserve memory for integration and construct Solver
     *
     * @param order Order of the BDF formula (1 <= order <= 6)
     * @param ps Parameters that are forwarded to the constructor of \c SolverType
     * @tparam SolverParams Type of parameters (deduced by the compiler)
     */
    template<class ...SolverParams>
    void construct(unsigned order, SolverParams&& ...ps)
    {
        m_solver = SolverType( std::forward<SolverParams>(ps)...);
        init_coeffs(order);
        m_k = order;
        m_u.assign( order, m_solver.copyable());
        m_f = m_solver.copyable();
    }
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_u[0];}
    ///Write access to the internal solver for the implicit part
    SolverType& solver() { return m_solver;}
    ///Read access to the internal solver for the implicit part
    const SolverType& solver() const { return m_solver;}

    ///@copydoc AdamsBashforth::init()
    template<class RHS>
    void init(RHS& rhs, value_type t0, const ContainerType& u0, value_type dt);

    ///@copydoc AdamsBashforth::step()
    template<class RHS>
    void step(RHS& rhs, value_type& t, container_type& u);
    private:
    void init_coeffs(unsigned order){
        switch (order){
            case 1: m_bdf = {1.}; m_beta = 1.; break;
            case 2: m_bdf = {4./3., -1./3.}; m_beta = 2./3.; break;
            case 3: m_bdf = { 18./11., -9./11., 2./11.}; m_beta = 6./11.; break;
            case 4: m_bdf = {48./25., -36./25., 16./25., -3./25.}; m_beta = 12./25.; break;
            case 5: m_bdf = { 300./137., -300./137., 200./137., -75./137., 12./137.}; m_beta = 60/137.; break;
            case 6: m_bdf = { 360./147., -450./147., 400./147., -225./147., 72./147., -10./147.}; m_beta = 60/147.; break;
            default: throw dg::Error(dg::Message()<<"Order not implemented in BDF!");
        }
    }
    SolverType m_solver;
    value_type m_tu, m_dt;
    std::vector<ContainerType> m_u;
    ContainerType m_f;
    std::vector<value_type> m_bdf;
    value_type m_beta;
    unsigned m_k;
};

template< class ContainerType, class SolverType>
template<class RHS>
void BDF<ContainerType, SolverType>::init(RHS& rhs, value_type t0,
    const ContainerType& u0, value_type dt)
{
    m_tu = t0, m_dt = dt;
    dg::blas1::copy(u0, m_u[0]);
    //Perform a number of backward euler steps
    for (unsigned i = 0; i<m_k-1; i++){
        rhs(t0-i*dt,m_u[i], m_f);
        dg::blas1::axpby(-dt,m_f,1.,m_u[i],m_u[i+1]);
    }
    rhs( t0, u0, m_f); // and set state in f to (t0,u0)
}

template< class ContainerType, class SolverType>
template<class RHS>
void BDF<ContainerType, SolverType>::step(RHS& rhs, value_type& t, container_type& u)
{
    //dg::WhichType<RHS> {};
    dg::blas1::axpby( m_bdf[0], m_u[0], 0., m_f);
    for (unsigned i = 1; i < m_k; i++){
        dg::blas1::axpby( m_bdf[i], m_u[i], 1., m_f);
    }
    t = m_tu = m_tu + m_dt;

    value_type alpha[2] = {2., -1.};
    if( m_k > 1 ) //everything higher than Euler
        dg::blas1::axpby( alpha[0], m_u[0], alpha[1],  m_u[1], u);
    else
        dg::blas1::copy( m_u[0], u);
    std::rotate(m_u.rbegin(), m_u.rbegin() + 1, m_u.rend()); //Rotate 1 to the right (note the reverse iterator here!)
    m_solver.solve( -m_dt*m_beta, rhs, t, u, m_f);
    dg::blas1::copy( u, m_u[0]);
}


/**
 * @brief Identify coefficients for linear multistep methods
 * @sa ExplicitMultistep
*  @ingroup time
 */
enum multistep_identifier
{
    /** The family of schemes described in <a href =
     "https://doi.org/10.1137/S0036142902406326"> Hundsdorfer, W., Ruuth, S.
     J., & Spiteri, R. J. (2003). Monotonicity-preserving linear multistep
     methods. SIAM Journal on Numerical Analysis, 41(2), 605-623 </a>
     as **extrapolated BDF**  where it is found to be TVB (**total variation
     bound**), i.e. \f$ || v^n|| \leq M ||v^0||\f$  where the norm signifies
     the total variation semi-norm. (Note that total variation diminishing
     (TVD) means M=1, and strong stability preserving (SSP) is the same as TVD, TVB schemes converge to the correct entropy solutions of hyperbolic conservation laws)
     is the same as the **Minimal Projecting** scheme
     described in <a href =
     "https://www.ams.org/journals/mcom/1979-33-148/S0025-5718-1979-0537965-0/S0025-5718-1979-0537965-0.pdf">
     Alfeld, P., Math. Comput. 33.148 1195-1212 (1979)</a>
     * **Possible orders are 1, 2,..., 7**

    */
    eBDF,
    /** The family of schemes described in <a href="https://doi.org/10.1016/j.jcp.2005.02.029">S.J. Ruuth and W. Hundsdorfer, High-order linear multistep methods with general monotonicity and boundedness properties, Journal of Computational Physics, Volume 209, Issue 1, 2005 </a> as Total variation Bound.
     * These schemes have larger stable step sizes than the eBDF family, for
     * example TVB3 has 38% larger stepsize than eBDF3 and TVB4 has 109% larger
     * stepsize than eBDF4.  **Possible orders are 1,2,...,6**, The CFL
     * conditions in relation to forward Euler are (1-1: 1, 2-2: 0.5, 3-3: 0.54, 4-4: 0.46, 5-5:
     * 0.38, 6-6: 0.33), we disregard the remaining schemes since their
     * CFL conditions are worse
     */
    TVB,
    /** The family of schemes described in <a href="https://doi.org/10.1007/BF02728985">Gottlieb, S. On high order strong stability preserving runge-kutta and multi step time discretizations. J Sci Comput 25, 105–128 (2005)</a> as Strong Stability preserving.
     * We implement the lowest order schemes for each stage. The CFL conditions
     * in relation to forward Euler are (1-1: 1, 2-2: 0.5, 3-2: 0.5, 4-2: 0.66, 5-3:
     * 0.5, 6-3: 0.567).  We disregard the remaining
     * schemes since their CFL condition is worse than a TVB scheme of the same
     * order. These schemes are noteworthy because the coefficients b_i are all positive except for the 2-2 method.
     */
    SSP
};
/**
* @brief Struct for general explicit linear multistep time-integration
* \f[
* \begin{align}
    v^{n+1} = \sum_{j=0}^{s-1} \alpha_j v^{n-j} + \Delta t\left(\sum_{j=0}^{s-1}\beta_j  \hat f\left(t^{n}-j\Delta t, v^{n-j}\right)\right)
    \end{align}
    \f]

    which discretizes
    \f[
    \frac{\partial v}{\partial t} = \hat f(t,v)
    \f]
    where \f$ f \f$ contains the equations.
    The coefficients for an order 3 "eBDF" scheme are given as an example:
    \f[
    \alpha_0 = \frac{18}{11}\ \alpha_1 = -\frac{9}{11}\ \alpha_2 = \frac{2}{11} \\
    \beta_0 = \frac{18}{11}\ \beta_1 = -\frac{18}{11}\ \beta_2 = \frac{6}{11}
\f]
@sa multistep_identifier
@note The schemes implemented here need more storage but may have **a larger region of absolute stability** than an AdamsBashforth method of the same order.
*
* @copydoc hide_note_multistep
* @copydoc hide_ContainerType
* @ingroup time
*/
template<class ContainerType>
struct ExplicitMultistep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc RungeKutta::RungeKutta()
    ExplicitMultistep(){}

    ///@copydoc construct()
    ExplicitMultistep( std::string method, unsigned stages, const ContainerType& copyable){
        std::unordered_map < std::string, enum multistep_identifier> str2id{
            {"eBDF", eBDF},
            {"TVB", TVB},
            {"SSP", SSP}
        };
        if( str2id.find(method) == str2id.end())
            throw dg::Error(dg::Message(_ping_)<<"Multistep coefficients for "<<method<<" not found!");
        else
            construct( str2id[method], stages, copyable);
    }
    ///@copydoc construct()
    ExplicitMultistep( enum multistep_identifier method, unsigned stages, const ContainerType& copyable){
        construct( method, stages, copyable);
    }
    /**
     * @brief Reserve memory for the integration
     *
     * Set the coefficients \f$ \alpha_i,\ \beta_i\f$
     * @param method the name of the family of schemes to be used (a string can be converted to an enum with the same spelling) @sa multistep_identifier
     * @param stages (global) stages (= number of steps in the multistep) of the method (Currently possible values depend on the method), does not necessarily coincide with the order of the method
     * @param copyable ContainerType of the size that is used in \c step
     * @note it does not matter what values \c copyable contains, but its size is important
     */
    void construct( enum multistep_identifier method, unsigned stages, const ContainerType& copyable){
        m_k = stages;
        m_f.assign( stages, copyable);
        m_u.assign( stages, copyable);
        init_coeffs(method, stages);
        m_counter = 0;
    }
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_u[0];}

    /**
     * @brief Initialize timestepper. Call before using the step function.
     *
     * This routine has to be called before the first timestep is made.
     * @copydoc hide_rhs
     * @param rhs The rhs functor
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep saved for later use
     * @note the implementation is such that on output the last call to the explicit part \c ex is at \c (t0,u0). This might be interesting if the call to \c ex changes its state.
     */
    template< class RHS>
    void init( RHS& rhs, value_type t0, const ContainerType& u0, value_type dt);

    /**
    * @brief Advance one timestep
    *
    * @copydoc hide_rhs
    * @param rhs The rhs functor
    * @param t (write-only), contains timestep corresponding to \c u on output
    * @param u (write-only), contains next step of time-integration on output
    * @note the implementation is such that on output the last call to the explicit part \c ex is at the new \c (t,u). This might be interesting if the call to \c ex changes its state.
    * @attention The first few steps after the call to the init function are performed with an explicit Runge-Kutta method
    */
    template< class RHS>
    void step( RHS& rhs, value_type& t, ContainerType& u);

  private:
    void init_coeffs(enum multistep_identifier method, unsigned stages){
        m_a.resize( stages);
        m_b.resize( stages);
        switch( method){
            case eBDF:
            switch (stages){
                case 1: m_a = {1.};
                        m_b = {1.}; break;
                case 2: m_a = {4./3., -1./3.};
                        m_b = {4./3., -2./3.}; break;
                case 3: m_a = { 18./11., -9./11., 2./11.};
                        m_b = { 18./11., -18./11., 6./11.}; break; //CLM = 0.38... (% of Euler step size)
                case 4: m_a = {48./25., -36./25., 16./25., -3./25.};
                        m_b = {48./25.,-72./25.,48./25.,-12./25.}; break; //CLM = 0.21...
                case 5: m_a = { 300./137., -300./137., 200./137., -75./137., 12./137.};
                        m_b = {300./137.,-600./137.,600./137.,-300./137.,60./137.}; break;
                case 6: m_a = { 360./147., -450./147., 400./147., -225./147., 72./147., -10./147.};
                        m_b = {360./147.,-900./147.,1200./147.,-900./147.,360./147.,-60./147.}; break;
                case 7: m_a = { 2940./1089.,-4410./1089.,4900./1089.,-3675./1089.,1764./1089.,-490./1089.,60./1089.};
                        m_b = { 2940./1089.,-8820./1089.,14700./1089.,-14700./1089.,8820./1089.,-2940./1089.,420./1089.}; break;
                default: throw dg::Error(dg::Message()<<"Order not implemented in Minimal Projecting!");
            }
            break;
            case TVB:
            switch(stages){
                case 1: m_a = {1.};
                        m_b = {1.}; break;
                case 2: m_a = {4./3., -1./3.};
                        m_b = {4./3., -2./3.}; break; //CLM = 0.5
                case 3: //CLM = 0.54...
                    m_a[0] =  1.908535476882378;     m_b[0] =  1.502575553858997;
                    m_a[1] = -1.334951446162515;     m_b[1] = -1.654746338401493;
                    m_a[2] = 0.426415969280137;      m_b[2] = 0.670051276940255;
                    break;
                case 4: //CLM = 0.45...
                    m_a[0] = 2.628241000683208;     m_b[0] = 1.618795874276609;
                    m_a[1] = -2.777506277494861;     m_b[1] = -3.052866947601049;
                    m_a[2] = 1.494730011212510;     m_b[2] = 2.229909318681302;
                    m_a[3] = -0.345464734400857;     m_b[3] = -0.620278703629274;
                    break;
                case 5: //CLM = 0.37...
                    m_a[0] = 3.308891758551210;     m_b[0] = 1.747442076919292;
                    m_a[1] = -4.653490937946655;     m_b[1] = -4.630745565661800;
                    m_a[2] = 3.571762873789854;     m_b[2] = 5.086056171401077;
                    m_a[3] = -1.504199914126327;     m_b[3] = -2.691494591660196;
                    m_a[4] = 0.277036219731918;     m_b[4] = 0.574321855183372;
                    break;
                case 6: //CLM = 0.32...
                    m_a[0] = 4.113382628475685;     m_b[0] = 1.825457674048542;
                    m_a[1] = -7.345730559324184;     m_b[1] = -6.414174588309508;
                    m_a[2] = 7.393648314992094;     m_b[2] = 9.591671249204753;
                    m_a[3] = -4.455158576186636;     m_b[3] = -7.583521888026967;
                    m_a[4] = 1.523638279938299;     m_b[4] = 3.147082225022105;
                    m_a[5] = -0.229780087895259;     m_b[5] = -0.544771649561925;
                    break;
                default: throw dg::Error(dg::Message()<<"Order not implemented in TVB scheme!");
            }
            break;
            case SSP:
            switch(stages){
                case 1: m_a = {1.};
                        m_b = {1.}; break;
                case 2: m_a = {4./5., 1./5.};
                        m_b = {8./5., -2./5.}; break; //CLM = 0.5 ... ,order 2
                case 3: m_a = { 3./4., 0., 1./4.};
                        m_b = { 3./2., 0., 0. }; break; //CLM = 0.5..., order 2
                case 4: m_a = {8./9., 0., 0., 1./9.};
                        m_b = {4./3., 0., 0., 0.}; break; //CLM = 0.66..., order 2
                case 5: m_a = {25./32., 0., 0., 0., 7./32.};
                        m_b = {25./16.,0.,0.,0.,5./16.}; break; //CLM 0.5, order 3
                case 6: m_a = {108./125.,0.,0.,0.,0.,17./125.};
                        m_b = {36./25.,0.,0.,0.,0.,6./25.}; break; //CLM 0.567, order 3
                default: throw dg::Error(dg::Message()<<"Stage not implemented in SSP scheme!");
            }
            break;
        }
    }
    std::vector<ContainerType> m_u, m_f;
    value_type m_tu, m_dt;
    std::vector<value_type> m_a, m_b;
    unsigned m_k, m_counter; //counts how often step has been called after init
};

///@cond
template< class ContainerType>
template< class RHS>
void ExplicitMultistep<ContainerType>::init( RHS& f, value_type t0, const ContainerType& u0, value_type dt)
{
    m_tu = t0, m_dt = dt;
    blas1::copy(  u0, m_u[m_k-1]);
    f(m_tu, m_u[m_k-1], m_f[m_k-1]); //call f on new point
    m_counter = 0;
}

template<class ContainerType>
template< class RHS>
void ExplicitMultistep<ContainerType>::step( RHS& f, value_type& t, ContainerType& u)
{
    if( m_counter < m_k-1)
    {
        ERKStep<ContainerType> erk( "ARK-4-2-3 (explicit)", u);
        ContainerType tmp ( u);
        erk.step( f, t, u, t, u, m_dt, tmp);
        m_counter++;
        m_tu = t;
        blas1::copy(  u, m_u[m_k-1-m_counter]);
        f( m_tu, m_u[m_k-1-m_counter], m_f[m_k-1-m_counter]);
        return;
    }
    //compute new t,u
    t = m_tu = m_tu + m_dt;
    dg::blas1::axpby( m_a[0], m_u[0], m_dt*m_b[0], m_f[0], u);
    for (unsigned i = 1; i < m_k; i++){
        dg::blas1::axpbypgz( m_a[i], m_u[i], m_dt*m_b[i], m_f[i], 1., u);
    }
    //permute m_f[m_k-1], m_u[m_k-1]  to be the new m_f[0], m_u[0]
    std::rotate( m_f.rbegin(), m_f.rbegin()+1, m_f.rend());
    std::rotate( m_u.rbegin(), m_u.rbegin()+1, m_u.rend());
    blas1::copy( u, m_u[0]); //store result
    f(m_tu, m_u[0], m_f[0]); //call f on new point
}
///@endcond

} //namespace dg
