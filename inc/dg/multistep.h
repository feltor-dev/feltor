#pragma once

#include "implicit.h"


/*! @file
  @brief contains multistep explicit& implicit time-integrators
  */
namespace dg{


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
* @copydoc hide_note_multistep
* @copydoc hide_ContainerType
* @ingroup time
*/
template<class ContainerType>
struct AdamsBashforth
{
    using real_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container = ContainerType; //!< the type of the vector class in use
    ///copydoc RungeKutta::RungeKutta()
    AdamsBashforth(){}
    ///@copydoc AdamsBashforth::construct()
    AdamsBashforth( const ContainerType& copyable, unsigned order){
        construct(copyable, order);
    }
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param copyable ContainerType of the size that is used in \c step
    * @param order (global) order (= number of steps in the multistep) of the method (Currently, one of 1, 2, 3, 4 or 5)
    * @note it does not matter what values \c copyable contains, but its size is important
    */
    void construct(const ContainerType& copyable, unsigned order){
        m_k = order;
        f_.assign( order, copyable);
        u_ = copyable;
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

    /**
     * @brief Initialize first step. Call before using the step function.
     *
     * This routine initiates the first steps in the multistep method by integrating
     * backwards in time with Euler's method. This routine has to be called
     * before the first timestep is made.
     * @copydoc hide_rhs
     * @param rhs The rhs functor
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep
     * @note the implementation is such that on output the last call to the rhs is at (t0,u0). This might be interesting if the call to the rhs changes its state.
     */
    template< class RHS>
    void init( RHS& rhs, real_type t0, const ContainerType& u0, real_type dt);
    /**
    * @brief Advance u0 one timestep
    *
    * @copydoc hide_rhs
    * @param f right hand side function or functor
    * @param t (write-only) contains timestep corresponding to \c u on output
    * @param u (write-only) contains next step of the integration on output
    * @note the implementation is such that on output the last call to the rhs is at the new (t,u). This might be interesting if the call to the rhs changes its state.
    */
    template< class RHS>
    void step( RHS& f, real_type& t, ContainerType& u);
  private:
    real_type tu_, dt_;
    std::vector<ContainerType> f_;
    ContainerType u_;
    std::vector<real_type> m_ab;
    unsigned m_k;
};

template< class ContainerType>
template< class RHS>
void AdamsBashforth<ContainerType>::init( RHS& f, real_type t0, const ContainerType& u0, real_type dt)
{
    tu_ = t0, dt_ = dt;
    f( t0, u0, f_[0]);
    //now do k Euler steps
    ContainerType u1(u0);
    for( unsigned i=1; i<m_k; i++)
    {
        blas1::axpby( 1., u1, -dt, f_[i-1], u1);
        tu_ -= dt;
        f( tu_, u1, f_[i]);
    }
    tu_ = t0;
    blas1::copy(  u0, u_);
    //finally evaluate f at u0 once more to set state in f
    f( tu_, u_, f_[0]);
}

template<class ContainerType>
template< class RHS>
void AdamsBashforth<ContainerType>::step( RHS& f, real_type& t, ContainerType& u)
{
    for( unsigned i=0; i<m_k; i++)
        blas1::axpby( dt_*m_ab[i], f_[i], 1., u_);
    //permute f_[k-1]  to be the new f_[0]
    for( unsigned i=m_k-1; i>0; i--)
        f_[i-1].swap( f_[i]);
    blas1::copy( u_, u);
    t = tu_ = tu_ + dt_;
    f( tu_, u_, f_[0]); //evaluate f at new point
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
*
* The necessary Inversion in the imlicit part is provided by the \c SolverType class.
* Per Default, a conjugate gradient method is used (therefore \f$ \hat I(t,v)\f$ must be linear in \f$ v\f$).
* @note The implicit part equals a third order backward differentiation formula (BDF) https://en.wikipedia.org/wiki/Backward_differentiation_formula
*
The following code example demonstrates how to implement the method of manufactured solutions on a 2d partial differential equation with the dg library:
* @snippet multistep_t.cu function
* In the main function:
* @snippet multistep_t.cu karniadakis
* @note In our experience the implicit treatment of diffusive or hyperdiffusive
terms can significantly reduce the required number of time steps. This
outweighs the increased computational cost of the additional matrix inversions.
* @copydoc hide_note_multistep
* @copydoc hide_ContainerType
* @ingroup time
*/
template<class ContainerType, class SolverType = dg::DefaultSolver<ContainerType>>
struct Karniadakis
{
    using real_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container = ContainerType; //!< the type of the vector class in use
    ///@copydoc RungeKutta::RungeKutta()
    Karniadakis(){}

    ///@copydoc construct()
    template<class ...SolverParams>
    Karniadakis( const ContainerType& copyable, SolverParams&& ...ps):m_solver( copyable, std::forward<SolverParams>(ps)...){
        f_.fill(copyable), u_.fill(copyable);
        init_coeffs();
    }
    /**
     * @brief Reserve memory for the integration
     *
     * @param copyable ContainerType of size which is used in integration (values do not matter, the size is important).
     * @param ps Parameters that, together with \c copyable as the first parameter,
     * are forwarded to the constructor of \c SolverType
     * @tparam SolverParams Type of parameters (deduced by the compiler)
    */
    template<class ...SolverParams>
    void construct( const ContainerType& copyable, SolverParams&& ...ps){
        f_.fill(copyable), u_.fill(copyable);
        m_solver = Solver( copyable, std::forward<SolverParams>(ps)...);
        init_coeffs();
    }

    /**
     * @brief Initialize by integrating two timesteps backward in time
     *
     * The backward integration uses the Lie operator splitting method, with explicit Euler substeps for both explicit and implicit part
     * @copydoc hide_explicit_implicit
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep saved for later use
     * @note the implementation is such that on output the last call to the explicit part \c ex is at \c (t0,u0). This might be interesting if the call to \c ex changes its state.
     */
    template< class Explicit, class Implicit>
    void init( Explicit& ex, Implicit& im, real_type t0, const ContainerType& u0, real_type dt);

    /**
    * @brief Advance one timestep
    *
    * @copydoc hide_explicit_implicit
    * @param t (write-only), contains timestep corresponding to \c u on output
    * @param u (write-only), contains next step of time-integration on output
     * @note the implementation is such that on output the last call to the explicit part \c ex is at the new \c (t,u). This might be interesting if the call to \c ex changes its state.
    */
    template< class Explicit, class Implicit>
    void step( Explicit& ex, Implicit& im, real_type& t, ContainerType& u);

  private:
    void init_coeffs(){
        //a[0] =  1.908535476882378;  b[0] =  1.502575553858997;
        //a[1] = -1.334951446162515;  b[1] = -1.654746338401493;
        //a[2] =  0.426415969280137;  b[2] =  0.670051276940255;
        a[0] =  18./11.;    b[0] =  18./11.;
        a[1] = -9./11.;     b[1] = -18./11.;
        a[2] = 2./11.;      b[2] = 6./11.;   //Karniadakis !!!
    }
    std::array<ContainerType,3> u_, f_;
    SolverType m_solver;
    real_type t_, dt_;
    real_type a[3];
    real_type b[3], g0 = 6./11.;
};

///@cond
template< class ContainerType, class SolverType>
template< class RHS, class Diffusion>
void Karniadakis<ContainerType, SolverType>::init( RHS& f, Diffusion& diff, real_type t0, const ContainerType& u0, real_type dt)
{
    //operator splitting using explicit Euler for both explicit and implicit part
    t_ = t0, dt_ = dt;
    blas1::copy(  u0, u_[0]);
    f( t0, u0, f_[0]); //f may not destroy u0
    blas1::axpby( 1., u_[0], -dt, f_[0], f_[1]); //Euler step
    detail::Implicit<Diffusion, ContainerType> implicit( -dt, t0, diff);
    implicit( f_[1], u_[1]); //explicit Euler step backwards
    f( t0-dt, u_[1], f_[1]);
    blas1::axpby( 1.,u_[1], -dt, f_[1], f_[2]);
    implicit.time() = t0 - dt;
    implicit( f_[2], u_[2]);
    f( t0-2*dt, u_[2], f_[2]); //evaluate f at the latest step
    f( t0, u0, f_[0]); // and set state in f to (t0,u0)
}

template<class ContainerType, class SolverType>
template< class RHS, class Diffusion>
void Karniadakis<ContainerType, SolverType>::step( RHS& f, Diffusion& diff, real_type& t, ContainerType& u)
{
    blas1::axpbypgz( dt_*b[0], f_[0], dt_*b[1], f_[1], dt_*b[2], f_[2]);
    blas1::axpbypgz( a[0], u_[0], a[1], u_[1], a[2], u_[2]);
    //permute f_[2], u_[2]  to be the new f_[0], u_[0]
    for( unsigned i=2; i>0; i--)
    {
        f_[i-1].swap( f_[i]);
        u_[i-1].swap( u_[i]);
    }
    blas1::axpby( 1., f_[0], 1., u_[0]);
    //compute implicit part
    real_type alpha[2] = {2., -1.};
    //real_type alpha[2] = {1., 0.};
    blas1::axpby( alpha[0], u_[1], alpha[1],  u_[2], u); //extrapolate previous solutions
    t = t_ = t_+ dt_;
    m_solver.solve( -dt_*g0, diff, t, u, u_[0]);
    blas1::copy( u, u_[0]); //store result
    f(t_, u_[0], f_[0]); //call f on new point
}
///@endcond



} //namespace dg
