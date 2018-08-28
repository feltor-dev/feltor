#pragma once

#include "runge_kutta.h"
#include "cg.h"


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
    Furthermore, the routines %weights(), %inv_weights() and %precond() must be callable
    and return diagonal weights, inverse weights and the preconditioner for the conjugate gradient.
    The return type of these member functions must be useable in blas2 functions together with the ContainerType type.
 * @param exp explic part
 * @param imp implicit part ( must be linear in its second argument and symmetric up to weights)
 */

///@cond
template< size_t k,class real_type>
struct ab_coeff;
template<class real_type>
struct ab_coeff<2,real_type>{
    const real_type b[2] = {
        1.5, -0.5
    };
};
template<class real_type>
struct ab_coeff<3,real_type>{
    const real_type b[3] = {
        23./12., -4./3., 5./12.
    };
};
template<class real_type>
struct ab_coeff<4,real_type>{
    const real_type b[4] = {
        55./24., -59./24., 37./24., -3./8.
    };
};
template<class real_type>
struct ab_coeff<5,real_type>{
    const real_type b[5] = {
        1901./720., -1387./360., 109./30., -637./360., 251./720.
    };
};
///@endcond

/**
* @brief Struct for Adams-Bashforth explicit multistep time-integration
* \f[ u^{n+1} = u^n + \Delta t\sum_{j=0}^k b_j f\left(u^{n-j}\right) \f]
*
* @ingroup time
*
* Computes \f[ u_{n+1} = u_n + dt\sum_{j=0}^k b_j f(u_{n-j}) \f]
* Uses only \c blas1::axpby routines to integrate one step
* and only one right-hand-side evaluation per step.
* @tparam k Order of the method (Currently one of 1, 2, 3, 4 or 5)
* @copydoc hide_ContainerType
*/
template< size_t k, class ContainerType>
struct AB
{
    using real_type = get_value_type<ContainerType>;
    ///copydoc RK_opt::RK_opt()
    AB(){}
    ///@copydoc RK_opt::construct(const ContainerType&)
    AB( const ContainerType& copyable){ construct(copyable); }
    ///@copydoc RK_opt::construct(const ContainerType&)
    void construct(const ContainerType& copyable){
        f_.fill( copyable);
        u_ = copyable;
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
    std::array<ContainerType,k> f_;
    ContainerType u_;
    ab_coeff<k,real_type> m_ab;
};

template< size_t k, class ContainerType>
template< class RHS>
void AB<k, ContainerType>::init( RHS& f, real_type t0, const ContainerType& u0, real_type dt)
{
    tu_ = t0, dt_ = dt;
    f( t0, u0, f_[0]);
    //now do k Euler steps
    ContainerType u1(u0);
    for( unsigned i=1; i<k; i++)
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

template< size_t k, class ContainerType>
template< class RHS>
void AB<k, ContainerType>::step( RHS& f, real_type& t, ContainerType& u)
{
    for( unsigned i=0; i<k; i++)
        blas1::axpby( dt_*m_ab.b[i], f_[i], 1., u_);
    //permute f_[k-1]  to be the new f_[0]
    for( unsigned i=k-1; i>0; i--)
        f_[i-1].swap( f_[i]);
    blas1::copy( u_, u);
    t = tu_ = tu_ + dt_;
    f( tu_, u_, f_[0]); //evaluate f at new point
}

///@cond
//Euler specialisation
template < class ContainerType>
struct AB<1, ContainerType>
{
    using real_type = get_value_type<ContainerType>;
    AB(){}
    AB( const ContainerType& copyable){
        construct(copyable);
    }
    void construct(const ContainerType& copyable){
       f_  = u_ = copyable;
    }
    template < class RHS>
    void init( RHS& f, real_type t0, const ContainerType& u0, real_type dt){
        u_ = u0;
        t_ = t0, dt_=dt;
        f( t_, u_, f_);
    }
    template < class RHS>
    void step( RHS& f, real_type& t, ContainerType& u)
    {
        //this implementation calls rhs at end point
        blas1::axpby( 1., u_, dt_, f_, u); //compute new u

        u_ = u; //store new u
        t = t_ = t_ + dt_; //and time
        f( t_, u_, f_); //and update rhs
    }
    private:
    real_type t_, dt_;
    ContainerType u_, f_;
};
///@endcond
///@cond
namespace detail{

//compute: y + alpha f(y,t)
template< class LinearOp, class ContainerType>
struct Implicit
{
    using real_type = get_value_type<ContainerType>;
    Implicit( real_type alpha, real_type t, LinearOp& f): f_(f), alpha_(alpha), t_(t){}
    void symv( const ContainerType& x, ContainerType& y)
    {
        if( alpha_ != 0)
            f_(t_,x,y);
        blas1::axpby( 1., x, alpha_, y, y);
        blas2::symv( f_.weights(), y, y);
    }
    //compute without weights
    void operator()( const ContainerType& x, ContainerType& y)
    {
        if( alpha_ != 0)
            f_(t_,x,y);
        blas1::axpby( 1., x, alpha_, y, y);
    }
    real_type& alpha( ){  return alpha_;}
    real_type alpha( ) const  {return alpha_;}
    real_type& time( ){  return t_;}
    real_type time( ) const  {return t_;}
  private:
    LinearOp& f_;
    real_type alpha_;
    real_type t_;
};

}//namespace detail
template< class M, class V>
struct TensorTraits< detail::Implicit<M, V> >
{
    using value_type = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
///@endcond

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
* Uses only one evaluation of the explicit part per step.
* Uses a conjugate gradient method for the implicit operator (therefore \f$ \hat I(t,v)\f$ must be linear in \f$ v\f$).
The following code example demonstrates how to implement the method of manufactured solutions on a 2d partial differential equation with the dg library:
@snippet multistep_t.cu function
In the main function:
@snippet multistep_t.cu karniadakis
@note In our experience the implicit treatment of diffusive or hyperdiffusive
terms can significantly reduce the required number of time steps. This
far outweighs the increased computational cost of the additional matrix inversions.
* @ingroup time
* @copydoc hide_ContainerType
*/
template<class ContainerType>
struct Karniadakis
{
    using real_type = get_value_type<ContainerType>;
    ///@copydoc RK_opt::RK_opt()
    Karniadakis(){}

    ///@copydoc construct()
    Karniadakis( const ContainerType& copyable, unsigned max_iter, real_type eps){
        construct( copyable, max_iter, eps);
    }
    /**
    * @brief Reserve memory for the integration
    *
    * @param copyable ContainerType of size which is used in integration (values do not matter, the size is important).
    * @param max_iter parameter for cg
    * @param eps  accuracy parameter for cg
    */
    void construct( const ContainerType& copyable, unsigned max_iter, real_type eps){
        f_.fill(copyable), u_.fill(copyable);
        pcg.construct( copyable, max_iter);
        eps_ = eps;
        //a[0] =  1.908535476882378;  b[0] =  1.502575553858997;
        //a[1] = -1.334951446162515;  b[1] = -1.654746338401493;
        //a[2] =  0.426415969280137;  b[2] =  0.670051276940255;
        a[0] =  18./11.;    b[0] =  18./11.;
        a[1] = -9./11.;     b[1] = -18./11.;
        a[2] = 2./11.;      b[2] = 6./11.;   //Karniadakis !!!
    }

    /**
     * @brief Initialize by integrating two timesteps backward in time
     *
     * The backward integration uses the Lie operator splitting method, with explicit Euler substeps for both explicit and implicit part
     * @copydoc hide_explicit_implicit
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep saved for later use
     * @note the implementation is such that on output the last call to the explicit part \c exp is at \c (t0,u0). This might be interesting if the call to \c exp changes its state.
     */
    template< class Explicit, class Implicit>
    void init( Explicit& exp, Implicit& imp, real_type t0, const ContainerType& u0, real_type dt);

    /**
    * @brief Advance one timestep
    *
    * @copydoc hide_explicit_implicit
    * @param t (write-only), contains timestep corresponding to \c u on output
    * @param u (write-only), contains next step of time-integration on output
     * @note the implementation is such that on output the last call to the explicit part \c exp is at the new \c (t,u). This might be interesting if the call to \c exp changes its state.
    */
    template< class Explicit, class Implicit>
    void step( Explicit& exp, Implicit& imp, real_type& t, ContainerType& u);

  private:
    std::array<ContainerType,3> u_, f_;
    CG< ContainerType> pcg;
    real_type eps_;
    real_type t_, dt_;
    real_type a[3];
    real_type b[3];
};

///@cond
template< class ContainerType>
template< class RHS, class Diffusion>
void Karniadakis<ContainerType>::init( RHS& f, Diffusion& diff, real_type t0, const ContainerType& u0, real_type dt)
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

template<class ContainerType>
template< class RHS, class Diffusion>
void Karniadakis<ContainerType>::step( RHS& f, Diffusion& diff, real_type& t, ContainerType& u)
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
    blas2::symv( diff.weights(), u_[0], u_[0]);
    t = t_ = t_+ dt_;
    detail::Implicit<Diffusion, ContainerType> implicit( -dt_*6./11., t, diff);
#ifdef DG_BENCHMARK
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif//MPI
    Timer ti;
    ti.tic();
    unsigned number = pcg( implicit, u, u_[0], diff.precond(), diff.inv_weights(), eps_);
    ti.toc();
#ifdef MPI_VERSION
    if(rank==0)
#endif//MPI
    std::cout << "# of pcg iterations for timestep: "<<number<<"/"<<pcg.get_max()<<" took "<<ti.diff()<<"s\n";
#else
    pcg( implicit, u, u_[0], diff.precond(), diff.inv_weights(), eps_);
#endif //BENCHMARK
    blas1::copy( u, u_[0]); //store result
    f(t_, u_[0], f_[0]); //call f on new point
}
///@endcond


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
    void step( Explicit& exp, Implicit& imp, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt)
    {
        exp(t0, u0, f_);
        imp(t0+d[0]*dt, u0, g_);
        dg::blas1::axpby( dt, f_, dt, g_, rhs_);
        detail::Implicit<Implicit, ContainerType> implicit( -dt*d[0], t0+d[0]*dt, imp);
        implicit.alpha() = -dt*d[0];
        implicit.time()  = t0 + (d[0])*dt;
        blas2::symv( imp.weights(), rhs_, rhs_);
        pcg( implicit, k_[0], rhs_, imp.precond(), imp.inv_weights(), eps_);

        dg::blas1::axpby( 1., u0, b[1][0], k_[0], rhs_);
        exp(t0+b[1][0]*dt, rhs_, f_);
        dg::blas1::axpby( 1., u0, c[1][0], k_[0], rhs_);
        imp(t0+(c[1][0]+d[1])*dt, rhs_, g_);
        dg::blas1::axpby( dt, f_, dt, g_, rhs_);
        implicit.alpha() = -dt*d[1];
        implicit.time()  =  t0 + (c[1][0]+d[1])*dt;
        blas2::symv( imp.weights(), rhs_, rhs_);
        pcg( implicit, k_[1], rhs_, imp.precond(), imp.inv_weights(), eps_);

        dg::blas1::axpby( 1., u0, b[2][0], k_[0], rhs_);
        dg::blas1::axpby( b[2][1], k_[1], 1., rhs_);
        exp(t0 + (b[2][1]+b[2][0])*dt, rhs_, f_);
        dg::blas1::axpby( 1., u0, c[2][0], k_[0], rhs_);
        dg::blas1::axpby( c[2][1], k_[1], 1., rhs_);
        imp(t0 + (c[2][1]+c[2][0] + d[2])*dt, rhs_, g_);
        dg::blas1::axpby( dt, f_, dt, g_, rhs_);
        implicit.alpha() = -dt*d[2];
        implicit.time()  =  t0 + (c[2][1]+c[2][0] + d[2])*dt;
        blas2::symv( imp.weights(), rhs_, rhs_);
        pcg( implicit, k_[2], rhs_, imp.precond(), imp.inv_weights(), eps_);
        //sum up results
        dg::blas1::copy( u0, u1);
        dg::blas1::axpby( 1., u1, w[0], k_[0], u1);
        dg::blas1::axpbypgz( w[1], k_[1], w[2], k_[2], 1., u1);
        t1 = t0 + dt;
    }

    /**
     * @brief adapt timestep (experimental)
     *
     * Make same timestep twice, once with half timestep. The resulting error should be smaller than some given tolerance
     *
     * @copydoc hide_explicit_implicit
     * @param t0 start time
     * @param u0 start point
     * @param t1 (write only) end time (equals \c t0+dt on output, may alias t0)
     * @param u1 (write only) contains result at \c t1 on output (may alias u0)
     * @param dt (read and write) contains new recommended timestep on output
     * @param tolerance tolerable error
     * @param verbose if true writes error to \c std::cout
     */
    template <class Explicit, class Implicit>
    void adaptive_step( Explicit& exp, Implicit& imp, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type& dt, real_type tolerance, bool verbose = false)
    {
        real_type t;
        step( exp, imp, t0, u0, t, temp_, dt/2.);
        step( exp, imp, t, temp_, t, temp_, dt/2.);
        step( exp, imp, t0, u0, t1, u1, dt); //one full step
        dg::blas1::axpby( 1., u1, -1., temp_);
        real_type error = dg::blas1::dot( temp_, temp_);
        if(verbose)std::cout << "Error " << error<<"\tTime "<<t1<<"\tdt "<<dt;
        real_type dt_old = dt;
        dt = 0.9*dt_old*sqrt(tolerance/error);
        if( dt > 1.33*dt_old) dt = 1.33*dt_old;
        if( dt < 0.75*dt_old) dt = 0.75*dt_old;
        if(verbose) std::cout <<"\tnew_dt "<<dt<<"\n";
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

} //namespace dg
