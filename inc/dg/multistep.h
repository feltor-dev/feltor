#pragma once

#include "cg.h"


/*! @file

  This file contains multistep explicit& implicit time-integrators
  */
namespace dg{

///@cond
template< size_t k>
struct ab_coeff
{
    static const double b[k];
};
template<>
const double ab_coeff<2>::b[2] = {1.5, -0.5};
template<>
const double ab_coeff<3>::b[3] = {23./12., -4./3., 5./12.};
template<>
const double ab_coeff<4>::b[4] = {55./24., -59./24., 37./24., -3./8.};
template<>
const double ab_coeff<5>::b[5] = {1901./720., -1387./360., 109./30., -637./360., 251/720};
///@endcond

/**
* @brief Struct for Adams-Bashforth explicit multistep time-integration
*
* @ingroup algorithms
*
* Computes \f[ u_{n+1} = u_n + dt\sum_{j=0}^k b_j f(u_{n-j}) \f]
* Uses only blas1::axpby routines to integrate one step
* and only one right-hand-side evaluation per step.
* @tparam k Order of the method (Currently one of 1, 2, 3, 4 or 5)
* @tparam Vector The Argument type used in the Functor class
*/
template< size_t k, class Vector>
struct AB
{
    /**
    * @brief Reserve memory for the integration
    *
    * @param copyable Vector of size which is used in integration. 
    * A Vector object must be copy-constructible from copyable.
    */
    AB( const Vector& copyable): f_(k, Vector(copyable)), u_(copyable){ }
   
    /**
     * @brief Init with initial value
     *
     * This routine initiates the first steps in the multistep method by integrating
     * backwards with a Euler method. This routine has to be called
     * before the first timestep is made and with the same initial value as the first timestep.
     * @tparam Functor models BinaryFunction with no return type (subroutine).
        Its arguments both have to be of type Vector.
        The first argument is the actual argument, the second contains
        the return value, i.e. y' = f(y) translates to f( y, y').
     * @param f The rhs functor
     * @param u0 The initial value of the integration
     * @param dt The timestep
     */
    template< class Functor>
    void init( Functor& f, const Vector& u0, double dt);
    /**
    * @brief Advance u0 one timestep
    *
    * @tparam Functor models BinaryFunction with no return type (subroutine)
        Its arguments both have to be of type Vector.
        The first argument is the actual argument, The second contains
        the return value, i.e. y' = f(y) translates to f( y, y').
    * @param f right hand side function or functor
    * @param u (write-only) contains next step of the integration on output
    */
    template< class Functor>
    void operator()( Functor& f, Vector& u);
  private:
    double dt_;
    std::vector<Vector> f_; //TODO std::array is more natural here (but unfortunately not available)
    Vector u_;
};

template< size_t k, class Vector>
template< class Functor>
void AB<k, Vector>::init( Functor& f, const Vector& u0,  double dt)
{
    dt_ = dt;
    Vector u1(u0), u2(u0);
    blas1::axpby( 1., u0, 0, u_);
    f( u1, f_[0]);
    for( unsigned i=1; i<k; i++)
    {
        blas1::axpby( 1., u2, -dt, f_[i-1], u1);
        blas1::axpby( 1., u1, 0, u2); //f may destroy u1
        f( u1, f_[i]);
    }
}

template< size_t k, class Vector>
template< class Functor>
void AB<k, Vector>::operator()( Functor& f, Vector& u)
{
    blas1::axpby( 1., u_, 0, u);
    f( u, f_[0]);
    for( unsigned i=0; i<k; i++)
        blas1::axpby( dt_*ab_coeff<k>::b[i], f_[i], 1., u_);
    //permute f_[k-1]  to be the new f_[0]
    for( unsigned i=k-1; i>0; i--)
        f_[i-1].swap( f_[i]);
    blas1::axpby( 1., u_, 0, u);
}

///@cond
//Euler specialisation
template < class Vector>
struct AB<1, Vector>
{
    AB(){}
    AB( const Vector& copyable):temp_(2, copyable){}
    template < class Functor>
    void init( Functor& f, const Vector& u0, double dt){ dt_=dt;}
    template < class Functor>
    void operator()( Functor& f, Vector& u)
    {
        blas1::axpby( 1., u, 0, temp_[0]);
        f( u, temp_[1]);
        blas1::axpby( 1., temp_[0], dt_, temp_[1], u);
    }
    private:
    double dt_;
    std::vector<Vector> temp_;
};
///@endcond
///@cond
namespace detail{

template< class LinearOp, class container>
struct Implicit
{
    Implicit( double alpha, LinearOp& f, container& reference): f_(f), alpha_(alpha), temp_(reference){}
    void symv( const container& x, container& y) 
    {
        blas1::axpby( 1., x, 0, temp_);//f_ might destroy x
        if( alpha_ != 0);
            f_( temp_,y);
        blas1::axpby( 1., x, alpha_, y, y);
        blas2::symv( f_.weights(), y,  y);
    }
    //compute without weights
    void operator()( const container& x, container& y) 
    {
        blas1::axpby( 1., x, 0, temp_);
        if( alpha_ != 0);
            f_( temp_,y);
        blas1::axpby( 1., x, alpha_, y, y);
    }
    double& alpha( ){  return alpha_;}
    double alpha( ) const  {return alpha_;}
  private:
    LinearOp& f_;
    double alpha_;
    container& temp_;

};

}//namespace detail
template< class M, class V>
struct MatrixTraits< detail::Implicit<M, V> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

/**
* @brief Struct for Karniadakis semi-implicit multistep time-integration
* \f[
    {\bar v}^n = \frac{1}{\gamma_0}\left(\sum_{q=0}^2 \alpha_q v^{n-q} + \Delta t\sum_{q=0}^2\beta_q  N( v^{n-q})\right) \f]
    \f[
    \left( 1  - \frac{\Delta t}{\gamma_0}  \hat L\right)  v^{n+1} = {\bar v}^n  
    \f]
* @ingroup algorithms
* Uses blas1::axpby routines to integrate one step
* and only one right-hand-side evaluation per step. 
* Uses a conjugate gradient method for the implicit operator  
* @tparam Vector The Argument type used in the Functor class
*/
template<class Vector>
struct Karniadakis
{

    /**
    * @brief Reserve memory for the integration
    *
    * @param copyable Vector of size which is used in integration. 
    * @param max_iter parameter for cg
    * @param eps  parameter for cg
    * A Vector object must be copy-constructible from copyable.
    */
    Karniadakis( const Vector& copyable, unsigned max_iter, double eps): u_(3, Vector(copyable)), f_(3, Vector(copyable)), pcg( copyable, max_iter), eps_(eps){
        //a[0] =  1.908535476882378;  b[0] =  1.502575553858997;
        //a[1] = -1.334951446162515;  b[1] = -1.654746338401493;
        //a[2] =  0.426415969280137;  b[2] =  0.670051276940255;
        a[0] =  18./11.;    b[0] =  18./11.;
        a[1] = -9./11.;     b[1] = -18./11.;
        a[2] = 2./11.;      b[2] = 6./11.;   //Karniadakis !!!
    }
   
    /**
     * @brief Initialize with initial value
     *
     * @tparam Functor models BinaryFunction with no return type (subroutine)
        Its arguments both have to be of type Vector.
        The first argument is the actual argument, The second contains
        the return value, i.e. y' = f(y) translates to f( y, y').
     * @tparam LinearOp models BinaryFunction with no return type (subroutine)
        Its arguments both have to be of type Vector.
        The first argument is the actual argument, The second contains
        the return value, i.e. y' = L(y) translates to diff( y, y').
        Furthermore the routines weights() and precond() must be callable
        and return diagonal weights and the preconditioner for the conjugate gradient. 
     * @param f right hand side function or functor
     * @param diff diffusion operator treated implicitely 
     * @param u0 The initial value you later use 
     * @param dt The timestep saved for later use
     */
    template< class Functor, class LinearOp>
    void init( Functor& f, LinearOp& diff, const Vector& u0, double dt);

    /**
    * @brief Advance u for one timestep
    *
    * @tparam Functor models BinaryFunction with no return type (subroutine)
        Its arguments both have to be of type Vector.
        The first argument is the actual argument, The second contains
        the return value, i.e. y' = f(y) translates to f( y, y').
    * @tparam LinearOp models BinaryFunction with no return type (subroutine)
        Its arguments both have to be of type Vector.
        The first argument is the actual argument, The second contains
        the return value, i.e. y' = L(y) translates to diff( y, y').
        Furthermore the routines weights() and precond() must be callable
        and return diagonal weights and the preconditioner for the conjugate gradient. 
    * @param f right hand side function or functor (is called for u)
    * @param diff diffusion operator treated implicitely 
    * @param u (write-only), contains next step of time-integration on output
    */
    template< class Functor, class LinearOp>
    void operator()( Functor& f, LinearOp& diff, Vector& u);


    /**
     * @brief return the current head of the computation
     *
     * @return current head
     */
    const Vector& head()const{return u_[0];}
    /**
     * @brief return the last vector for which f was called
     *
     * @return current head^
     */
    const Vector& last()const{return u_[1];}
  private:
    std::vector<Vector> u_, f_; 
    CG< Vector> pcg;
    double eps_;
    double dt_;
    double a[3];
    double b[3];

};

///@cond
template< class Vector>
template< class Functor, class Diffusion>
void Karniadakis<Vector>::init( Functor& f, Diffusion& diff,  const Vector& u0,  double dt)
{
    dt_ = dt;
    Vector temp_(u0);
    detail::Implicit<Diffusion, Vector> implicit( -dt, diff, temp_);
    blas1::axpby( 1., u0, 0, temp_); //copy u0
    f( temp_, f_[0]);
    blas1::axpby( 1., u0, 0, u_[0]); 
    blas1::axpby( 1., u_[0], -dt, f_[0], f_[1]); //Euler step
    implicit( f_[1], u_[1]); //explicit Euler step backwards, might destroy f_[1]
    blas1::axpby( 1., u_[1], 0, temp_); 
    f( temp_, f_[1]);
    blas1::axpby( 1.,u_[1], -dt, f_[1], f_[2]);
    implicit( f_[2], u_[2]);
    blas1::axpby( 1., u_[2], 0, temp_); 
    f( temp_, f_[2]);
}

template<class Vector>
template< class Functor, class Diffusion>
void Karniadakis<Vector>::operator()( Functor& f, Diffusion& diff, Vector& u)
{

    blas1::axpby( 1., u_[0], 0, u); //save u_[0]
    f( u, f_[0]);
    blas1::axpby( dt_*b[1], f_[1], dt_*b[2], f_[2], f_[2]);
    blas1::axpby( dt_*b[0], f_[0],       1., f_[2], f_[2]);
    blas1::axpby( a[1], u_[1], a[2], u_[2], u_[2]);
    blas1::axpby( a[0], u_[0],   1., u_[2], u_[2]);
    blas1::axpby( 1., u_[2], 1., f_[2], u);
    //permute f_[2], u_[2]  to be the new f_[0], u_[0]
    for( unsigned i=2; i>0; i--)
    {
        f_[i-1].swap( f_[i]);
        u_[i-1].swap( u_[i]);
    }
    //compute implicit part
    double alpha[2] = {2., -1.};
    //double alpha[2] = {1., 0.};
    blas1::axpby( alpha[0], u_[1], -alpha[1],  u_[2], u_[0]); //extrapolate previous solutions
    blas2::symv( diff.weights(), u, u);
    detail::Implicit<Diffusion, Vector> implicit( -dt_/11.*6., diff, f_[0]);
#ifdef DG_BENCHMARK
    Timer t;
    t.tic(); 
    unsigned number = pcg( implicit, u_[0], u, diff.precond(), eps_);
    t.toc();
    std::cout << "# of pcg iterations for timestep: "<<number<<"/"<<pcg.get_max()<<" took "<<t.diff()<<"s\n";
#else
    pcg( implicit, u_[0], u, diff.precond(), eps_);
#endif
    blas1::axpby( 1., u_[0], 0, u); //save u_[0]


}
///@endcond

} //namespace dg
