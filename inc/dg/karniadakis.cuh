#pragma once

#include "cg.cuh"


namespace dg{
namespace detail{

template< class LinearOp>
struct Implicit
{
    Implicit( double alpha, LinearOp& f): f_(f), alpha_(alpha){}
    template<class container>
    void symv( const container& x, container& y) const
    {
        if( alpha_ != 0);
            f_( x,y);
        blas1::axpby( 1., x, alpha_, y);
        blas1::pointwiseDot( f_.weights(), y,  y);
    }
    //compute without weights
    template<class container>
    void operator()( const container& x, container& y) 
    {
        f_( x,y);
        blas1::axpby( 1., x, alpha_, y, y);
    }
    double& alpha( ){  return alpha_;}
    double alpha( ) const  {return alpha_;}
  private:
    LinearOp& f_;
    double alpha_;

};

}//namespace detail
///@cond
template< class M>
struct MatrixTraits< detail::Implicit<M> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

template<class Vector>
struct Karniadakis
{
    /**
    * @brief Reserve memory for the integration
    *
    * @param copyable Vector of size which is used in integration. 
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
     * @brief Init with initial value
     *
     * This routine initiates the first steps in the multistep method by integrating
     * backwards with a Euler steps. This routine has to be called
     * before the first timestep is made and with the same initial value as the first timestep.
     * @tparam Functor models BinaryFunction with no return type (subroutine).
        Its arguments both have to be of type Vector.
        The first argument is the actual argument, the second contains
        the return value, i.e. y' = f(y) translates to f( y, y').
     * @param f The rhs functor
     * @param u0 The initial value you later use 
     * @param dt The timestep
     */
    template< class Functor, class LinearOp>
    void init( Functor& f, LinearOp& diff, const Vector& u0, double dt);
    /**
    * @brief Advance u0 one timestep
    *
    * @tparam Functor models BinaryFunction with no return type (subroutine)
        Its arguments both have to be of type Vector.
        The first argument is the actual argument, The second contains
        the return value, i.e. y' = f(y) translates to f( y, y').
    * @param f right hand side function or functor
    * @param u0 initial value
    * @param u1 contains result on output. u0 and u1 may be the same ( if the Functor allows that)
    * @param dt The timestep.
    * @note The fist u0 must be the same you use in the init routine.
    */
    template< class Functor, class LinearOp>
    void operator()( Functor& f, LinearOp& diff, Vector& u);
  private:
    std::vector<Vector> u_, f_; 
    CG< Vector> pcg;
    double eps_;
    double dt_;
    double a[3];
    double b[3];

};
template< class Vector>
template< class Functor, class Diffusion>
void Karniadakis<Vector>::init( Functor& f, Diffusion& diff,  const Vector& u0,  double dt)
{
    dt_ = dt;
    detail::Implicit<Diffusion> implicit( -dt, diff);
    u_[0] = u0;
    f( u_[0], f_[0]);
    blas1::axpby( 1.,u_[0], -dt, f_[0], f_[1]);
    implicit( f_[1], u_[1]);
    f( u_[1], f_[1]);
    blas1::axpby( 1.,u_[1], -dt, f_[1], f_[2]);
    implicit( f_[2], u_[2]);
    f( u_[2], f_[2]);
}

template<class Vector>
template< class Functor, class Diffusion>
void Karniadakis<Vector>::operator()( Functor& f, Diffusion& diff, Vector& u)
{
    //u_[0] can be deleted
    detail::Implicit<Diffusion> implicit( -dt_/11.*6., diff);
    f( u_[0], f_[0]);
    blas1::axpby( a[0], u_[0], dt_*b[0], f_[0], u);
    blas1::axpby( a[1], u_[1], 1., u);
    blas1::axpby( a[2], u_[2], 1., u);
    for( unsigned i=1; i<3; i++)
        blas1::axpby( dt_*b[i], f_[i], 1., u);
    //permute f_[2], u_[2]  to be the new f_[0], u_[0]
    for( unsigned i=2; i>0; i--)
    {
        f_[i-1].swap( f_[i]);
        u_[i-1].swap( u_[i]);
    }
    //compute implicit part
    blas1::axpby( 2., u_[1], -1.,  u_[2], u_[0]); //extrapolate previous solutions
    blas2::symv( diff.weights(), u, u);
#ifdef DG_BENCHMARK
    unsigned number = pcg( implicit, u_[0], u, diff.precond(), eps_);
    std::cout << " # of pcg iterations for timestep: "<<number<<"/"<<pcg.get_max()<<"\n";
#else
    pcg( implicit, u_[0], u, diff.precond(), eps_);
#endif
    u = u_[0];


}

} //namespace dg
