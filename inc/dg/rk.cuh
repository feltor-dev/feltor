#ifndef _DG_RK_
#define _DG_RK_

#include <cassert>
#include <vector>

#include "blas1.h"


/*! @file

  Contrary to its name the file contains not only runge-kutta but also adams-bashforth
  explicit time-integrators
  */
namespace dg{

//namespace detail <--------- ??

/*! @brief coefficients for explicit RK methods
 *
 * To derive these coefficients from the butcher tableau
 * consider 
 * \f[ y = Bhk = (B-D)hk + Dhk = (B-D)B^{-1}y + Dhk 
 *       = ( 1- DB^{-1})y + Dhk = \alpha y + \beta h k\f]
 *  where \f[ B\f] is the butcher tableau of order k+1 and \f[ D\f] its
 *  diagonal part. 
 * @tparam k Order of the method. Currently 2,3 and 4 are available
 */
template< size_t k>
struct rk_coeff
{
    static const double alpha[k][k]; 
    static const double beta[k];
};
///@cond
/*
template<>
const double rk_coeff<1>::alpha[1][1] = { {1}};
template<>
const double rk_coeff<1>::beta[1] = {1};
*/

//from Cockburn paper
template<>
const double rk_coeff<2>::alpha[2][2] = {
    { 1,   0},
    { 0.5, 0.5}
};
template<>
const double rk_coeff<2>::beta[2] = {
     1, 0.5
};
//from Cockburn paper
template<>
const double rk_coeff<3>::alpha[3][3] = {
    { 1,     0,    0},
    { 0.75,  0.25, 0},
    { 1./3., 0.,   2./3.}
};
template<>
const double rk_coeff<3>::beta[3] = {
     1, 0.25, 2./3.
};
//classic RK4 coefficients (matlab used to compute from normal form)
template<>
const double rk_coeff<4>::alpha[4][4] = {
    { 1,    0., 0, 0 },
    { 1.,   0., 0, 0 },
    { 1.,   0., 0, 0 },
    {-1./3., 1./3., 2./3., 1./3.}
};
template<>
const double rk_coeff<4>::beta[4] = {
     0.5, 0.5, 1.0, 1./6.
};
///@endcond
//RHS contains Information about Vector type it uses
//k is the order of the method
// Vector f( const Vector& v)
// Vector should probably be rvalue assignable

/**
* @brief Struct for Runge-Kutta explicit time-integration
*
* @ingroup algorithms
* Uses only blas1::axpby routines to integrate one step.
* The coefficients are chosen in a form that require a minimum of 
* axpby function calls (check for alpha==0, beta==1) and else 
* assumes that most of the work is done in the computation of the rhs.
* @tparam k Order of the method (1, 2, 3 or 4)
* @tparam Vector The argument type used in the Functor class
*/
template< size_t k, class Vector>
struct RK
{
    /**
    * @brief Reserve memory for the integration
    *
    * @param copyable Vector of size which is used in integration. 
    * A Vector object must be copy-constructible from copyable.
    */
    RK( const Vector& copyable): u_(k-1, Vector(copyable)){ }
    /**
    * @brief Advance u0 one timestep
    *
    * @tparam Functor models BinaryFunction with no return type (subroutine)
        Its arguments both have to be of type Vector.
        The first argument is the actual argument, The second contains
        the return value, i.e. y' = f(y) translates to f( y, y').
    * @param f right hand side function
    * @param u0 initial value
    * @param u1 contains result on output. u0 and u1 may currently not be the same.
    * @param dt The timestep.
    */
    template< class Functor>
    void operator()( Functor& f, const Vector& u0, Vector& u1, double dt);
  private:
    std::vector<Vector> u_; //TODO std::array is more natural here (but unfortunately not available)
};

//u0 and u1 may not be the same vector
//TO DO: this might be cured by adding u0 first to u_[0] in the last step
//f( y, yp) where y is const and yp contains the result
template< size_t k, class Vector>
template< class Functor>
void RK<k, Vector>::operator()( Functor& f, const Vector& u0, Vector& u1, double dt)
{
    assert( &u0 != &u1);
    f(u0, u_[0]);
    blas1::axpby( rk_coeff<k>::alpha[0][0], u0, dt*rk_coeff<k>::beta[0], u_[0]);
    //cudaThreadSynchronize();
    for( unsigned i=1; i<k-1; i++)
    {
        f( u_[i-1], u_[i]);
        blas1::axpby( rk_coeff<k>::alpha[i][0], u0, dt*rk_coeff<k>::beta[i], u_[i]);
        //cudaThreadSynchronize();
        for( unsigned l=1; l<=i; l++)
        {
            blas1::axpby( rk_coeff<k>::alpha[i][l], u_[l-1],1., u_[i]); //Fall alpha = 0 muss axpby abfangen!!
            //cudaThreadSynchronize();
        }

    }
    //Now add everything up to u1
    f( u_[k-2], u1);
    blas1::axpby( rk_coeff<k>::alpha[k-1][0], u0, dt*rk_coeff<k>::beta[k-1], u1);
    //cudaThreadSynchronize();
    for( unsigned l=1; l<=k-1; l++)
    {
        blas1::axpby( rk_coeff<k>::alpha[k-1][l], u_[l-1],1., u1);
        //cudaThreadSynchronize();
    }
}

///@cond
//Euler specialisation
template < class Vector>
struct RK<1, Vector>
{
    RK(){}
    RK( const Vector& copyable){}
    template < class Functor>
    void operator()( Functor& f, const Vector& u0, Vector& u1, double dt)
    {
        f( u0, u1);
        blas1::axpby( 1., u0, dt, u1);
    }
};


template< size_t k>
struct ab_coeff
{
    static const double b[k];
};
template<>
const double ab_coeff<2>::b[2] = {1.5, -0.5};
template<>
const double ab_coeff<3>::b[3] = {23./12., -16./12., 5./12.};
///@endcond

/**
* @brief Struct for Adams-Bashforth explicit multistep time-integration
*
* @ingroup algorithms
* Uses only blas1::axpby routines to integrate one step
* and only one right-hand-side evaluation per step.
* @tparam k Order of the method (1, 2, or 3)
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
    AB( const Vector& copyable): u_(k, Vector(copyable)){ }
   
    /**
     * @brief Init with initial value
     *
     * This routine initiates the first steps in the multistep method by integrating
     * backwards with a runge-kutta method of same order. This routine has to be called
     * before the first timestep is made and with the same initial value as the first timestep.
     * @tparam Functor models BinaryFunction with no return type (subroutine).
        Its arguments both have to be of type Vector.
        The first argument is the actual argument, the second contains
        the return value, i.e. y' = f(y) translates to f( y, y').
     * @param f The rhs functor
     * @param u0 The initial value you later use 
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
    * @param u0 initial value
    * @param u1 contains result on output. u0 and u1 may be the same ( if the Functor allows that)
    * @param dt The timestep.
    * @note The fist u0 must be the same you use in the init routine.
    */
    template< class Functor>
    void operator()( Functor& f, const Vector& u0, Vector& u1, double dt);
  private:
    std::vector<Vector> u_; //TODO std::array is more natural here (but unfortunately not available)
};

//compute two steps backwards with same order RK scheme 
template< size_t k, class Vector>
template< class Functor>
void AB<k, Vector>::init( Functor& f, const Vector& u0,  double dt)
{
    Vector u1(u0);
    f( u0, u_[0]);
    for( unsigned i=1; i<k; i++)
    {
        blas1::axpby( 1., u1, -dt, u_[i-1], u1);
        f( u1, u_[i]);
    }
}

//u0 and u1 can be the same
template< size_t k, class Vector>
template< class Functor>
void AB<k, Vector>::operator()( Functor& f, const Vector& u0, Vector& u1, double dt)
{
    //u_[0] can be deleted
    f( u0, u_[0]);
    blas1::axpby( dt*ab_coeff<k>::b[0], u_[0], 1., u0, u1);
    for( unsigned i=1; i<k; i++)
        blas1::axpby( dt*ab_coeff<k>::b[i], u_[i], 1., u1);
    //permute u_[k-1]  to be the new u_[0]
    for( unsigned i=k-1; i>0; i--)
        u_[i-1].swap( u_[i]);
}

///@cond
//Euler specialisation
template < class Vector>
struct AB<1, Vector>
{
    AB(){}
    AB( const Vector& copyable){}
    template < class Functor>
    void init( Functor& f, const Vector& u0, double dt){}
    template < class Functor>
    void operator()( Functor& f, const Vector& u0, Vector& u1, double dt)
    {
        f( u0, u1);
        blas1::axpby( 1., u0, dt, u1);
    }
};
///@endcond


template<class Vector>
struct TVB
{
    /**
    * @brief Reserve memory for the integration
    *
    * @param copyable Vector of size which is used in integration. 
    * A Vector object must be copy-constructible from copyable.
    */
    TVB( const Vector& copyable): u_(2, Vector(copyable)), f_(3, Vector(copyable)){
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
    * @param u0 initial value
    * @param u1 contains result on output. u0 and u1 may be the same ( if the Functor allows that)
    * @param dt The timestep.
    * @note The fist u0 must be the same you use in the init routine.
    */
    template< class Functor>
    void operator()( Functor& f, const Vector& u0, Vector& u1, double dt);
  private:
    std::vector<Vector> u_, f_; 
    double a[3];
    double b[3];

};
template< class Vector>
template< class Functor>
void TVB<Vector>::init( Functor& f, const Vector& u0,  double dt)
{
    f( u0, f_[0]);
    blas1::axpby( 1.,u0, -dt, f_[0], u_[0]);
    f( u_[0], f_[1]);
    blas1::axpby( 1.,u_[0], -dt, f_[1], u_[1]);
    f( u_[1], f_[2]);
}

template<class Vector>
template< class Functor>
void TVB<Vector>::operator()( Functor& f, const Vector& u0, Vector& u1, double dt)
{
    //u_[0] can be deleted
    f( u0, f_[0]);
    blas1::axpby( a[0], u0, dt*b[0], f_[0], u1);
    blas1::axpby( a[1], u_[0], 1., u1);
    blas1::axpby( a[2], u_[1], 1., u1);
    for( unsigned i=1; i<3; i++)
        blas1::axpby( dt*b[i], f_[i], 1., u1);
    //permute f_[k-1]  to be the new f_[0]
    for( unsigned i=2; i>0; i--)
        f_[i-1].swap( f_[i]);
    //throw away u_[1]
    u_[1] = u_[0];
    u_[0] = u0;

}

} //namespace dg

#endif //_DG_RK_
