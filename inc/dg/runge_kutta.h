#ifndef _DG_RK_
#define _DG_RK_

#include <cassert>
#include <vector>

#include "exceptions.h"
#include "blas1.h"


/*! @file

  This file contains runge-kutta explicit time-integrators
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
    static const double alpha[k][k];  //!< alpha
    static const double beta[k]; //!< beta
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
*
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
    for( unsigned i=1; i<k-1; i++)
    {
        f( u_[i-1], u_[i]);
        blas1::axpby( rk_coeff<k>::alpha[i][0], u0, dt*rk_coeff<k>::beta[i], u_[i]);
        for( unsigned l=1; l<=i; l++)
        {
            blas1::axpby( rk_coeff<k>::alpha[i][l], u_[l-1],1., u_[i]); //Fall alpha = 0 muss axpby abfangen!!
        }

    }
    //Now add everything up to u1
    f( u_[k-2], u1);
    blas1::axpby( rk_coeff<k>::alpha[k-1][0], u0, dt*rk_coeff<k>::beta[k-1], u1);
    for( unsigned l=1; l<=k-1; l++)
    {
        blas1::axpby( rk_coeff<k>::alpha[k-1][l], u_[l-1],1., u1);
    }
}

struct NotANumber : public std::exception
{
    /**
     * @brief Construct 
     *
     */
    NotANumber( ) {}
    /**
     * @brief What string
     *
     * @return string "NaN returned!"
     */
    char const* what() const throw(){ return "NaN returned!";}
};

/**
 * @brief Integrates the differential equation using RK4 and a rudimentary stepsize-control
 *
 * @ingroup algorithms
 * Doubles the number of timesteps until the desired accuracy is reached
 * @tparam RHS The right-hand side class
 * @tparam Vector Vector-class (needs to be copyable)
 * @param rhs The right-hand-side
 * @param begin initial condition
 * @param end (write-only) contains solution on output
 * @param T_max final time
 * @param eps_abs desired absolute accuracy
 */
template< class RHS, class Vector>
void integrateRK4(RHS& rhs, const Vector& begin, Vector& end, double T_max, double eps_abs )
{
    RK<4, Vector > rk( begin); 
    Vector old_end(begin), temp(begin),diffm(begin);
    end = begin;
    if( T_max == 0) return;
    double dt = T_max/1;
    unsigned NT = 1;
    double error = 1e10;
    bool flag = false; 
 
    while( error > eps_abs && NT < pow( 2, 18) )
    {
        dt /= 2.;
        NT *= 2;
        end = begin;

        int i=0;
        while (i<NT && NT < pow( 2, 18))
        {
            rk( rhs, end, temp, dt); 
            end.swap( temp); //end is one step further 
            dg::blas1::axpby( 1., end, -1., old_end,diffm); //abs error=oldend = end-oldend
            error = sqrt( dg::blas1::dot( diffm, diffm));
            if ( isnan(end[0]) || isnan(end[1]) || isnan(end[2])        ) 
            {
                dt /= 2.;
                NT *= 2;
                i=-1;
                end = begin;
                #ifdef DG_DEBUG
                    std::cout << "---------Got NaN -> choosing smaller step size and redo integration" << " NT "<<NT<<" dt "<<dt<< std::endl;
                #endif
            }
            //if new integrated point outside domain
            if ((1e-5 > end[0]  ) || (1e10 < end[0])  ||(-1e10  > end[1]  ) || (1e10 < end[1])||(-1e10 > end[2]  ) || (1e10 < end[2])  )
            {
                error = eps_abs/10;
                #ifdef DG_DEBUG
                    std::cout << "---------Point outside box -> stop integration" << std::endl; 
                #endif
                i=NT;
            }
            i++;
        }  


        old_end = end;
#ifdef DG_DEBUG
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        std::cout << "NT "<<NT<<" dt "<<dt<<" error "<<error<<"\n";
#endif //DG_DEBUG
    }

    if( isnan(error) )
    {
        std::cerr << "ATTENTION: Choose more parallel planes for convergence! "<<std::endl;
        throw NotANumber();
    }
    if( error > eps_abs )
    {
        std::cerr << "ATTENTION: error is "<<error<<std::endl;
        throw Fail( eps_abs);
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
///@endcond





} //namespace dg

#endif //_DG_RK_
