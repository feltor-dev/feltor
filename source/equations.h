#ifndef _TL_EQUATIONS_
#define _TL_EQUATIONS_

#include <array>
#include "matrix.h"
#include "blueprint.h"

namespace toefl{
   
/*! @brief Yield the coefficients for the local Poisson equation
 */
class Poisson
{
  private:
    const double a_i, mu_i, tau_i;
    const double a_z, mu_z, tau_z;
  public: 
    Poisson( const Physical& phys);
    /*! @brief Compute prefactors for ne and ni in local poisson equation
     *
     * @param phi   
     *  Contains the two prefactors in the local Poisson equation:
     *  phi[0] multiplies with ne, phi[1]  with ni
     * @param laplace 
     *  The laplacian in fourier space
     */
    void operator()( std::array< double,2>& phi, const double laplace);
    /*! @brief Compute prefactors for ne, ni and nz in local poisson equation
     *
     * @param phi   
     *  Contains the three prefactors in the local Poisson equation:
     *  phi[0] multiplies with ne, phi[1]  with ni, phi[2] with nz
     * @param laplace   
     *  The laplacian in fourier space
     */
    void operator()( std::array< double,3>& phi, const double laplace);
    /*! @brief Compute Gamma_i
     *
     * @param laplace The laplacian in fourier space
     * @return Gamma_i
     */
    inline double gamma1_i( const double laplace);
    /*! @brief Compute Gamma_z
     *
     * @param laplace The laplacian in fourier space
     * @return Gamma_z
     */
    inline double gamma1_z( const double laplace);
    /*! @brief Compute Gamma2_i
     *
     * @param laplace The laplacian in fourier space
     * @return Gamma2_i
     */
    inline double gamma2_i( const double laplace);
    inline double gamma2_z( const double laplace);
};
Poisson::Poisson(const Physical& phys):a_i(phys.a[0]), mu_i(1.0), tau_i(phys.tau[0]),
                                      a_z(phys.a[1]), mu_z(phys.mu[1]), tau_z(phys.tau[1]) 
                                      {}
/*! @brief Yield the linear part of the local toefl equations
 *
 * \attention
 * The sine functions are not eigenfunctions of simple derivatives like e.g. dx!!
 */
class Equations
{
  private:
    typedef std::complex<double> complex;
    Poisson p;
    const double dd, nu;
    const double g_e, g_i, g_z;
    const double kappa_y;
    const double tau_i, tau_z;
    double laplace, rho;
    complex curv;
  public:
    Equations( const Physical& phys):
        p( phys), 
        dd(phys.d), nu(phys.nu), 
        g_e(phys.g_e), g_i(phys.g[0]), g_z(phys.g[1]),
        kappa_y(phys.kappa),
        tau_i(phys.tau[0]), tau_z(phys.tau[1])
    {}

    /*! @brief compute the linear part of the toefl equations without impurities
     *
     * @param coeff Contains the coefficients on output
     * @param dx    The value of the x-derivative in fourier space
     * @param dy    The value of the y-derivative in fourier space
     * \note This way you have the freedom to use various expansion functions (e.g. sine, cosine or exponential functions) 
     */
    void operator()( QuadMat< complex,2>& coeff, const complex dx, const complex dy);
    /*! @brief compute the linear part of the toefl equations with impurities
     *
     * @param coeff Contains the coefficients on output
     * @param dx    The value of the x-derivative in fourier space
     * @param dy    The value of the y-derivative in fourier space
     * \note This way you have the freedom to use various expansion functions (e.g. sine, cosine or exponential functions) 
     */
    void operator()( QuadMat< complex,3>& coeff, const complex dx, const complex dy);
};


void Equations::operator()(QuadMat< complex, 2>& c, const complex dx, const complex dy) 
{
    std::array< double,2> phi;
    laplace = (dx*dx + dy*dy).real(); 
    p( phi, laplace); //prefactors in Poisson equations (phi = phi[0]*ne + phi[1]*ni)
    curv = kappa_y*dy; //note that curv is complex
    complex P = -g_e*dy + curv + dd;
    complex Q = -g_i*dy*p.gamma1_i(laplace) + curv*( p.gamma1_i(laplace) + 0.5 *p.gamma2_i(laplace));

    c(0,0) = P*phi[0] - curv - dd + nu*laplace*laplace; c(0,1) = P*phi[1];
    c(1,0) = Q*phi[0];                      c(1,1) = Q*phi[1] + tau_i*curv + nu*laplace*laplace;
}
void Equations::operator()(QuadMat< complex, 3>& c, const complex dx, const complex dy) 
{
    std::array< double,3> phi;
    laplace = (dx*dx + dy*dy).real(); 
    p( phi, laplace);
    curv = kappa_y*dy; //note that curv is complex
    complex P = -g_e*dy + curv + dd;
    complex Q = -g_i*dy*p.gamma1_i(laplace) + curv*( p.gamma1_i(laplace) + 0.5 *p.gamma2_i(laplace));
    complex R = -g_z*dy*p.gamma1_z(laplace) + curv*( p.gamma1_z(laplace) + 0.5 *p.gamma2_z(laplace));

    c(0,0) = P*phi[0] - curv - dd + nu*laplace*laplace; c(0,1) = P*phi[1];                       c(0,2) = P*phi[2];
    c(1,0) = Q*phi[0];                      c(1,1) = Q*phi[1] + tau_i*curv + nu*laplace*laplace; c(1,2) = Q*phi[2];
    c(2,0) = R*phi[0];                      c(2,1) = R*phi[1];                       c(2,2) = R*phi[2] + tau_z*curv + nu*laplace*laplace;
}

void Poisson::operator()( std::array< double, 2>& c, const double laplace)
{
    double rho = - a_i*mu_i*laplace/(1.+ tau_i*mu_i*laplace);
    c[0] = 1./rho;
    c[1] = -a_i*gamma1_i(laplace)/rho;
}
void Poisson::operator()( std::array< double, 3>& c, const double laplace)
{
    double rho = (a_i*mu_i/(1.- tau_i*mu_i*laplace) + a_z*mu_z/(1.-tau_z*mu_z*laplace))*laplace;
    c[0] = 1./rho;
    c[1] = -a_i*gamma1_i(laplace)/rho;
    c[2] = -a_z*gamma1_z(laplace)/rho;
}
double Poisson::gamma1_i( const double laplace)
{
    return (1./(1. - 0.5*tau_i*mu_i*laplace));
}
double Poisson::gamma1_z( const double laplace)
{
    return (1./(1. - 0.5*tau_z*mu_z*laplace));
}
double Poisson::gamma2_i( const double laplace)
{
    double gamma = gamma1_i(laplace);
    return 0.5*tau_i*mu_i*laplace*gamma*gamma;
}
double Poisson::gamma2_z( const double laplace)
{
    double gamma = gamma1_z(laplace);
    return 0.5*tau_z*mu_z*laplace*gamma*gamma;
}

} //namespace toefl



#endif //_TL_EQUATIONS_
