#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "dg/topology/functions.h"
#include "dg/functors.h"
#include "polynomial_parameters.h"
#include "magnetic_field.h"


/*!@file
 *
 * MagneticField objects
 */
namespace dg
{
namespace geo
{

/**
 * @brief \f[ \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} c_{i*N+j} x^i y^j  \f]
 */
struct Horner2d
{
    Horner2d(): m_c( 1, 1), m_M(1), m_N(1){}
    Horner2d( std::vector<double> c, unsigned M, unsigned N): m_c(c), m_M(M), m_N(N){}
    double operator()( double x, double y) const
    {
        std::vector<double> cx( m_M);
        for( unsigned i=0; i<m_M; i++)
            cx[i] = horner( &m_c[i*m_N], m_N, y);
        return horner( &cx[0], m_M, x);
    }
    private:
    double horner( const double * c, unsigned M, double x) const
    {
        double b = c[M-1];
        for( unsigned i=0; i<M-1; i++)
            b = c[M-2-i] + b*x;
        return b;
    }
    std::vector<double> m_c;
    unsigned m_M, m_N;
};

/**
 * @brief Contains a polynomial approximation type flux function
 */
namespace polynomial
{
///@addtogroup polynomial
///@{

/**
 * @brief \f[ \hat{\psi}_p  \f]
 *
 * \f[ \hat{\psi}_p(R,Z) =
      \hat{R}_0P_{\psi}\Bigg\{ \sum_i \sum_j c_{i*N+j} x^i y^j \Bigg\}
      \Bigg\} \f]
      with \f$ \bar R := \frac{ R}{R_0} \f$ and \f$\bar Z := \frac{Z}{R_0}\f$
 *
 */
struct Psip: public aCylindricalFunctor<Psip>
{
    /**
     * @brief Construct from given geometric parameters
     *
     * @param gp geometric parameters
     */
    Psip( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp), m_horner(gp.c, gp.M, gp.N) {}
    double do_compute(double R, double Z) const
    {
        return m_R0*m_pp*m_horner( R/m_R0,Z/m_R0);
    }
  private:
    double m_R0, m_pp;
    Horner2d m_horner;
};

struct PsipR: public aCylindricalFunctor<PsipR>
{
    ///@copydoc Psip::Psip()
    PsipR( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp){
        std::vector<double>  beta ( (gp.M-1)*gp.N);
        for( unsigned i=0; i<gp.M-1; i++)
            for( unsigned j=0; j<gp.N; j++)
                beta[i*gp.N+j] = (double)(i+1)*gp.c[ ( i+1)*gp.N +j];
        m_horner = Horner2d( beta, gp.M-1, gp.N);
    }
    double do_compute(double R, double Z) const
    {
        return m_pp*m_horner( R/m_R0,Z/m_R0);
    }
  private:
    double m_R0, m_pp;
    Horner2d m_horner;
};
struct PsipRR: public aCylindricalFunctor<PsipRR>
{
    ///@copydoc Psip::Psip()
    PsipRR( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp){
        std::vector<double>  beta ( (gp.M-2)*gp.N);
        for( unsigned i=0; i<gp.M-2; i++)
            for( unsigned j=0; j<gp.N; j++)
                beta[i*gp.N+j] = (double)((i+2)*(i+1))*gp.c[ (i+2)*gp.N +j];
        m_horner = Horner2d( beta, gp.M-2, gp.N);
    }
    double do_compute(double R, double Z) const
    {
        return m_pp/m_R0*m_horner( R/m_R0,Z/m_R0);
    }
  private:
    double m_R0, m_pp;
    Horner2d m_horner;
};
struct PsipZ: public aCylindricalFunctor<PsipZ>
{
    ///@copydoc Psip::Psip()
    PsipZ( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp){
        std::vector<double>  beta ( gp.M*(gp.N-1));
        for( unsigned i=0; i<gp.M; i++)
            for( unsigned j=0; j<gp.N-1; j++)
                beta[i*(gp.N-1)+j] = (double)(j+1)*gp.c[ i*gp.N +j+1];
        m_horner = Horner2d( beta, gp.M, gp.N-1);
    }
    double do_compute(double R, double Z) const
    {
        return m_pp*m_horner( R/m_R0,Z/m_R0);
    }
  private:
    double m_R0, m_pp;
    Horner2d m_horner;
};
struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
    ///@copydoc Psip::Psip()
    PsipZZ( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp){
        std::vector<double>  beta ( gp.M*(gp.N-2));
        for( unsigned i=0; i<gp.M; i++)
            for( unsigned j=0; j<gp.N-2; j++)
                beta[i*(gp.N-2)+j] = (double)((j+2)*(j+1))*gp.c[ i*gp.N +j+2];
        m_horner = Horner2d( beta, gp.M, gp.N-2);
    }
    double do_compute(double R, double Z) const
    {
        return m_pp/m_R0*m_horner(R/m_R0,Z/m_R0);
    }
  private:
    double m_R0, m_pp;
    Horner2d m_horner;
};
struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
    ///@copydoc Psip::Psip()
    PsipRZ( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp){
        std::vector<double>  beta ( (gp.M-1)*(gp.N-1));
        for( unsigned i=0; i<gp.M-1; i++)
            for( unsigned j=0; j<gp.N-1; j++)
                beta[i*(gp.N-1)+j] = (double)((j+1)*(i+1))*gp.c[ (i+1)*gp.N +j+1];
        m_horner = Horner2d( beta, gp.M-1, gp.N-1);
    }
    double do_compute(double R, double Z) const
    {
        return m_pp/m_R0*m_horner(R/m_R0,Z/m_R0);
    }
  private:
    double m_R0, m_pp;
    Horner2d m_horner;
};

static inline dg::geo::CylindricalFunctorsLvl2 createPsip( Parameters gp)
{
    return CylindricalFunctorsLvl2( Psip(gp), PsipR(gp), PsipZ(gp),
        PsipRR(gp), PsipRZ(gp), PsipZZ(gp));
}
static inline dg::geo::CylindricalFunctorsLvl1 createIpol( Parameters gp)
{
    return CylindricalFunctorsLvl1( Constant( gp.pi), Constant(0), Constant(0));
}

///@}

///@cond
namespace mod
{

struct Psip: public aCylindricalFunctor<Psip>
{
    Psip( Parameters gp, double psi0, double alpha, double sign = -1) :
        m_ipoly( psi0, alpha, sign), m_psip(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        return m_ipoly( psip);
    }
    private:
    dg::IPolynomialHeaviside m_ipoly;
    polynomial::Psip m_psip;
};
struct PsipR: public aCylindricalFunctor<PsipR>
{
    PsipR( Parameters gp, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_psip(gp), m_psipR(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipR = m_psipR(R,Z);
        return psipR*m_poly( psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    polynomial::Psip m_psip;
    polynomial::PsipR m_psipR;
};
struct PsipZ: public aCylindricalFunctor<PsipZ>
{
    PsipZ( Parameters gp, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_psip(gp), m_psipZ(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipZ = m_psipZ(R,Z);
        return psipZ*m_poly( psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    polynomial::Psip m_psip;
    polynomial::PsipZ m_psipZ;
};

struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
    PsipZZ( Parameters gp, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(gp), m_psipZ(gp), m_psipZZ(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipZ = m_psipZ(R,Z);
        double psipZZ = m_psipZZ(R,Z);
        return psipZZ*m_poly( psip) + psipZ*psipZ*m_dpoly(psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    dg::DPolynomialHeaviside m_dpoly;
    polynomial::Psip m_psip;
    polynomial::PsipZ m_psipZ;
    polynomial::PsipZZ m_psipZZ;
};
struct PsipRR: public aCylindricalFunctor<PsipRR>
{
    PsipRR( Parameters gp, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(gp), m_psipR(gp), m_psipRR(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipR = m_psipR(R,Z);
        double psipRR = m_psipRR(R,Z);
        return psipRR*m_poly( psip) + psipR*psipR*m_dpoly(psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    dg::DPolynomialHeaviside m_dpoly;
    polynomial::Psip m_psip;
    polynomial::PsipR m_psipR;
    polynomial::PsipRR m_psipRR;
};
struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
    PsipRZ( Parameters gp, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(gp), m_psipR(gp), m_psipZ(gp), m_psipRZ(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipR = m_psipR(R,Z);
        double psipZ = m_psipZ(R,Z);
        double psipRZ = m_psipRZ(R,Z);
        return psipRZ*m_poly( psip) + psipR*psipZ*m_dpoly(psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    dg::DPolynomialHeaviside m_dpoly;
    polynomial::Psip m_psip;
    polynomial::PsipR m_psipR;
    polynomial::PsipZ m_psipZ;
    polynomial::PsipRZ m_psipRZ;
};

static inline dg::geo::CylindricalFunctorsLvl2 createPsip( Parameters gp,
    double psi0, double alpha, double sign = -1)
{
    return CylindricalFunctorsLvl2( Psip(gp, psi0, alpha, sign), PsipR(gp,
    psi0, alpha, sign), PsipZ(gp, psi0, alpha, sign), PsipRR(gp, psi0, alpha,
    sign), PsipRZ(gp, psi0, alpha, sign), PsipZZ(gp, psi0, alpha, sign));
}
static inline dg::geo::CylindricalFunctorsLvl1 createIpol( Parameters gp,
    double psi0, double alpha, double sign = -1)
{
    return CylindricalFunctorsLvl1( Constant( gp.pi), Constant( 0), Constant( 0));
}

} //namespace mod
///@endcond

///////////////////////////////////////introduce fields into polynomial namespace


} //namespace polynomial

/**
 * @brief Create a Polynomial Magnetic field
 *
 * Based on \c dg::geo::polynomial::Psip(gp) and \c dg::geo::polynomial::Ipol(gp)
 * @param gp Polynomial parameters
 * @return A magnetic field object
 * @ingroup geom
 */
static inline dg::geo::TokamakMagneticField createPolynomialField(
    dg::geo::polynomial::Parameters gp)
{
    MagneticFieldParameters params( gp.a, gp.elongation, gp.triangularity,
            equilibrium::polynomial, modifier::none, str2form.at( gp.form));
    return TokamakMagneticField( gp.R_0, polynomial::createPsip(gp),
        polynomial::createIpol(gp), params);
}
/**
 * @brief Create a modified Polynomial Magnetic field
 *
 * Based on \c dg::geo::polynomial::mod::Psip(gp) and
 * \c dg::geo::polynomial::mod::Ipol(gp)
 * We modify psi above a certain value to a constant using the
 * \c dg::IPolynomialHeaviside function (an approximation to the integrated Heaviside
 * function with width alpha), i.e. we replace psi with IPolynomialHeaviside(psi).
 * This subsequently modifies all derivatives of psi and the poloidal
 * current.
 * @param gp Polynomial parameters
 * @param psi0 boundary value where psi is modified to a constant psi0
 * @param alpha radius of the transition region where the modification acts (smaller is quicker)
 * @param sign determines which side of Psi to dampen (negative or positive, forwarded to \c dg::IPolynomialHeaviside)
 * @return A magnetic field object
 * @ingroup geom
 */
static inline dg::geo::TokamakMagneticField createModifiedPolynomialField(
    dg::geo::polynomial::Parameters gp, double psi0, double alpha, double sign = -1)
{
    MagneticFieldParameters params( gp.a, gp.elongation, gp.triangularity,
            equilibrium::polynomial, modifier::heaviside, str2form.at( gp.form));
    return TokamakMagneticField( gp.R_0, polynomial::mod::createPsip(gp, psi0,
    alpha, sign), polynomial::mod::createIpol(gp, psi0, alpha, sign), params);
}

} //namespace geo
} //namespace dg

