#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/algorithm.h"
#include "modified.h"
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
 * @brief A polynomial approximation type flux function
 */
namespace polynomial
{
///@addtogroup polynomial
///@{

/**
 * @brief \f$ \psi_p(R,Z) =
      R_0P_{\psi}\Bigg\{ \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} c_{iN+j} \bar R^i \bar Z^j \Bigg\}
      \f$

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

inline dg::geo::CylindricalFunctorsLvl2 createPsip( Parameters gp)
{
    return CylindricalFunctorsLvl2( Psip(gp), PsipR(gp), PsipZ(gp),
        PsipRR(gp), PsipRZ(gp), PsipZZ(gp));
}
inline dg::geo::CylindricalFunctorsLvl1 createIpol( Parameters gp)
{
    return CylindricalFunctorsLvl1( Constant( gp.pi), Constant(0), Constant(0));
}

///@}

} //namespace polynomial

/**
 * @brief Create a Polynomial Magnetic field
 *
 * \f[ \psi_p(R,Z) =
      R_0P_{\psi}\Bigg\{ \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} c_{iN+j} \bar R^i \bar Z^j \Bigg\}
  \f]
  \f[
  I = P_I
  \f]
  with \f$ \bar R := \frac{ R}{R_0} \f$ and \f$\bar Z := \frac{Z}{R_0}\f$
 *
 * Based on \c dg::geo::polynomial::Psip(gp) and  \c dg::Constant(gp.pi)
 * @param gp Polynomial parameters
 * @return A magnetic field object
 * @ingroup polynomial
 */
inline dg::geo::TokamakMagneticField createPolynomialField(
    dg::geo::polynomial::Parameters gp)
{
    MagneticFieldParameters params( gp.a, gp.elongation, gp.triangularity,
            equilibrium::polynomial, modifier::none, str2description.at( gp.description));
    return TokamakMagneticField( gp.R_0, polynomial::createPsip(gp),
        polynomial::createIpol(gp), params);
}

} //namespace geo
} //namespace dg

