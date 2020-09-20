#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "dg/topology/functions.h"
#include "dg/functors.h"
#include "magnetic_field.h"


/*!@file
 *
 * Modified MagneticField objects
 */
namespace dg
{
namespace geo
{
namespace mod
{

struct Psip: public aCylindricalFunctor<Psip>
{
    Psip( std::function<double(double,double)> psip, double psi0, double alpha, double sign = -1) :
        m_ipoly( psi0, alpha, sign), m_psip(psip)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        return m_ipoly( psip);
    }
    private:
    dg::IPolynomialHeaviside m_ipoly;
    std::function<double(double,double)> m_psip;
};
struct PsipR: public aCylindricalFunctor<PsipR>
{
    PsipR( std::function<double(double,double)> psip, std::function<double(double,double)> psipR, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_psip(psip), m_psipR(psipR)
    { }
    double do_compute(double R, double Z) const
    {
        double psip  = m_psip(R,Z);
        double psipR = m_psipR(R,Z);
        return psipR*m_poly( psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    std::function<double(double,double)> m_psip, m_psipR;
};
struct PsipZ: public aCylindricalFunctor<PsipZ>
{
    PsipZ( std::function<double(double,double)> psip, std::function<double(double,double)> psipZ, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_psip(psip), m_psipZ(psipZ)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipZ = m_psipZ(R,Z);
        return psipZ*m_poly( psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    std::function<double(double,double)> m_psip, m_psipZ;
};

struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
    PsipZZ( std::function<double(double,double)> psip, std::function<double(double,double)> psipZ, std::function<double(double,double)> psipZZ, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(psip), m_psipZ(psipZ), m_psipZZ(psipZZ)
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
    std::function<double(double,double)> m_psip, m_psipZ, m_psipZZ;
};

struct PsipRR: public aCylindricalFunctor<PsipRR>
{
    PsipRR( std::function<double(double,double)> psip, std::function<double(double,double)> psipR, std::function<double(double,double)> psipRR, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(psip), m_psipR(psipR), m_psipRR(psipRR)
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
    std::function<double(double,double)> m_psip, m_psipR, m_psipRR;
};
struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
    PsipRZ( std::function<double(double,double)> psip, std::function<double(double,double)> psipR, std::function<double(double,double)> psipZ, std::function<double(double,double)> psipRZ, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(psip), m_psipR(psipR), m_psipZ(psipZ), m_psipRZ(psipRZ)
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
    std::function<double(double,double)> m_psip, m_psipR, m_psipZ, m_psipRZ;
};

static inline dg::geo::CylindricalFunctorsLvl2 createPsip( const CylindricalFunctorsLvl2& psip,
    double psi0, double alpha, double sign = -1)
{
    return CylindricalFunctorsLvl2(
            mod::Psip(psip.f(), psi0, alpha, sign),
            mod::PsipR(psip.f(), psip.dfx(), psi0, alpha, sign),
            mod::PsipZ(psip.f(), psip.dfy(), psi0, alpha, sign),
            mod::PsipRR(psip.f(), psip.dfx(), psip.dfxx(), psi0, alpha, sign),
            mod::PsipRZ(psip.f(), psip.dfx(), psip.dfy(), psip.dfxy(), psi0, alpha, sign),
            mod::PsipZZ(psip.f(), psip.dfy(), psip.dfyy(), psi0, alpha, sign));
}

} //namespace mod

//Create heaviside modification, does not modify Ipol
static inline dg::geo::TokamakMagneticField createModifiedField(
    const dg::geo::TokamakMagneticField& in, double psi0, double alpha, double sign = -1)
{
    const MagneticFieldParameters inp = in.params();
    MagneticFieldParameters params = { inp.a(), inp.elongation(), inp.triangularity(),
            inp.getEquilibrium(), modifier::heaviside, inp.getForm()};
    return TokamakMagneticField( in.R0(), mod::createPsip(in.get_psip(), psi0,
        alpha, sign), in.get_ipol(), params);
}


} //namespace geo
} //namespace dg
