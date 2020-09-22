#include "magnetic_field.h"
#include "solovev.h"
#include "guenther.h"
#include "polynomial.h"
#include "toroidal.h"
//#include "taylor.h" //only if boost is included
#include <dg/file/json_utilities.h>


namespace dg{
namespace geo{

static inline TokamakMagneticField createMagneticField( Json::Value js, file::error mode)
{
    std::string e = file::get( mode, js, "equilibrium", "solovev" ).asString();
    equilibrium equi = str2equilibrium.at( e);
    switch( equi){
        case equilibrium::polynomial:
        {
            polynomial::Parameters gp( js, mode);
            return createPolynomialField( gp);
        }
        case equilibrium::toroidal:
        {
            double R0 = file::get( mode, js, "R_0", 10).asDouble();
            return createToroidalField( R0);
        }
        case equilibrium::guenther:
        {
            double I0 = file::get( mode, js, "I_0", 20).asDouble();
            double R0 = file::get( mode, js, "R_0", 10).asDouble();
            return createGuentherField( R0, I0);
        }
        case equilibrium::circular:
        {
            double I0 = file::get( mode, js, "I_0", 20).asDouble();
            double R0 = file::get( mode, js, "R_0", 10).asDouble();
            return createCircularField( R0, I0);
        }
        default:
        {
            solovev::Parameters gp( js, mode);
            return createSolovevField( gp);
        }
    }
}

/**
 * @brief Modify Magnetic Field above or below certain Psi values according to given parameters
 *
 * We modify psi above or below certain Psip values to a constant using the
 * \c dg::IPolynomialHeaviside function (an approximation to the integrated Heaviside
 * function with width alpha), i.e. we replace psi with IPolynomialHeaviside(psi).
 * This subsequently modifies all derivatives of psi and the poloidal
 * current in this region.
 * @param in Magnetic field to change
 * @param psi0 boundary value where psi is modified to a constant psi0
 * @param alpha radius of the transition region where the modification acts (smaller is quicker)
 * @param sign determines which side of Psi to dampen (negative or positive, forwarded to \c dg::IPolynomialHeaviside)
 * @param ZX Here you can give a Z value (of the X-point).
 * @param side  The modification will only happen on either the upper (positive) or lower (negative) side of ZX in Z.
 * @note Per default the dampening happens everywhere
 * @return A magnetic field object
 * @ingroup geom
 */
static inline TokamakMagneticField createModifiedField( Json::Value js, Json::Value jsmod, file::error mode, CylindricalFunctor& damping, CylindricalFunctor& transition)
{
    std::string e = file::get( mode, js, "equilibrium", "solovev" ).asString();
    equilibrium equi = str2equilibrium.at( e);
    std::string m = file::get( mode, jsmod, "damping", "modifier", "heaviside" ).asString();
    modifier mod = str2modifier.at( m);
    std::string d = file::get( mode, js, "description", "standardX" ).asString();
    description desc = str2description.at( d);
    TokamakMagneticField mag = createMagneticField( js, mode);
    const MagneticFieldParameters& inp = mag.params();
    MagneticFieldParameters mod_params{ inp.a(), inp.elongation(),
        inp.triangularity(), inp.getEquilibrium(), mod, inp.getDescription()};
    CylindricalFunctorsLvl2 mod_psip;
    switch (mod) {
        default: //none
        {
            damping = mod::DampingRegion( mod::nowhere, mag.psip(), 0, 0, 0);
            transition = mod::MagneticTransition( mod::nowhere, mag.psip(), 0, 0, 0);
            return mag;
        }
        case modifier::heaviside:
        {
            double psi0 = file::get( mode, jsmod, "damping", "boundary", 1.1 ).asDouble();
            double alpha = file::get( mode, jsmod, "damping", "alpha", 0.2 ).asDouble();
            double sign = -1;
            if( desc == description::standardX)
            {
                double RO=mag.R0(), ZO=0.;
                dg::geo::findOpoint( mag.get_psip(), RO, ZO);
                double psipO = mag.psip()( RO, ZO);
                double damping_psi0p = (1.-psi0*psi0)*psipO;
                double damping_alphap = -(2.*psi0+alpha)*alpha*psipO;
                //std::cout<< " damping "<< damping_psi0p << " "<<damping_alphap<<"\n";
                psi0 = damping_psi0p + damping_alphap/2.;
                alpha = fabs( damping_alphap/2.);
                sign = ((psipO>0)-(psipO<0));
            }
            else
                sign = file::get( mode, jsmod, "damping", "sign", -1. ).asDouble();

            mod_psip = mod::createPsip( mod::everywhere, mag.get_psip(), psi0, alpha, sign);
            damping = mod::DampingRegion( mod::everywhere, mag.psip(), psi0, alpha, -sign);
            transition = mod::MagneticTransition( mod::everywhere, mag.psip(), psi0, alpha, sign);
            break;
        }
        case modifier::sol_pfr:
        {
            CylindricalFunctorsLvl2 mod0_psip;
            double psi0 = file::get_idx( mode, jsmod, "damping", "boundary",0, 1.1 ).asDouble();
            double alpha0 = file::get_idx( mode, jsmod, "damping", "alpha",0, 0.2 ).asDouble();
            double psi1 = file::get_idx( mode, jsmod, "damping", "boundary",1, 0.97 ).asDouble();
            double alpha1 = file::get_idx( mode, jsmod, "damping", "alpha",1, 0.2 ).asDouble();
            switch( desc){
                case description::standardX:
                {
                    //we can find the X-point
                    double RX = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
                    double ZX = -1.1*mag.params().elongation()*mag.params().a();
                    dg::geo::findXpoint( mag.get_psip(), RX, ZX);
                    //and the X-point
                    double RO=mag.R0(), ZO=0.;
                    dg::geo::findOpoint( mag.get_psip(), RO, ZO);
                    double psipO = mag.psip()( RO, ZO);
                    double damping_psi0p = (1.-psi0*psi0)*psipO;
                    double damping_alpha0p = -(2.*psi0+alpha0)*alpha0*psipO;
                    double damping_psi1p = (1.-psi1*psi1)*psipO;
                    double damping_alpha1p = -(2.*psi1+alpha1)*alpha1*psipO;
                    psi0 = damping_psi0p + damping_alpha0p/2.;
                    psi1 = damping_psi1p - damping_alpha1p/2.;
                    alpha0 = fabs( damping_alpha0p/2.);
                    alpha1 = fabs( damping_alpha1p/2.);
                    double sign0 = ((psipO>0)-(psipO<0));
                    double sign1 = -sign0;
                    mod0_psip = mod::createPsip(
                            mod::everywhere, mag.get_psip(), psi0, alpha0, sign0);
                    mod_psip = mod::createPsip(
                            mod::HeavisideZ( ZX, -1), mod0_psip, psi1, alpha1, sign1);
                    CylindricalFunctor damping0 = mod::DampingRegion( mod::everywhere, mag.psip(), psi0, alpha0, -sign0);
                    CylindricalFunctor transition0 = mod::MagneticTransition( mod::everywhere, mag.psip(), psi0, alpha0, sign0);
                    CylindricalFunctor damping1 = mod::DampingRegion( mod::HeavisideZ(ZX, -1), mag.psip(), psi1, alpha1, -sign1);
                    CylindricalFunctor transition1 = mod::MagneticTransition( mod::HeavisideZ(ZX, -1), mag.psip(), psi1, alpha1, sign1);
                    damping = mod::Combine( damping0, damping1);
                    transition = mod::Combine( transition0, transition1);
                    break;
                }
                default:
                {
                    double sign0 = file::get_idx( mode, jsmod, "damping", "sign",0, -1. ).asDouble();
                    double sign1 = file::get_idx( mode, jsmod, "damping", "sign", 1, +1. ).asDouble();
                    mod0_psip = mod::createPsip(
                            mod::everywhere, mag.get_psip(), psi0, alpha0, sign0);
                    mod_psip = mod::createPsip(
                            mod::everywhere, mod0_psip, psi1, alpha1, sign1);
                    CylindricalFunctor damping0 = mod::DampingRegion( mod::everywhere, mag.psip(), psi0, alpha0, sign0);
                    CylindricalFunctor transition0 = mod::MagneticTransition( mod::everywhere, mag.psip(), psi0, alpha0, sign0);
                    CylindricalFunctor damping1 = mod::DampingRegion( mod::everywhere, mag.psip(), psi1, alpha1, sign1);
                    CylindricalFunctor transition1 = mod::MagneticTransition( mod::everywhere, mag.psip(), psi1, alpha1, sign1);
                    damping = mod::Combine( damping0, damping1);
                    transition = mod::Combine( transition0, transition1);
                    break;
                }
            }
        }
    }
    switch( equi){
        case equilibrium::solovev:
        {
            solovev::Parameters gp( js, mode);
            return TokamakMagneticField( gp.R_0,mod_psip,
                solovev::createIpol( gp, mod_psip), mod_params);
        }
        default:
        {
            return TokamakMagneticField( mag.R0(), mod_psip,
                    mag.get_ipol(), mod_params);
        }
    }
}

} //namespace geo
}//namespace dg
