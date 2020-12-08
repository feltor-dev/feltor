#ifdef JSONCPP_VERSION_STRING
#include "magnetic_field.h"
#include "solovev.h"
#include "guenther.h"
#include "polynomial.h"
#include "toroidal.h"
#include <dg/file/json_utilities.h>

/*!@file
 *
 * Making of penalization regions
 */
namespace dg{
namespace geo{
///@addtogroup geom
///@{

/**
 * @brief Create a Magnetic field based on the given parameters
 *
 * This function abstracts the Magnetic field generation. It reads an
 * input Json file that tells this function via the "equilibrium" parameter which
 * field to generate and which parameters to expect in the file, for example if
 * "equilibrium" reads "toroidal", then only on additional parameter "R_0" is
 * read from the file and a field is constructed
 * @param gs Has to contain "equilibrium" which is converted dg::geo::equilibrium,
 * i.e. "solovev", "polynomial", .... After that the respective parameters are created,
 * for example if "solovev", then the dg::geo::solovev::Parameters( gs, mode) is called and forwarded to dg::geo::createSolovevField(gp); similar for the rest
 * @param mode signifies what to do if an error occurs
 * @return A magnetic field object
 * @attention This function is only defined if \c json/json.h is included before \c dg/geometries/geometries.h
 */
static inline TokamakMagneticField createMagneticField( Json::Value gs, file::error mode)
{
    std::string e = file::get( mode, gs, "equilibrium", "solovev" ).asString();
    equilibrium equi = str2equilibrium.at( e);
    switch( equi){
        case equilibrium::polynomial:
        {
            polynomial::Parameters gp( gs, mode);
            return createPolynomialField( gp);
        }
        case equilibrium::toroidal:
        {
            double R0 = file::get( mode, gs, "R_0", 10).asDouble();
            return createToroidalField( R0);
        }
        case equilibrium::guenther:
        {
            double I0 = file::get( mode, gs, "I_0", 20).asDouble();
            double R0 = file::get( mode, gs, "R_0", 10).asDouble();
            return createGuentherField( R0, I0);
        }
        case equilibrium::circular:
        {
            double I0 = file::get( mode, gs, "I_0", 20).asDouble();
            double R0 = file::get( mode, gs, "R_0", 10).asDouble();
            return createCircularField( R0, I0);
        }
        default:
        {
            solovev::Parameters gp( gs, mode);
            return createSolovevField( gp);
        }
    }
}


///@cond
namespace detail{
void transform_psi( TokamakMagneticField mag, double& psi0, double& alpha0, double& sign0)
{
    double RO=mag.R0(), ZO=0.;
    dg::geo::findOpoint( mag.get_psip(), RO, ZO);
    double psipO = mag.psip()( RO, ZO);
    double wall_psi0p = (1.-psi0*psi0)*psipO;
    double wall_alpha0p = -(2.*psi0+alpha0)*alpha0*psipO;
    psi0 = wall_psi0p + sign0*wall_alpha0p/2.;
    alpha0 = fabs( wall_alpha0p/2.);
    sign0 = sign0*((psipO>0)-(psipO<0));
}
}//namespace detail
///@endcond

/**
 * @brief Modify Magnetic Field above or below certain Psi values according to given parameters
 *
 * We modify psi above or below certain Psip values to a constant using the
 * \c dg::IPolynomialHeaviside function (an approximation to the integrated Heaviside
 * function with width alpha), i.e. we replace psi with IPolynomialHeaviside(psi).
 * This subsequently modifies all derivatives of psi and the poloidal
 * current in this region.
 * @param gs forwarded to dg::geo::createMagneticField
 * @param jsmod must contain the field "wall": "type" which has one of the values "none", then no other values are required; "heaviside" then requires
 *  "wall": "boundary" value where psi is modified to a constant psi0
 * "wall": "alpha" radius of the transition region where the modification acts (smaller is quicker);
 * or "sol_pfr", then requires
 *  "wall": "boundary" and
 * "wall": "alpha" must be arrays of size 2 to indicate values for the SOL and the PFR respectively
 * @param mode Determines behaviour in case of an error
 * @param wall (out) On output contains the region where the wall is applied
 * @param transition (out) On output contains the region where the transition of Psip to a constant value occurs
 * @note Per default the dampening happens nowhere
 * @return A magnetic field object
 * @attention This function is only defined if \c json/json.h is included before \c dg/geometries/geometries.h
 */
static inline TokamakMagneticField createModifiedField( Json::Value gs, Json::Value jsmod, file::error mode, CylindricalFunctor& wall, CylindricalFunctor& transition)
{
    std::string e = file::get( mode, gs, "equilibrium", "solovev" ).asString();
    equilibrium equi = str2equilibrium.at( e);
    std::string m = file::get( mode, jsmod, "wall", "type", "heaviside" ).asString();
    modifier mod = str2modifier.at( m);
    std::string d = file::get( mode, gs, "description", "standardX" ).asString();
    description desc = str2description.at( d);
    TokamakMagneticField mag = createMagneticField( gs, mode);
    const MagneticFieldParameters& inp = mag.params();
    MagneticFieldParameters mod_params{ inp.a(), inp.elongation(),
        inp.triangularity(), inp.getEquilibrium(), mod, inp.getDescription()};
    CylindricalFunctorsLvl2 mod_psip;
    switch (mod) {
        default: //none
        {
            wall = mod::DampingRegion( mod::nowhere, mag.psip(), 0, 0, 0);
            transition = mod::MagneticTransition( mod::nowhere, mag.psip(), 0, 0, 0);
            return mag;
        }
        case modifier::heaviside:
        {
            double psi0 = file::get( mode, jsmod, "wall", "boundary", 1.1 ).asDouble();
            double alpha = file::get( mode, jsmod, "wall", "alpha", 0.2 ).asDouble();
            double sign = +1;
            if( desc == description::standardX || desc == description::standardO ||
                    desc == description::doubleX)
                detail::transform_psi( mag, psi0, alpha, sign);
            else
                sign = file::get( mode, jsmod, "wall", "sign", -1. ).asDouble();

            mod_psip = mod::createPsip( mod::everywhere, mag.get_psip(), psi0, alpha, sign);
            wall = mod::DampingRegion( mod::everywhere, mag.psip(), psi0, alpha, -sign);
            transition = mod::MagneticTransition( mod::everywhere, mag.psip(), psi0, alpha, sign);
            break;
        }
        case modifier::sol_pfr:
        {
            double psi0 = file::get_idx( mode, jsmod, "wall", "boundary",0, 1.1 ).asDouble();
            double alpha0 = file::get_idx( mode, jsmod, "wall", "alpha",0, 0.2 ).asDouble();
            double psi1 = file::get_idx( mode, jsmod, "wall", "boundary",1, 0.97 ).asDouble();
            double alpha1 = file::get_idx( mode, jsmod, "wall", "alpha",1, 0.2 ).asDouble();
            switch( desc){
                case description::standardX:
                {
                    double sign0 = +1., sign1 = -1.;
                    detail::transform_psi( mag, psi0, alpha0, sign0);
                    detail::transform_psi( mag, psi1, alpha1, sign1);
                    //we can find the X-point
                    double RX = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
                    double ZX = -1.1*mag.params().elongation()*mag.params().a();
                    dg::geo::findXpoint( mag.get_psip(), RX, ZX);
                    CylindricalFunctorsLvl2 mod0_psip;
                    mod0_psip = mod::createPsip(
                            mod::everywhere, mag.get_psip(), psi0, alpha0, sign0);
                    mod_psip = mod::createPsip(
                            mod::HeavisideZ( ZX, -1), mod0_psip, psi1, alpha1, sign1);
                    CylindricalFunctor wall0 = mod::DampingRegion( mod::everywhere, mag.psip(), psi0, alpha0, -sign0);
                    CylindricalFunctor transition0 = mod::MagneticTransition( mod::everywhere, mag.psip(), psi0, alpha0, sign0);
                    CylindricalFunctor wall1 = mod::DampingRegion( mod::HeavisideZ(ZX, -1), mag.psip(), psi1, alpha1, -sign1);
                    CylindricalFunctor transition1 = mod::MagneticTransition( mod::HeavisideZ(ZX, -1), mag.psip(), psi1, alpha1, sign1);
                    wall = mod::SetUnion( wall0, wall1);
                    transition = mod::SetUnion( transition0, transition1);
                    break;
                }
                case description::doubleX:
                {
                    double sign0 = +1., sign1 = -1.;
                    detail::transform_psi( mag, psi0, alpha0, sign0);
                    detail::transform_psi( mag, psi1, alpha1, sign1);
                    //we can find the X-point
                    double RX1 = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
                    double ZX1 = -1.1*mag.params().elongation()*mag.params().a();
                    dg::geo::findXpoint( mag.get_psip(), RX1, ZX1);
                    double RX2 = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
                    double ZX2 = +1.1*mag.params().elongation()*mag.params().a();
                    dg::geo::findXpoint( mag.get_psip(), RX2, ZX2);
                    CylindricalFunctorsLvl2 mod0_psip, mod1_psip;
                    mod0_psip = mod::createPsip(
                            mod::everywhere, mag.get_psip(), psi0, alpha0, sign0);
                    mod1_psip = mod::createPsip(
                            mod::HeavisideZ( ZX1, -1), mod0_psip, psi1, alpha1, sign1);
                    mod_psip = mod::createPsip(
                            mod::HeavisideZ( ZX2, +1), mod1_psip, psi1, alpha1, sign1);
                    CylindricalFunctor wall0 = mod::DampingRegion( mod::everywhere, mag.psip(), psi0, alpha0, -sign0);
                    CylindricalFunctor wall1 = mod::DampingRegion( mod::HeavisideZ(ZX1, -1), mag.psip(), psi1, alpha1, -sign1);
                    CylindricalFunctor wall2 = mod::DampingRegion( mod::HeavisideZ(ZX2, +1), mag.psip(), psi1, alpha1, -sign1);
                    CylindricalFunctor transition0 = mod::MagneticTransition( mod::everywhere, mag.psip(), psi0, alpha0, sign0);
                    CylindricalFunctor transition1 = mod::MagneticTransition( mod::HeavisideZ(ZX1, -1), mag.psip(), psi1, alpha1, sign1);
                    CylindricalFunctor transition2 = mod::MagneticTransition( mod::HeavisideZ(ZX2, +1), mag.psip(), psi1, alpha1, sign1);
                    transition = mod::SetUnion( mod::SetUnion( transition0, transition1), transition2);
                    wall = mod::SetUnion( mod::SetUnion( wall0, wall1), wall2);
                    break;
                }
                default:
                {
                    double sign0 = file::get_idx( mode, jsmod, "wall", "sign",0, -1. ).asDouble();
                    double sign1 = file::get_idx( mode, jsmod, "wall", "sign", 1, +1. ).asDouble();
                    CylindricalFunctorsLvl2 mod0_psip;
                    mod0_psip = mod::createPsip(
                            mod::everywhere, mag.get_psip(), psi0, alpha0, sign0);
                    mod_psip = mod::createPsip(
                            mod::everywhere, mod0_psip, psi1, alpha1, sign1);
                    CylindricalFunctor wall0 = mod::DampingRegion( mod::everywhere, mag.psip(), psi0, alpha0, sign0);
                    CylindricalFunctor transition0 = mod::MagneticTransition( mod::everywhere, mag.psip(), psi0, alpha0, sign0);
                    CylindricalFunctor wall1 = mod::DampingRegion( mod::everywhere, mag.psip(), psi1, alpha1, sign1);
                    CylindricalFunctor transition1 = mod::MagneticTransition( mod::everywhere, mag.psip(), psi1, alpha1, sign1);
                    wall = mod::SetUnion( wall0, wall1);
                    transition = mod::SetUnion( transition0, transition1);
                    break;
                }
            }
        }
    }
    switch( equi){
        case equilibrium::solovev:
        {
            solovev::Parameters gp( gs, mode);
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
///@}

///@addtogroup profiles
///@{

static inline CylindricalFunctor createWallRegion( Json::Value gs, Json::Value jsmod, file::error mode)
{
    CylindricalFunctor wall, transition;
    TokamakMagneticField mag = createModifiedField( gs, jsmod, mode, wall, transition);
    return wall;
}

/**
 * @brief Create the sheath region where fieldlines intersect the boundary
 *
 * Check if any fieldlines that are not in the wall region intersect the boundary
 * and determine whether the poloidal field points towards or away from the wall
 * @param jsmod must contain the field
 * "sheath": "boundary" value where sheath region begins in units of minor radius a
 * "sheath": "alpha" radius of the transition region where the modification acts in units of minor radius a
 * @param mode Determines behaviour in case of an error
 * @param mag (in) the magnetic field to find the direction of the field
 * towards or away from the sheath
 * @param wall (in) the penalization region that represents the actual
 * (perpendicular) wall without the divertor
 * @param R0 left boundary
 * @param R1 right boundary
 * @param Z0 bottom boundary
 * @param Z1 top boundary
 * @param sheath (out) contains the region recognized as sheath
 * @param direction (out) contains (+/-) indicating direction of magnetic field
 * to closest sheath boundary (defined on entire box)
 *
 * @return sheath region
 */
static inline void createSheathRegion(
        Json::Value jsmod, file::error mode, TokamakMagneticField mag,
        CylindricalFunctor wall, double R0, double R1, double Z0, double Z1,
        CylindricalFunctor& sheath, CylindricalFunctor& direction )
{
    Grid1d gR1d ( R0, R1, 1, 100);
    Grid1d gZ1d ( Z0, Z1, 1, 100);
    std::array<bool,2> sheathR = {false,false}, sheathZ = {false,false};
    for ( unsigned i=0; i<100; i++)
    {
        if( wall( R0+i*gR1d.h(), Z0) == 0)
            sheathR[0] = true;
        if( wall( R0+i*gR1d.h(), Z1) == 0)
            sheathR[1] = true;
        if( wall( R0, Z0 + i*gZ1d.h()) == 0)
            sheathZ[0] = true;
        if( wall( R1, Z0 + i*gZ1d.h()) == 0)
            sheathZ[1] = true;
    }
    std::vector<double> horizontal_sheath, vertical_sheath;
    if( true == sheathR[0])
        horizontal_sheath.push_back( Z0);
    if( true == sheathR[1])
        horizontal_sheath.push_back( Z1);
    if( true == sheathZ[0])
        vertical_sheath.push_back( R0);
    if( true == sheathZ[1])
        vertical_sheath.push_back( R1);
    //direction
    direction = dg::geo::WallDirection( mag, vertical_sheath, horizontal_sheath);
    //sheath
    CylindricalFunctor dist = dg::WallDistance( vertical_sheath, horizontal_sheath);
    double boundary = file::get( mode, jsmod, "sheath", "boundary", 0.1 ).asDouble();
    double alpha = file::get( mode, jsmod, "sheath", "alpha", 0.01 ).asDouble();
    double a = mag.params().a();
    dg::PolynomialHeaviside poly( boundary*a - alpha*a/2., alpha*a/2., -1);
    sheath = dg::compose( poly, dist);
    sheath = mod::SetIntersection( mod::SetNot( wall), sheath);
}
///@}

} //namespace geo
}//namespace dg
#endif //JSONCPP_VERSION_STRING
