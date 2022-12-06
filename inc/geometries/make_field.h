#ifdef JSONCPP_VERSION_STRING
#include "magnetic_field.h"
#include "solovev.h"
#include "guenter.h"
#include "polynomial.h"
#include "toroidal.h"
#include "fieldaligned.h"
#include <dg/file/json_utilities.h>

/*!@file
 *
 * Making of penalization regions
 */
namespace dg{
namespace geo{

/**
 * @brief Create a Magnetic field based on the given parameters
 *
 * This function abstracts the Magnetic field generation. It reads an
 * input Json file that tells this function via the "equilibrium" parameter which
 * field to generate and which parameters to expect in the file.
 * See a list of possible combinations in the following
 * @copydoc hide_solovev_json
 * @copydoc hide_polynomial_json
 *
 * @code
// Purely toroidal magnetic field
{
    "equilibrium" : "toroidal",
    "R_0" : 10
}
// Automatically chosen:
// description : "none",
// a : 1.0,
// elongation : 1.0,
// triangularity : 0.0
 * @endcode
 * @code
// Circular/ellipsoid flux surfaces
{
    "equilibrium" : "circular",
    "I_0" : 10,
    "R_0" : 3,
    "a" : 1.0,
    "b" : 1.0
}
// Automatically chosen:
// description : "standardO",
// elongation : 1.0,
// triangularity : 0.0
 * @endcode
 * @code
// The guenter magnetic field
{
    "equilibrium" : "guenter",
    "I_0" : 10,
    "R_0" : 3
}
// Automatically chosen:
// description : "square",
// a : 1.0,
// elongation : 1.0,
// triangularity : 0.0
 * @endcode
 * @sa \c dg::geo::description to see valid values for the %description field
 *
 * @param gs Has to contain "equilibrium" which is converted \c dg::geo::equilibrium,
 * i.e. "solovev", "polynomial", .... After that the respective parameters are created,
 * for example if "solovev", then the dg::geo::solovev::Parameters( gs) is called and forwarded to dg::geo::createSolovevField(gp); similar for the rest
 * @return A magnetic field object
 * @attention This function is only defined if \c json/json.h is included before \c dg/geometries/geometries.h
 * @ingroup geom
 */
static inline TokamakMagneticField createMagneticField( dg::file::WrappedJsonValue gs)
{
    std::string e = gs.get( "equilibrium", "solovev" ).asString();
    equilibrium equi = equilibrium::solovev;
    try{
        equi = str2equilibrium.at( e);
    }catch ( std::out_of_range& err)
    {
        std::string message = "ERROR: Key \"" + e
            + "\" not valid in field:\n\t"
            + gs.access_string() + "\"equilibrium\" \n";
        throw std::out_of_range(message);
    }
    switch( equi){
        case equilibrium::polynomial:
        {
            polynomial::Parameters gp( gs);
            return createPolynomialField( gp);
        }
        case equilibrium::toroidal:
        {
            double R0 = gs.get( "R_0", 10.0).asDouble();
            return createToroidalField( R0);
        }
        case equilibrium::guenter:
        {
            double I0 = gs.get( "I_0", 20.0).asDouble();
            double R0 = gs.get( "R_0", 10.0).asDouble();
            return createGuenterField( R0, I0);
        }
        case equilibrium::circular:
        {
            double I0 = gs.get( "I_0", 20.0).asDouble();
            double R0 = gs.get( "R_0", 10.0).asDouble();
            double a = gs.get( "a", 1.0).asDouble();
            double b = gs.get( "b", 1.0).asDouble();
            return createCircularField( R0, I0, a, b);
        }
#ifdef BOOST_VERSION
        case equilibrium::taylor:
        {
            solovev::Parameters gp( gs);
            return createTaylorField( gp);
        }
#endif
        default:
        {
            solovev::Parameters gp( gs);
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

///@addtogroup wall
///@{
/**
 * @brief Modify Magnetic Field above or below certain Psi values according to given parameters
 *
 * We modify psi above or below certain Psip values to a constant using the
 * \c dg::IPolynomialHeaviside function (an approximation to the integrated Heaviside
 * function with width alpha), i.e. we replace psi with IPolynomialHeaviside(psi).
 * This subsequently modifies all derivatives of psi and the poloidal
 * current in this region.
 *
@code
{
    "type": "none"
}
{
    "type": "heaviside",
    "boundary": 1.1,
    "alpha": 0.20
}
{
    "type": "sol_pfr",
    "boundary": [1.1,0.998],
    // First value indicates SOL, second the PFR
    "alpha": [0.10,0.10]
}
@endcode
@sa dg::geo::modification for possible values of "type" parameter
 * @param gs forwarded to \c dg::geo::createMagneticField
 * @param jsmod contains the fields described above to steer the creation of a modification region
 * @param wall (out) On output contains the region where the wall is applied, the functor returns 1 where the wall is, 0 where there it is not and 0<f<1 in the transition region
 * @param transition (out) On output contains the region where the transition of Psip to a constant value occurs, the functor returns 0<f<=1 for when there is a transition and 0 else
 * @note Per default the dampening happens nowhere
 * @return A magnetic field object
 * @attention This function is only defined if \c json/json.h is included before \c dg/geometries/geometries.h
 */
static inline TokamakMagneticField createModifiedField(
        dg::file::WrappedJsonValue gs, dg::file::WrappedJsonValue jsmod,
        CylindricalFunctor& wall, CylindricalFunctor& transition)
{
    TokamakMagneticField mag = createMagneticField( gs);
    const MagneticFieldParameters& inp = mag.params();
    description desc = inp.getDescription();
    equilibrium equi = inp.getEquilibrium();
    std::string m = jsmod.get( "type", "heaviside" ).asString();
    modifier mod = str2modifier.at( m);
    MagneticFieldParameters mod_params{ inp.a(), inp.elongation(),
        inp.triangularity(), equi, mod, desc};
    CylindricalFunctorsLvl2 mod_psip;
    switch (mod) {
        default: //none
        {
            wall = mod::DampingRegion( mod::nowhere, mag.psip(), 0, 1, -1);
            transition = mod::MagneticTransition( mod::nowhere, mag.psip(),
                    0,  1, -1);
            return mag;
        }
        case modifier::heaviside:
        {
            double psi0 = jsmod.get( "boundary", 1.1 ).asDouble();
            double alpha = jsmod.get( "alpha", 0.2 ).asDouble();
            double sign = +1;
            if( desc == description::standardX || desc == description::standardO ||
                    desc == description::doubleX)
                detail::transform_psi( mag, psi0, alpha, sign);
            else
                sign = jsmod.get( "sign", -1. ).asDouble();

            mod_psip = mod::createPsip( mod::everywhere, mag.get_psip(), psi0, alpha, sign);
            wall = mod::DampingRegion( mod::everywhere, mag.psip(), psi0, alpha, -sign);
            transition = mod::MagneticTransition( mod::everywhere, mag.psip(), psi0, alpha, sign);
            break;
        }
        case modifier::sol_pfr:
        {
            double psi0     = jsmod["boundary"].get( 0, 1.1 ).asDouble();
            double alpha0   = jsmod["alpha"].get(0, 0.2 ).asDouble();
            double psi1     = jsmod["boundary"].get(1, 0.97 ).asDouble();
            double alpha1   = jsmod[ "alpha"].get(1, 0.2 ).asDouble();
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
                    double sign0 = jsmod[ "sign"].get( 0, -1. ).asDouble();
                    double sign1 = jsmod[ "sign"].get( 1, +1. ).asDouble();
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
            solovev::Parameters gp( gs);
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


/// A convenience function call for \c dg::geo::createModifiedField that ignores the transition parameter and returns the wall
static inline CylindricalFunctor createWallRegion( dg::file::WrappedJsonValue gs, dg::file::WrappedJsonValue jsmod)
{
    CylindricalFunctor wall, transition;
    TokamakMagneticField mag = createModifiedField( gs, jsmod, wall, transition);
    return wall;
}

/**
 * @brief Create the sheath region where fieldlines intersect the boundary
 *
 * The sheath functor that comes out of this does
 * (i) on each of the four lines defined by the two vertical (R0, R1) and two
 * horizontal (Z0, Z1) boundaries check if the "wall" functor is zero
 * anywhere on the line: if not then move this boundary far away
 * (ii) Measure the angular distance along the fieldline (both in positive and
 * negative direction) to the remaining walls
 * (iii) Modify the angular distances with a dg::PolynomialHeaviside functor
 * with parameters given in jsmod:
@code
{
    "boundary" : 0.0625, // value where sheath region begins in units of 2Pi
    "alpha" : 0.015625 // diameter of the transition region in units of 2Pi
}
@endcode
 * (iv) The sheath region is the SetUnion of positive and negative functor,
 * together with the SetIntersection with the SetNot(wall) region.
 * @param jsmod must contain fields as described above
 * @param mag (in) the (unmodified) magnetic field, used to integrate
 * the field towards or away from the sheath
 * @param wall (in) the penalization region that represents the actual
 * (perpendicular) wall without the divertor (if 0 on the boundary the boundary will be considered to be a sheath, else the boundary will be ignored)
 * @param sheath_walls (inout) on input contains the box boundaries, on output
 * the non-sheath boundaries are moved far away
 * @param sheath (out) contains the region recognized as sheath (returning +1 within
 * the sheath and 0 outside of it and something in-between in the transition region)
 */
static inline void createSheathRegion(
    dg::file::WrappedJsonValue jsmod, TokamakMagneticField mag,
    CylindricalFunctor wall, dg::Grid2d& sheath_walls,
    CylindricalFunctor& sheath)
{
    double R0 = sheath_walls.x0(), R1 = sheath_walls.x1();
    double Z0 = sheath_walls.y0(), Z1 = sheath_walls.y1();
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
    if( false == sheathR[0]) Z0 = -1e10;
    if( false == sheathR[1]) Z1 = 1e10;
    if( false == sheathZ[0]) R0 = -1e10;
    if( false == sheathZ[1]) R1 = 1e10;
    sheath_walls = dg::Grid2d( R0, R1, Z0, Z1, 1,1,1);
    double boundary = jsmod.get( "boundary", 0.0625 ).asDouble(); // 1/16
    double alpha    = jsmod.get( "alpha", 0.015625 ).asDouble(); // 1/64
    CylindricalFunctor distM = dg::geo::WallFieldlineDistance( dg::geo::createBHat(
            mag), sheath_walls, (-boundary-1e-3)*2.0*M_PI,
            1e-6, "phi");
    CylindricalFunctor distP = dg::geo::WallFieldlineDistance( dg::geo::createBHat(
            mag), sheath_walls, (+boundary+1e-3)*2.0*M_PI,
            1e-6, "phi");
    dg::PolynomialHeaviside polyM( -boundary*2.*M_PI + alpha*M_PI, alpha*M_PI, +1);
    dg::PolynomialHeaviside polyP(  boundary*2.*M_PI - alpha*M_PI, alpha*M_PI, -1);
    auto sheathM = dg::compose( polyM, distM); //positive (because distance)
    auto sheathP = dg::compose( polyP, distP);
    sheath = mod::SetUnion( sheathM, sheathP);
    sheath = mod::SetIntersection( mod::SetNot( wall), sheath);
}
///@}

} //namespace geo
}//namespace dg
#endif //JSONCPP_VERSION_STRING
