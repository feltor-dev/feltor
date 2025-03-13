#pragma once

#include <cmath>
#include <array>

#include "dg/algorithm.h"
#include "fieldaligned.h"
#include "modified.h"
#include "magnetic_field.h"
#include "fluxfunctions.h"

namespace dg{
namespace geo{


/**
 * @brief %Distance to wall along fieldline in phi or s coordinate
 *
 * The following differential equation is integrated
 * \f[ \frac{ d R}{d \varphi} = b^R / b^\varphi \\
 *     \frac{ d Z}{d \varphi} = b^Z / b^\varphi \\
 *     \frac{ d s}{s \varphi} = 1   / b^\varphi
 * \f]
 * for initial conditions \f$ (R,Z,0)\f$ until either a maximum angle is reached or until \f$ (R,Z) \f$ leaves the given domain. In the latter case a bisection algorithm is used to find the exact angle \f$\varphi_l\f$ of leave. Either the angle \f$ \varphi_l\f$ or the corresponding \f$ s_l\f$ is returned by the function.
 * @ingroup wall
 * @attention The sign of the angle coordinate in this class (unlike in
 * Fieldaligned) is defined with respect to the direction of the magnetic
 * field. Thus, for a positive maxPhi, the distance (both "phi" and "s")
 * will be positive and for negative maxPhi the distance is negative.
 */
struct WallFieldlineDistance : public aCylindricalFunctor<WallFieldlineDistance>
{
    /**
     * @brief Construct with vector field, domain
     *
     * @param vec The vector field to integrate
     * @param domain The box
     * @param maxPhi the maximum angle to integrate to (something like plus or minus 2.*M_PI)
     * @param eps the accuracy of the fieldline integrator
     * @param type either "phi" then the distance is computed in the angle coordinate
     * or "s" then the distance is computed in the s parameter
     * @param predicate only integrate points where predicate is true
     */
    WallFieldlineDistance(
        const dg::geo::CylindricalVectorLvl0& vec,
        const dg::aRealTopology2d<double>& domain,
        double maxPhi, double eps, std::string type,
        std::function<bool(double,double)> predicate = mod::everywhere):
        m_pred( predicate),
        m_domain( domain), m_cyl_field(vec),
        m_deltaPhi( maxPhi), m_eps( eps), m_type(type)
    {
        if( m_type != "phi" && m_type != "s")
            throw std::runtime_error( "Distance type "+m_type+" not recognized!\n");
    }
    /**
     * @brief Integrate fieldline until wall is reached
     *
     * If wall is not reached integration stops at maxPhi given in the constructor
     * @param R starting coordinate in R
     * @param Z starting coordinate in Z
     *
     * @return distance in phi or s
     * @sa dg::integrateERK
     */
    double do_compute( double R, double Z) const
    {
        std::array<double,3> coords{ R, Z, 0}, coordsP(coords);
        // determine sign
        m_cyl_field( 0., coords, coordsP);
        double sign = coordsP[2] > 0 ? +1. : -1.;
        double phi1 = sign*m_deltaPhi; // we integrate negative ...
        if( m_pred(R,Z)) // only integrate if necessary
        {
            try{
                dg::Adaptive<dg::ERKStep<std::array<double,3>>> adapt(
                        "Dormand-Prince-7-4-5", coords);
                dg::AdaptiveTimeloop<std::array<double,3>> odeint( adapt,
                        m_cyl_field, dg::pid_control, dg::fast_l2norm, m_eps,
                        1e-10); // using 1e-10 instead of eps may cost 10% of performance but is what we also use in Fieldaligned
                odeint.integrate_in_domain( 0., coords, phi1, coordsP, 0.,
                        m_domain, m_eps);
                //integration
            }catch (std::exception& e)
            {
                // if not possible the distance is large
                //std::cerr << e.what();
                phi1 = sign*m_deltaPhi;
                coordsP[2] = 1e6*phi1;
            }
        }
        else
            coordsP[2] = 1e6*phi1;
        if( m_type == "phi")
            return sign*phi1;
        return coordsP[2];
    }

    private:
    std::function<bool(double,double)> m_pred;
    const dg::Grid2d m_domain;
    dg::geo::detail::DSFieldCylindrical3 m_cyl_field;
    double m_deltaPhi, m_eps;
    std::string m_type;
};

/**
 * @brief Normalized coordinate relative to wall along fieldline in phi or s coordinate
 *
 * The following differential equation is integrated
 * \f[ \frac{ d R}{d \varphi} = b^R / b^\varphi \\
 *     \frac{ d Z}{d \varphi} = b^Z / b^\varphi \\
 *     \frac{ d s}{s \varphi} = 1   / b^\varphi
 * \f]
 * for initial conditions \f$ (R,Z,0)\f$ until either a maximum angle is reached or until \f$ (R,Z) \f$ leaves the given domain. In the latter case a bisection algorithm is used to find the exact angle \f$\varphi_l\f$ of leave.
 *
 * The difference to \c WallFieldlineDistance is that this class integrates the differential equations in **both** directions and normalizes the output to \f$ [-1,1]\f$.
 * -1 means at the negative sheath (you have to go agains the field to go out
 *  of the box), +1 at the postive sheath (you have to go with the field to go
 *  out of the box) and anything else is in-between; when the sheath cannot be
 *  reached 0 is returned
 * @ingroup wall
 * @attention The sign of the coordinate (both angle and distance) is defined
 * with respect to the direction of the magnetic field (not the angle
 * coordinate like in Fieldaligned)
 */
struct WallFieldlineCoordinate : public aCylindricalFunctor<WallFieldlineCoordinate>
{
    ///@copydoc WallFieldlineDistance::WallFieldlineDistance()
    ///@note The sign of \c maxPhi does not matter here as both directions are integrated
    WallFieldlineCoordinate(
        const dg::geo::CylindricalVectorLvl0& vec,
        const dg::aRealTopology2d<double>& domain,
        double maxPhi, double eps, std::string type,
        std::function<bool(double,double)> predicate = mod::everywhere
        ) :
        m_pred(predicate),
        m_domain( domain), m_cyl_field(vec),
        m_deltaPhi( maxPhi), m_eps( eps), m_type(type)
    {
        if( m_type != "phi" && m_type != "s")
            throw std::runtime_error( "Distance type "+m_type+" not recognized!\n");
    }
    double do_compute( double R, double Z) const
    {
        double phiP = m_deltaPhi, phiM = -m_deltaPhi;
        std::array<double,3> coords{ R, Z, 0}, coordsP(coords), coordsM(coords);
        // determine sign
        m_cyl_field( 0., coords, coordsP);
        double sign = coordsP[2] > 0 ? +1. : -1.;
        if( m_pred(R,Z)) // only integrate if necessary
        {
            try{
                dg::AdaptiveTimeloop<std::array<double,3>> odeint(
                        dg::Adaptive<dg::ERKStep<std::array<double,3>>>(
                            "Dormand-Prince-7-4-5", coords), m_cyl_field,
                        dg::pid_control, dg::fast_l2norm, m_eps,
                        1e-10); // using 1e-10 instead of eps may cost 10% of performance but is what we also use in Fieldaligned
                odeint.integrate_in_domain( 0., coords, phiP, coordsP, 0.,
                        m_domain, m_eps);
                odeint.integrate_in_domain( 0., coords, phiM, coordsM, 0.,
                        m_domain, m_eps);
            }catch (std::exception& e)
            {
                // if not possible the distance is large
                phiP = m_deltaPhi;
                coordsP[2] = 1e6*phiP;
                phiM = -m_deltaPhi;
                coordsM[2] = 1e6*phiM;
            }
        }
        else
        {
            coordsP[2] = 1e6*phiP;
            coordsM[2] = 1e6*phiM;
        }
        if( m_type == "phi")
            return sign*(-phiP-phiM)/(phiP-phiM);
        double sP = coordsP[2], sM = coordsM[2];
        double value = sign*(-sP-sM)/(sP-sM);
        if( (phiM <= -m_deltaPhi) and (phiP >= m_deltaPhi))
            return 0.; //return exactly zero if sheath not reached
        if( (phiM <= -m_deltaPhi))
            return value*sign > 0 ? value : 0.; // avoid false negatives
        if( (phiP >= m_deltaPhi))
            return value*sign < 0 ? value : 0.; // avoid false positives
        return value;
    }

    private:
    std::function<bool(double,double)> m_pred;
    const dg::Grid2d m_domain;
    dg::geo::detail::DSFieldCylindrical3 m_cyl_field;
    double m_deltaPhi, m_eps;
    std::string m_type;
};

}//namespace geo
}//namespace dg
