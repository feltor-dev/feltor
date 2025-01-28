#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/algorithm.h"

#include "magnetic_field.h"


/*!@file
 *
 * Modified MagneticField objects
 */
namespace dg
{
namespace geo
{
/**
 * @brief A modification flux function
 */
namespace mod
{
//modify with a polynomial Heaviside function
///@addtogroup mod
///@{

/**
 * @brief \f$ \psi_{mod} := \begin{cases} H(\psi_p(R,Z))\text{ for } P(R,Z) \\
 * \psi_p(R,Z) \text { else }
 * \end{cases}
 * \f$
 *
 * where H is the integrated dg::IPolynomialHeaviside function and P is a predicate that returns either true or false.
 * @note the predicate can usually be true everywhere, the idea for the predicate is to be able to selectively target the private flux region(s) for modification.
 */
struct Psip: public aCylindricalFunctor<Psip>
{
    Psip( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, double psi0, double alpha, double sign = -1) :
        m_ipoly( psi0, alpha, sign), m_psip(psip), m_pred(predicate)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        if( m_pred( R,Z))
            return m_ipoly( psip);
        else
            return psip;
    }
    private:
    dg::IPolynomialHeaviside m_ipoly;
    std::function<double(double,double)> m_psip;
    std::function<bool(double,double)> m_pred;
};
struct PsipR: public aCylindricalFunctor<PsipR>
{
    PsipR( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, std::function<double(double,double)> psipR, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_psip(psip), m_psipR(psipR), m_pred(predicate)
    { }
    double do_compute(double R, double Z) const
    {
        double psip  = m_psip(R,Z);
        double psipR = m_psipR(R,Z);
        if( m_pred( R,Z))
            return psipR*m_poly( psip);
        else
            return psipR;
    }
    private:
    dg::PolynomialHeaviside m_poly;
    std::function<double(double,double)> m_psip, m_psipR;
    std::function<bool(double,double)> m_pred;
};
struct PsipZ: public aCylindricalFunctor<PsipZ>
{
    PsipZ( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, std::function<double(double,double)> psipZ, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_psip(psip), m_psipZ(psipZ), m_pred(predicate)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipZ = m_psipZ(R,Z);
        if( m_pred( R,Z))
            return psipZ*m_poly( psip);
        else
            return psipZ;
    }
    private:
    dg::PolynomialHeaviside m_poly;
    std::function<double(double,double)> m_psip, m_psipZ;
    std::function<bool(double,double)> m_pred;
};

struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
    PsipZZ( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, std::function<double(double,double)> psipZ, std::function<double(double,double)> psipZZ, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(psip), m_psipZ(psipZ), m_psipZZ(psipZZ), m_pred(predicate)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipZ = m_psipZ(R,Z);
        double psipZZ = m_psipZZ(R,Z);
        if( m_pred( R,Z))
            return psipZZ*m_poly( psip) + psipZ*psipZ*m_dpoly(psip);
        else
            return psipZZ;
    }
    private:
    dg::PolynomialHeaviside m_poly;
    dg::DPolynomialHeaviside m_dpoly;
    std::function<double(double,double)> m_psip, m_psipZ, m_psipZZ;
    std::function<bool(double,double)> m_pred;
};

struct PsipRR: public aCylindricalFunctor<PsipRR>
{
    PsipRR( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, std::function<double(double,double)> psipR, std::function<double(double,double)> psipRR, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(psip), m_psipR(psipR), m_psipRR(psipRR), m_pred(predicate)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipR = m_psipR(R,Z);
        double psipRR = m_psipRR(R,Z);
        if( m_pred( R,Z))
            return psipRR*m_poly( psip) + psipR*psipR*m_dpoly(psip);
        else
            return psipRR;
    }
    private:
    dg::PolynomialHeaviside m_poly;
    dg::DPolynomialHeaviside m_dpoly;
    std::function<double(double,double)> m_psip, m_psipR, m_psipRR;
    std::function<bool(double,double)> m_pred;
};
struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
    PsipRZ( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, std::function<double(double,double)> psipR, std::function<double(double,double)> psipZ, std::function<double(double,double)> psipRZ, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(psip), m_psipR(psipR), m_psipZ(psipZ), m_psipRZ(psipRZ), m_pred(predicate)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipR = m_psipR(R,Z);
        double psipZ = m_psipZ(R,Z);
        double psipRZ = m_psipRZ(R,Z);
        if( m_pred( R,Z))
            return psipRZ*m_poly( psip) + psipR*psipZ*m_dpoly(psip);
        else
            return psipRZ;
    }
    private:
    dg::PolynomialHeaviside m_poly;
    dg::DPolynomialHeaviside m_dpoly;
    std::function<double(double,double)> m_psip, m_psipR, m_psipZ, m_psipRZ;
    std::function<bool(double,double)> m_pred;
};


/**
 * @copydoc dg::geo::mod::Psip
 *
 * @note This is a helper function used in the implementation of \c dg::geo::createModifiedField
 * @param predicate P(R,Z) indicates the positions where Psi is to be modified (true) or not (false)
 * @param psip the flux function to be modified
 * @param psi0 parameter for dg::PolynomialHeaviside function
 * @param alpha parameter for dg::PolynomialHeaviside function
 * @param sign parameter for dg::PolynomialHeaviside function
 * @sa createModifiedField
 *
 * @return  the modified flux function
 */
inline dg::geo::CylindricalFunctorsLvl2 createPsip(
        const std::function<bool(double,double)> predicate,
        const CylindricalFunctorsLvl2& psip,
    double psi0, double alpha, double sign = -1)
{
    return CylindricalFunctorsLvl2(
            mod::Psip(predicate,psip.f(), psi0, alpha, sign),
            mod::PsipR(predicate,psip.f(), psip.dfx(), psi0, alpha, sign),
            mod::PsipZ(predicate,psip.f(), psip.dfy(), psi0, alpha, sign),
            mod::PsipRR(predicate,psip.f(), psip.dfx(), psip.dfxx(), psi0, alpha, sign),
            mod::PsipRZ(predicate,psip.f(), psip.dfx(), psip.dfy(), psip.dfxy(), psi0, alpha, sign),
            mod::PsipZZ(predicate,psip.f(), psip.dfy(), psip.dfyy(), psi0, alpha, sign));
}

///@}
///@cond
struct DampingRegion : public aCylindricalFunctor<DampingRegion>
{
    DampingRegion( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, double psi0, double alpha, double sign = -1) :
        m_poly( psi0, alpha, sign), m_psip(psip), m_pred(predicate)
    { }
    double do_compute(double R, double Z) const
    {
        if( m_pred( R,Z))
            return m_poly( m_psip(R,Z));
        else
            return 0;
    }
    private:
    dg::PolynomialHeaviside m_poly;
    std::function<double(double,double)> m_psip;
    std::function<bool(double,double)> m_pred;
};
struct MagneticTransition : public aCylindricalFunctor<MagneticTransition>
{
    MagneticTransition( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, double psi0, double alpha, double sign = -1) :
        m_dpoly( psi0, alpha, sign), m_psip(psip), m_pred(predicate)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        if( m_pred( R,Z))
            return m_dpoly( psip);
        else
            return 0;
    }
    private:
    dg::DPolynomialHeaviside m_dpoly;
    std::function<double(double,double)> m_psip;
    std::function<bool(double,double)> m_pred;
};
//some possible predicates
inline constexpr bool nowhere( double R, double Z){return false;}
inline constexpr bool everywhere( double R, double Z){return true;}
// positive above certain Z value ( deprecated in favor of Above)
struct HeavisideZ
{
    HeavisideZ( double Z_X, int side): m_ZX( Z_X), m_side(side) {}
    bool operator()(double R, double Z)const{
        if( Z < m_ZX && m_side <= 0) return true;
        if( Z >= m_ZX && m_side > 0) return true;
        return false;
    }
    private:
    double m_ZX;
    int m_side;
};

/// This one checks if a point lies on the right of a line stretching to
/// infinity given by either two or three points
struct RightSideOf
{
    RightSideOf( std::array<double,2> p1, std::array<double,2> p2)
    : RightSideOf( std::vector{p1,p2})
    {
    }
    RightSideOf( std::array<double,2> p1, std::array<double,2> p2,
        std::array<double,2> p3) : RightSideOf( std::vector{p1,p2,p3})
    {
    }
    RightSideOf( std::vector<std::array<double,2>> ps): m_ps(ps)
    {
        if( ps.size() != 2 and ps.size() != 3)
            throw Error( Message(_ping_) << "Give either 2 or 3 Points");
        for( unsigned u=0; u<ps.size(); u++)
        for( unsigned k=0; k<ps.size(); k++)
            if( u!=k and ps[u][0] == ps[k][0] and ps[u][1] == ps[k][1])
                throw Error( Message(_ping_) << "Points " <<k <<" and "<<u<<" must be different!");
    }
    bool operator() ( double R, double Z) const
    {
        // I'm sure it is possible to generalize this to more than three points
        // but it hurts my head right now
        std::array<double,2> x = {R,Z};
        std::vector<bool> right_of( m_ps.size()-1);
        // For each consecutive pair of points check if x lies on the right
        for( unsigned u=0; u<m_ps.size()-1; u++)
        {
            right_of[u] = right_handed( m_ps[u], x, m_ps[u+1]);
        }
        if( m_ps.size() == 2)
            return right_of[0];
        // else we have 3 points
        if ( right_handed( m_ps[0], m_ps[2], m_ps[1]))
        {
            if( right_of[0] and right_of[1]) // it is right of both lines
                return true;
            else
                return false;
        }
        else // left handed points
        {
            if( not right_of[0] and not right_of[1]) // it is left of both lines
                return false;
            else
                return true;
        }
    }
    private:
    // is true if p0,p1,p2 forms a right handed triangle i.e. go counter-clockwise
    // which is equivalent to saying that p1 is right of [p0,p2]
    bool right_handed( const std::array<double,2>& p0,
                       const std::array<double,2>& p1,
                       const std::array<double,2>& p2) const
    {
        //std::cout<< "p0 "<<p0[0]<<" "<<p0[1]<<" p1 "<<p1[0]<<" "<<p1[1]<<" p2 "<<p2[0]<<" "<<p2[1]<<"\n";
        // if v1 x v2 points up the system is right handed
        std::array<double,2> v1 = { p1[0]- p0[0], p1[1] - p0[1]};
        std::array<double,2> v2 = { p2[0]- p0[0], p2[1] - p0[1]};
        // Now check if z-component of cross product points up or down
        double v3z = v1[0]*v2[1] - v1[1]*v2[0];
        //std::cout<< "v3z "<<v3z<<"\n";
        return (v3z >= 0);
    }
    std::vector<std::array<double,2>> m_ps;
};

// Check if a point lies above or below a plane given by origin p0 and its normal vector (p1-p0)
struct Above
{
    // normal vector is defined by n = p1 - p0
    // if above is false the predicate returns false for points above the plane
    Above( std::array<double,2> p0, std::array<double,2> p1, bool above = true)
    : m_p0( p0), m_vec{ p1[0]-p0[0], p1[1]-p0[1]},  m_above(above){}
    bool operator() (double R, double Z) const
    {
        R -= m_p0[0];
        Z -= m_p0[1];
        double res = m_vec[0]* R + m_vec[1]*Z;
        return m_above == (res > 0); // true if both above and res > 0 or below and res <= 0
    }
    private:
    std::array<double,2> m_p0, m_vec;
    bool m_above;
};
/*! @brief Predicate returning true in closed fieldline region
 */
struct ClosedFieldlineRegion
{
    /// if closed is false then the Functor acts as the OpenFieldLineRegion  = not ClosedFieldlineRegion
    /// If no O-point exists no closed fieldline region exists
    ClosedFieldlineRegion( const TokamakMagneticField& mag, bool closed = true):
        m_psip(mag.psip()), m_closed(closed)
    {
        double RO = mag.R0(), ZO= 0.;
        description desc = mag.params().getDescription();
        if( desc != description::none and desc != description::centeredX)
        {
            dg::geo::findOpoint( mag.get_psip(), RO, ZO);
            m_opoint = true;
        }
        else
        {
            m_opoint = false;
            double psipO = mag.psip()( RO, ZO);
            m_psipO_pos = psipO > 0;
            double RX1 = 0., ZX1 = 0., RX2 = 0., ZX2 = 0.;
            description desc = mag.params().getDescription();
            if( desc != description::none and desc != description::centeredX)
                dg::geo::findOpoint( mag.get_psip(), RO, ZO);
            if ( desc == description::standardX or desc == description::doubleX)
            {
                // Find first X-point
                RX1 = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
                ZX1 = -1.1*mag.params().elongation()*mag.params().a();
                dg::geo::findXpoint( mag.get_psip(), RX1, ZX1);
                m_above.push_back( mod::Above( {RX1, ZX1}, {RO, ZO}));
            }
            if ( desc == description::doubleX)
            {
                // Find second X-point
                RX2 = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
                ZX2 = +1.1*mag.params().elongation()*mag.params().a();
                dg::geo::findXpoint( mag.get_psip(), RX2, ZX2);
                m_above.push_back( mod::Above( {RX2, ZX2}, {RO, ZO}));
            }
        }
    }
    bool operator()( double R, double Z) const
    {
        if( !m_opoint)
            return m_closed ? false : true;
        for( unsigned u=0; u<m_above.size(); u++)
            if( !m_above[u](R,Z))
                return m_closed ? false : true;

        double psip = m_psip(R,Z);
        if( m_psipO_pos == (psip > 0) ) // true if O-point >0 and psip >0 or if O-point <0 and psip <0
            return m_closed ? true : false;
        return m_closed ? false : true;
    }
    private:
    bool m_opoint;
    bool m_psipO_pos;
    std::vector<dg::geo::mod::Above> m_above;
    CylindricalFunctor m_psip;
    bool m_closed;
};

/*! @brief The default predicate for sheath integration
 *
 * The SOL is everything that is not a wall (wall(R,Z) == 1)
 * and not inside the closed Fieldline region
 */
struct SOLRegion
{
    SOLRegion( const TokamakMagneticField& mag, CylindricalFunctor wall): m_wall(wall),
        m_closed(mag){}
    bool operator()( double R, double Z)
    {
        return m_wall(R,Z) != 1 and not m_closed(R,Z);
    }
    private:
    CylindricalFunctor m_wall;
    ClosedFieldlineRegion m_closed;
};


///@endcond

///@addtogroup profiles
///@{

/**
 * @brief \f$ f( f_1(R,Z) , f_2(R,Z)) \f$
 *
 * General composition of two functions
 */
struct SetCompose : public aCylindricalFunctor<SetCompose>
{
    SetCompose( std::function<double(double,double)> fct_mod,
            std::function<double(double,double)> fct1,
            std::function<double(double,double)> fct2) :
        m_fct1(fct1), m_fct2(fct2), m_fct_mod( fct_mod)
    { }
    double do_compute(double R, double Z) const
    {
        return m_fct_mod( m_fct1(R,Z), m_fct2(R,Z));
    }
    private:
    std::function<double(double,double)> m_fct1, m_fct2, m_fct_mod;
};
/**
 * @brief \f$ f_1 + f_2 - f_1 f_2 \equiv f_1 \cup f_2\f$
 *
 * If f_1 and f_2 are functions between 0 and 1 this operation
 * represents the union of two masking regions
 */
struct SetUnion : public aCylindricalFunctor<SetUnion>
{
    SetUnion( std::function<double(double,double)> fct1,
            std::function<double(double,double)> fct2) :
        m_fct1(fct1), m_fct2(fct2)
    { }
    double do_compute(double R, double Z) const
    {
        double f1 = m_fct1(R,Z), f2 = m_fct2( R,Z);
        return f1 + f2 - f1*f2;
    }
    private:
    std::function<double(double,double)> m_fct1, m_fct2;
};
/**
 * @brief \f$ f_1 f_2 \equiv f_1 \cap f_2\f$
 *
 * If f_1 and f_2 are functions between 0 and 1 this operation
 * represents the intersection of two masking regions
 */
struct SetIntersection : public aCylindricalFunctor<SetIntersection>
{
    SetIntersection( std::function<double(double,double)> fct1,
            std::function<double(double,double)> fct2) :
        m_fct1(fct1), m_fct2(fct2)
    { }
    double do_compute(double R, double Z) const
    {
        double f1 = m_fct1(R,Z), f2 = m_fct2( R,Z);
        return f1*f2;
    }
    private:
    std::function<double(double,double)> m_fct1, m_fct2;
};
/**
 * @brief \f$ 1-f \equiv \bar f\f$
 *
 * If f is a function between 0 and 1 this operation
 * represents the negation of the masking region
 */
struct SetNot : public aCylindricalFunctor<SetNot>
{
    SetNot( std::function<double(double,double)> fct) :
        m_fct(fct)
    { }
    double do_compute(double R, double Z) const
    {
        return 1-m_fct(R,Z);
    }
    private:
    std::function<double(double,double)> m_fct;
};

///@}

} //namespace mod

} //namespace geo
} //namespace dg
