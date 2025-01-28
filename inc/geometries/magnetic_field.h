#pragma once

#include <map>
#include "fluxfunctions.h"

/*!@file
 *
 * MagneticField objects
 */
namespace dg
{
namespace geo
{
/*!@class hide_toroidal_approximation_note
 @note We implicitly assume the toroidal field line approximation, i.e. all curvature
 and other perpendicular terms assume that the perpendicular
 direction lies within the
 R-Z planes of a cylindrical grid (The plane \f$ \perp \hat{ e}_\varphi\f$ )
 */

///@addtogroup magnetic
///@{

///@brief How flux-function is computed. Decides how to construct magnetic field.
enum class equilibrium
{
    solovev, //!< dg::geo::solovev::Psip
    taylor, //!< dg::geo::taylor::Psip
    polynomial, //!< dg::geo::polynomial::Psip
    guenter, //!< dg::geo::guenter::Psip
    toroidal, //!< dg::geo::createToroidalField
    circular //!< dg::geo::circular::Psip
};
///@brief How flux-function is modified
enum class modifier
{
    none, //!< no modification
    heaviside, //!< Psip is dampened to a constant outside a critical value
    sol_pfr, //!< Psip is dampened in the SOL and PFR regions but not in the closed field line region
    sol_pfr_2X, //!< Psip is dampened in the SOL and PFR regions of each of 2 X-points but not in the closed field line region
    // TODO There should be the "circular" parameter from feltor and should there be a "heavisideX"?
};
/**
 * @brief How flux function looks like. Decider on whether and what flux aligned grid to construct
 *
 * The reason for this enum is that it is very hard to automatically detect if the construction
 * of a flux aligned X-grid is possible, but for a human it is very easy to see.
 */
enum class description
{
    standardO, //!< closed flux surfaces centered around an O-point located near (R_0, 0); flux-aligned grids can be constructed
    standardX, //!< closed flux surfaces centered around an O-point located near (R_0, 0) and bordered by a separatrix with a single X-point; flux-aligned X-grids can be constructed
    doubleX, //!< closed flux surfaces centered around an O-point located near (R_0, 0) and bordered by a separatrix with two X-points; flux-aligned X-grids cannnot be constructed
    none, //!< no shaping: Purely toroidal magnetic field
    square, //!< closed flux surfaces centered around an O-point and bordered by a square  with four X-points in the corners (mainly the Guenter field)
    centeredX //!< one X-point in the middle, no O-point, only open flux surfaces, X-grids cannot be constructed
};
///@cond
inline const std::map<std::string, equilibrium> str2equilibrium{
    {"solovev", equilibrium::solovev},
    {"taylor", equilibrium::taylor},
    {"polynomial", equilibrium::polynomial},
    {"guenter", equilibrium::guenter},
    {"toroidal", equilibrium::toroidal},
    {"circular", equilibrium::circular}
};
inline const std::map<std::string, modifier> str2modifier{
    {"none", modifier::none},
    {"heaviside", modifier::heaviside},
    {"sol_pfr", modifier::sol_pfr},
    {"sol_pfr_2X", modifier::sol_pfr_2X}
};
inline const std::map<std::string, description> str2description{
    {"standardO", description::standardO},
    {"standardX", description::standardX},
    {"doubleX", description::doubleX},
    {"square", description::square},
    {"none", description::none},
    {"centeredX", description::centeredX}
};
///@endcond

//Meta-data about magnetic fields
//
/**
 * @brief Meta-data about the magnetic field in particular the flux function
 *
 * The purpose of this is to give a unified set of parameters for all
 * equilibria that can be used to stear program execution based on
 * characteristics of the magnetic flux functions (for example double X-point
 * vs single X-point vs no X-point)
 */
struct MagneticFieldParameters
{
    /**
     * @brief Default values are for a Toroidal field
     */
    MagneticFieldParameters( ){
        m_a = 1, m_elongation = 1, m_triangularity = 0;
        m_equilibrium = equilibrium::toroidal;
        m_modifier = modifier::none;
        m_description = description::none;
    }
    /**
     * @brief Constructor
     *
     * @param a The minor radius; the purpose of this parameter is not to be exact but to serve as a refernce of how to setup the size of a simulation box
     * @param elongation (maximum Z - minimum Z of lcfs)/2a; 1 for a circle; the purpose of this parameter is not to be exact but more to be a reference of how to setup the aspect ratio of a simulation box
     * @param triangularity (R_0 - R_X) /a;  The purpose of this parameter is to find the approximate location of R_X (if an X-point is present, Z_X is given by elongation) the exact location can be computed by the \c findXpoint function
     * @param equ the way the flux function is computed
     * @param mod the way the flux function is modified
     * @param des human readable descriptor of how the flux function looks
     */
    MagneticFieldParameters( double a, double elongation, double triangularity,
            equilibrium equ, modifier mod, description des): m_a(a),
        m_elongation(elongation),
        m_triangularity( triangularity),
        m_equilibrium( equ),
        m_modifier(mod), m_description( des){}
    /**
     * @brief The minor radius
     *
     * the purpose of this parameter is not to be exact but to serve as a refernce of how to setup the size of a simulation box
     */
    double a() const{return m_a;}
    /**
     * @brief \f$ e := \frac{\max Z_{\mathrm{lcfs}} - \min Z_{\mathrm{lcfs}}}{2a}\f$
     *
     * (1 for a circle); the purpose of this parameter is not to be exact but more to be a reference of how to setup the aspect ratio of a simulation box
     */
    double elongation() const{return m_elongation;}
    /**
     * @brief \f$ \delta := \frac{R_0 - R_X}{a}\f$
     *
     * The purpose of this parameter is to find the approximate location of R_X (if an X-point is present, Z_X is given by elongation) the exact location can be computed by the \c findXpoint function
     */
    double triangularity() const{return m_triangularity;}
    /// the way the flux function is computed
    equilibrium getEquilibrium() const{return m_equilibrium;}
    ///  the way the flux function is modified
    modifier getModifier() const{return m_modifier;}
    /// how the flux function looks
    description getDescription() const{return m_description;}
    private:
    double m_a,
           m_elongation,
           m_triangularity;
    equilibrium m_equilibrium;
    modifier m_modifier;
    description m_description;
};

/**
* @brief A tokamak field as given by R0, Psi and Ipol plus Meta-data like shape and equilibrium

 This is the representation of toroidally axisymmetric magnetic fields that can be modeled in the description
 \f$
 \vec B(R,Z,\varphi) = \frac{R_0}{R} \left( I(\psi_p) \hat e_\varphi + \nabla \psi_p \times \hat e_\varphi\right)
 \f$
 where \f$ R_0\f$ is a normalization constant, \f$ I\f$ the poloidal current
 and \f$ \psi_p\f$ the poloidal flux function.
 @snippet ds_t.cu doxygen
*/
struct TokamakMagneticField
{
    ///as long as the field stays empty the access functions are undefined
    TokamakMagneticField(){}
    TokamakMagneticField( double R0, const CylindricalFunctorsLvl2& psip, const
            CylindricalFunctorsLvl1& ipol , MagneticFieldParameters gp
            ): m_R0(R0), m_psip(psip), m_ipol(ipol), m_params(gp){}
    void set( double R0, const CylindricalFunctorsLvl2& psip, const
            CylindricalFunctorsLvl1& ipol , MagneticFieldParameters gp)
    {
        m_R0=R0;
        m_psip=psip;
        m_ipol=ipol;
        m_params = gp;
    }
    /// \f$ R_0 \f$
    double R0()const {return m_R0;}
    /// \f$ \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const CylindricalFunctor& psip()const{return m_psip.f();}
    /// \f$ \partial_R \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const CylindricalFunctor& psipR()const{return m_psip.dfx();}
    /// \f$ \partial_Z \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const CylindricalFunctor& psipZ()const{return m_psip.dfy();}
    /// \f$ \partial_R\partial_R \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const CylindricalFunctor& psipRR()const{return m_psip.dfxx();}
    /// \f$ \partial_R\partial_Z \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const CylindricalFunctor& psipRZ()const{return m_psip.dfxy();}
    /// \f$ \partial_Z\partial_Z \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const CylindricalFunctor& psipZZ()const{return m_psip.dfyy();}
    /// \f$ I(\psi_p) \f$ the current
    const CylindricalFunctor& ipol()const{return m_ipol.f();}
    /// \f$ \partial_R I(\psi_p) \f$
    const CylindricalFunctor& ipolR()const{return m_ipol.dfx();}
    /// \f$ \partial_Z I(\psi_p) \f$
    const CylindricalFunctor& ipolZ()const{return m_ipol.dfy();}

    const CylindricalFunctorsLvl2& get_psip() const{return m_psip;}
    const CylindricalFunctorsLvl1& get_ipol() const{return m_ipol;}
    /**
     * @brief Access Meta-data of the field
     *
     * @return Meta-data
     */
    const MagneticFieldParameters& params() const{return m_params;}

    private:
    double m_R0;
    CylindricalFunctorsLvl2 m_psip;
    CylindricalFunctorsLvl1 m_ipol;
    MagneticFieldParameters m_params;
};

///@cond
inline CylindricalFunctorsLvl1 periodify( const CylindricalFunctorsLvl1& in, double R0, double R1, double Z0, double Z1, bc bcx, bc bcy)
{
    return CylindricalFunctorsLvl1(
            Periodify( in.f(),   R0, R1, Z0, Z1, bcx, bcy),
            Periodify( in.dfx(), R0, R1, Z0, Z1, inverse(bcx), bcy),
            Periodify( in.dfy(), R0, R1, Z0, Z1, bcx, inverse(bcy)));
}
inline CylindricalFunctorsLvl2 periodify( const CylindricalFunctorsLvl2& in, double R0, double R1, double Z0, double Z1, bc bcx, bc bcy)
{
    return CylindricalFunctorsLvl2(
            Periodify( in.f(),   R0, R1, Z0, Z1, bcx, bcy),
            Periodify( in.dfx(), R0, R1, Z0, Z1, inverse(bcx), bcy),
            Periodify( in.dfy(), R0, R1, Z0, Z1, bcx, inverse(bcy)),
            Periodify( in.dfxx(), R0, R1, Z0, Z1, bcx, bcy),
            Periodify( in.dfxy(), R0, R1, Z0, Z1, inverse(bcx), inverse(bcy)),
            Periodify( in.dfyy(), R0, R1, Z0, Z1, bcx, bcy));
}
///@endcond
/**
 * @brief Use dg::geo::Periodify to periodify every function in the magnetic field
 *
 * Note that derivatives are periodified with dg::inverse boundary conditions
 * @param mag The magnetic field to periodify
 * @param R0 left boundary in R
 * @param R1 right boundary in R
 * @param Z0 lower boundary in Z
 * @param Z1 upper boundary in Z
 * @param bcx boundary condition in x (determines how function is periodified)
 * @param bcy boundary condition in y (determines how function is periodified)
 * @attention So far this was only tested for Neumann boundary conditions. It is uncertain if Dirichlet boundary conditions work
 *
 * @return new periodified magnetic field
 */
inline TokamakMagneticField periodify( const TokamakMagneticField& mag, double R0, double R1, double Z0, double Z1, dg::bc bcx, dg::bc bcy)
{
    return TokamakMagneticField( mag.R0(),
            periodify( mag.get_psip(), R0, R1, Z0, Z1, bcx, bcy),
            //what if Dirichlet BC in the current? Won't that generate a NaN?
            periodify( mag.get_ipol(), R0, R1, Z0, Z1, bcx, bcy), mag.params());
}

///@brief \f$   \Delta\psi_p = \psi_R/R + \psi_{RR}+\psi_{ZZ}   \f$
struct LaplacePsip : public aCylindricalFunctor<LaplacePsip>
{
    LaplacePsip( const TokamakMagneticField& mag): m_mag(mag)  { }
    double do_compute(double R, double Z) const
    {
        return m_mag.psipR()(R,Z)/R+ m_mag.psipRR()(R,Z) + m_mag.psipZZ()(R,Z);
    }
  private:
    TokamakMagneticField m_mag;
};

///@brief \f$   |B| = R_0\sqrt{I^2+(\nabla\psi)^2}/R   \f$
struct Bmodule : public aCylindricalFunctor<Bmodule>
{
    Bmodule( const TokamakMagneticField& mag): m_mag(mag)  { }
    double do_compute(double R, double Z) const
    {
        double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z), ipol = m_mag.ipol()(R,Z);
        return m_mag.R0()/R*sqrt(ipol*ipol+psipR*psipR +psipZ*psipZ);
    }
  private:
    TokamakMagneticField m_mag;
};

/**
 * @brief \f$  |B|^{-1} = R/R_0\sqrt{I^2+(\nabla\psi)^2}    \f$

    \f$   \frac{1}{\hat{B}} =
        \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
        + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f$
 */
struct InvB : public aCylindricalFunctor<InvB>
{
    InvB(  const TokamakMagneticField& mag): m_mag(mag){ }
    double do_compute(double R, double Z) const
    {
        double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z), ipol = m_mag.ipol()(R,Z);
        return R/(m_mag.R0()*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
    }
  private:
    TokamakMagneticField m_mag;
};

/**
 * @brief \f$   \ln{|B|}  \f$
 *
   \f$   \ln{(   \hat{B})} = \ln{\left[
          \frac{\hat{R}_0}{\hat{R}} \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
          + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2} \right] } \f$
 */
struct LnB : public aCylindricalFunctor<LnB>
{
    LnB(const TokamakMagneticField& mag): m_mag(mag) { }
    double do_compute(double R, double Z) const
    {
        double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z), ipol = m_mag.ipol()(R,Z);
        return log(m_mag.R0()/R*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
    }
  private:
    TokamakMagneticField m_mag;
};

/**
 * @brief \f$  \frac{\partial |B| }{ \partial R}  \f$
 *
 \f$  \frac{\partial \hat{B} }{ \partial \hat{R}} =
      -\frac{1}{\hat B \hat R}
      +  \frac{\hat I \left(\frac{\partial\hat I}{\partial\hat R} \right)
      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\right)
      + \left( \frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)\left( \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\right)}
      {\hat{R}^2 \hat{R}_0^{-2}\hat{B}} \f$
 */
struct BR: public aCylindricalFunctor<BR>
{
    BR(const TokamakMagneticField& mag): m_invB(mag), m_mag(mag) { }
    double do_compute(double R, double Z) const
    {
        double Rn = R/m_mag.R0();
        double invB = m_invB(R,Z);
        return -1./R/invB + invB/Rn/Rn*(m_mag.ipol()(R,Z)*m_mag.ipolR()(R,Z) + m_mag.psipR()(R,Z)*m_mag.psipRR()(R,Z) + m_mag.psipZ()(R,Z)*m_mag.psipRZ()(R,Z));
    }
  private:
    InvB m_invB;
    TokamakMagneticField m_mag;
};

/**
 * @brief \f$  \frac{\partial |B| }{ \partial Z}  \f$
 *
  \f$  \frac{\partial \hat{B} }{ \partial \hat{Z}} =
     \frac{ \hat I \left(\frac{\partial \hat I}{\partial\hat Z}    \right)+
     \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\right)
          + \left( \frac{\partial \hat{\psi}_p }{ \partial \hat{Z}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2} \right)}{\hat{R}^2 \hat{R}_0^{-2}\hat{B}} \f$
 */
struct BZ: public aCylindricalFunctor<BZ>
{
    BZ(const TokamakMagneticField& mag ): m_mag(mag), m_invB(mag) { }
    double do_compute(double R, double Z) const
    {
        double Rn = R/m_mag.R0();
        return (m_invB(R,Z)/Rn/Rn)*(m_mag.ipol()(R,Z)*m_mag.ipolZ()(R,Z) + m_mag.psipR()(R,Z)*m_mag.psipRZ()(R,Z) + m_mag.psipZ()(R,Z)*m_mag.psipZZ()(R,Z));
    }
  private:
    TokamakMagneticField m_mag;
    InvB m_invB;
};

///@brief Approximate \f$ \mathcal{K}^{R}_{\nabla B} \f$
///
/// \f$ \mathcal{\hat{K}}^{\hat{R}}_{\nabla B} =-\frac{1}{ \hat{B}^2}  \frac{\partial \hat{B}}{\partial \hat{Z}}  \f$
///@copydoc hide_toroidal_approximation_note
struct CurvatureNablaBR: public aCylindricalFunctor<CurvatureNablaBR>
{
    CurvatureNablaBR(const TokamakMagneticField& mag, int sign): m_invB(mag), m_bZ(mag) {
        if( sign >0)
            m_sign = +1.;
        else
            m_sign = -1;
    }
    double do_compute( double R, double Z) const
    {
        return -m_sign*m_invB(R,Z)*m_invB(R,Z)*m_bZ(R,Z);
    }
    private:
    double m_sign;
    InvB m_invB;
    BZ m_bZ;
};

///@brief Approximate \f$  \mathcal{K}^{Z}_{\nabla B}  \f$
///
/// \f$  \mathcal{\hat{K}}^{\hat{Z}}_{\nabla B} =\frac{1}{ \hat{B}^2}   \frac{\partial \hat{B}}{\partial \hat{R}} \f$
///@copydoc hide_toroidal_approximation_note
struct CurvatureNablaBZ: public aCylindricalFunctor<CurvatureNablaBZ>
{
    CurvatureNablaBZ( const TokamakMagneticField& mag, int sign): m_invB(mag), m_bR(mag) {
        if( sign >0)
            m_sign = +1.;
        else
            m_sign = -1;
    }
    double do_compute( double R, double Z) const
    {
        return m_sign*m_invB(R,Z)*m_invB(R,Z)*m_bR(R,Z);
    }
    private:
    double m_sign;
    InvB m_invB;
    BR m_bR;
};

///@brief Approximate \f$ \mathcal{K}^{R}_{\vec{\kappa}}=0 \f$
///
/// \f$ \mathcal{\hat{K}}^{\hat{R}}_{\vec{\kappa}} =0  \f$
///@copydoc hide_toroidal_approximation_note
struct CurvatureKappaR: public aCylindricalFunctor<CurvatureKappaR>
{
    CurvatureKappaR( ){ }
    CurvatureKappaR( const TokamakMagneticField& mag, int sign = +1){ }
    double do_compute( double R, double Z) const
    {
        return  0.;
    }
    private:
};

///@brief Approximate \f$  \mathcal{K}^{Z}_{\vec{\kappa}}  \f$
///
/// \f$  \mathcal{\hat{K}}^{\hat{Z}}_{\vec{\kappa}} = - \frac{1}{\hat{R} \hat{B}} \f$
///@copydoc hide_toroidal_approximation_note
struct CurvatureKappaZ: public aCylindricalFunctor<CurvatureKappaZ>
{
    CurvatureKappaZ( const TokamakMagneticField& mag, int sign): m_invB(mag) {
        if( sign >0)
            m_sign = +1.;
        else
            m_sign = -1;
    }
    double do_compute( double R, double Z) const
    {
        return -m_sign*m_invB(R,Z)/R;
    }
    private:
    double m_sign;
    InvB m_invB;
};

///@brief Approximate \f$  \vec{\nabla}\cdot \mathcal{K}_{\vec{\kappa}}  \f$
///
///  \f$  \vec{\hat{\nabla}}\cdot \mathcal{\hat{K}}_{\vec{\kappa}}  = \frac{1}{\hat{R}  \hat{B}^2 } \partial_{\hat{Z}} \hat{B}\f$
///@copydoc hide_toroidal_approximation_note
struct DivCurvatureKappa: public aCylindricalFunctor<DivCurvatureKappa>
{
    DivCurvatureKappa( const TokamakMagneticField& mag, int sign): m_invB(mag), m_bZ(mag){
        if( sign >0)
            m_sign = +1.;
        else
            m_sign = -1;
    }
    double do_compute( double R, double Z) const
    {
        return m_sign*m_bZ(R,Z)*m_invB(R,Z)*m_invB(R,Z)/R;
    }
    private:
    double m_sign;
    InvB m_invB;
    BZ m_bZ;
};

///@brief Approximate \f$  \vec{\nabla}\cdot \mathcal{K}_{\nabla B}  \f$
///
///  \f$  \vec{\hat{\nabla}}\cdot \mathcal{\hat{K}}_{\nabla B}  = -\frac{1}{\hat{R}  \hat{B}^2 } \partial_{\hat{Z}} \hat{B}\f$
///@copydoc hide_toroidal_approximation_note
struct DivCurvatureNablaB: public aCylindricalFunctor<DivCurvatureNablaB>
{
    DivCurvatureNablaB( const TokamakMagneticField& mag, int sign): m_div(mag, sign){ }
    double do_compute( double R, double Z) const
    {
        return -m_div(R,Z);
    }
    private:
    DivCurvatureKappa m_div;
};
///@brief True \f$ \mathcal{K}^{R}_{\nabla B} \f$
///
/// \f$ \mathcal{K}^R_{\nabla B} =-\frac{R_0I}{ B^3R}  \frac{\partial B}{\partial Z}  \f$
struct TrueCurvatureNablaBR: public aCylindricalFunctor<TrueCurvatureNablaBR>
{
    TrueCurvatureNablaBR(const TokamakMagneticField& mag): m_R0(mag.R0()), m_mag(mag), m_invB(mag), m_bZ(mag) { }
    double do_compute( double R, double Z) const
    {
        double invB = m_invB(R,Z), ipol = m_mag.ipol()(R,Z);
        return -invB*invB*invB*ipol*m_R0/R*m_bZ(R,Z);
    }
    private:
    double m_R0;
    TokamakMagneticField m_mag;
    InvB m_invB;
    BZ m_bZ;
};

///@brief True \f$ \mathcal{K}^{Z}_{\nabla B} \f$
///
/// \f$ \mathcal{K}^Z_{\nabla B} =\frac{R_0I}{ B^3R}  \frac{\partial B}{\partial R}  \f$
struct TrueCurvatureNablaBZ: public aCylindricalFunctor<TrueCurvatureNablaBZ>
{
    TrueCurvatureNablaBZ(const TokamakMagneticField& mag): m_R0(mag.R0()), m_mag(mag), m_invB(mag), m_bR(mag) { }
    double do_compute( double R, double Z) const
    {
        double invB = m_invB(R,Z), ipol = m_mag.ipol()(R,Z);
        return invB*invB*invB*ipol*m_R0/R*m_bR(R,Z);
    }
    private:
    double m_R0;
    TokamakMagneticField m_mag;
    InvB m_invB;
    BR m_bR;
};

///@brief True \f$ \mathcal{K}^{\varphi}_{\nabla B} \f$
///
/// \f$ \mathcal{K}^\varphi_{\nabla B} =\frac{1}{ B^3R^2}\left( \frac{\partial\psi}{\partial Z} \frac{\partial B}{\partial Z} + \frac{\partial \psi}{\partial R}\frac{\partial B}{\partial R} \right) \f$
struct TrueCurvatureNablaBP: public aCylindricalFunctor<TrueCurvatureNablaBP>
{
    TrueCurvatureNablaBP(const TokamakMagneticField& mag): m_mag(mag), m_invB(mag),m_bR(mag), m_bZ(mag) { }
    double do_compute( double R, double Z) const
    {
        double invB = m_invB(R,Z);
        return m_mag.R0()*invB*invB*invB/R/R*(m_mag.psipZ()(R,Z)*m_bZ(R,Z) + m_mag.psipR()(R,Z)*m_bR(R,Z));
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;
    BR m_bR;
    BZ m_bZ;
};

///@brief True \f$ \mathcal{K}^R_{\vec{\kappa}} \f$
struct TrueCurvatureKappaR: public aCylindricalFunctor<TrueCurvatureKappaR>
{
    TrueCurvatureKappaR( const TokamakMagneticField& mag):m_mag(mag), m_invB(mag), m_bZ(mag){ }
    double do_compute( double R, double Z) const
    {
        double invB = m_invB(R,Z);
        return m_mag.R0()*invB*invB/R*(m_mag.ipolZ()(R,Z) - m_mag.ipol()(R,Z)*invB*m_bZ(R,Z));
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;
    BZ m_bZ;
};

///@brief True \f$ \mathcal{K}^Z_{\vec{\kappa}} \f$
struct TrueCurvatureKappaZ: public aCylindricalFunctor<TrueCurvatureKappaZ>
{
    TrueCurvatureKappaZ( const TokamakMagneticField& mag):m_mag(mag), m_invB(mag), m_bR(mag){ }
    double do_compute( double R, double Z) const
    {
        double invB = m_invB(R,Z);
        return m_mag.R0()*invB*invB/R*( - m_mag.ipolR()(R,Z) + m_mag.ipol()(R,Z)*invB*m_bR(R,Z));
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;
    BR m_bR;
};
///@brief True \f$ \mathcal{K}^\varphi_{\vec{\kappa}} \f$
struct TrueCurvatureKappaP: public aCylindricalFunctor<TrueCurvatureKappaP>
{
    TrueCurvatureKappaP( const TokamakMagneticField& mag):m_mag(mag), m_invB(mag), m_bR(mag), m_bZ(mag){ }
    double do_compute( double R, double Z) const
    {
        double invB = m_invB(R,Z);
        return m_mag.R0()*invB*invB/R/R*(
            + invB*m_mag.psipZ()(R,Z)*m_bZ(R,Z) + invB *m_mag.psipR()(R,Z)*m_bR(R,Z)
            + m_mag.psipR()(R,Z)/R - m_mag.psipRR()(R,Z) - m_mag.psipZZ()(R,Z));
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;
    BR m_bR;
    BZ m_bZ;
};

///@brief True \f$  \vec{\nabla}\cdot \mathcal{K}_{\vec{\kappa}}  \f$
struct TrueDivCurvatureKappa: public aCylindricalFunctor<TrueDivCurvatureKappa>
{
    TrueDivCurvatureKappa( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag), m_bR(mag), m_bZ(mag){}
    double do_compute( double R, double Z) const
    {
        double invB = m_invB(R,Z);
        return m_mag.R0()*invB*invB*invB/R*( m_mag.ipolR()(R,Z)*m_bZ(R,Z) - m_mag.ipolZ()(R,Z)*m_bR(R,Z) );
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;
    BR m_bR;
    BZ m_bZ;
};

///@brief True \f$  \vec{\nabla}\cdot \mathcal{K}_{\nabla B}  \f$
struct TrueDivCurvatureNablaB: public aCylindricalFunctor<TrueDivCurvatureNablaB>
{
    TrueDivCurvatureNablaB( const TokamakMagneticField& mag): m_div(mag){}
    double do_compute( double R, double Z) const {
        return - m_div(R,Z);
    }
    private:
    TrueDivCurvatureKappa m_div;
};

/**
 * @brief \f$  \nabla_\parallel \ln{(B)} \f$
 *
 *    \f$  \hat{\nabla}_\parallel \ln{(\hat{B})} = \frac{1}{\hat{R}\hat{B}^2 } \left[ \hat{B}, \hat{\psi}_p\right]_{\hat{R}\hat{Z}} \f$
 */
struct GradLnB: public aCylindricalFunctor<GradLnB>
{
    GradLnB( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag), m_bR(mag), m_bZ(mag) { }
    double do_compute( double R, double Z) const
    {
        double invB = m_invB(R,Z);
        return m_mag.R0()*invB*invB*(m_bR(R,Z)*m_mag.psipZ()(R,Z)-m_bZ(R,Z)*m_mag.psipR()(R,Z))/R ;
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;
    BR m_bR;
    BZ m_bZ;
};
/**
 * @brief \f$  \nabla \cdot \vec b \f$
 *
 * \f$  \nabla\cdot \vec b = -\nabla_\parallel \ln B \f$
 * @sa \c GradLnB
 */
struct Divb: public aCylindricalFunctor<Divb>
{
    Divb( const TokamakMagneticField& mag): m_gradLnB(mag) { }
    double do_compute( double R, double Z) const
    {
        return -m_gradLnB(R,Z);
    }
    private:
    GradLnB m_gradLnB;
};

///@brief \f$ B^\varphi = R_0I/R^2\f$
struct BFieldP: public aCylindricalFunctor<BFieldP>
{
    BFieldP( const TokamakMagneticField& mag): m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        return m_mag.R0()*m_mag.ipol()(R,Z)/R/R;
    }
    private:

    TokamakMagneticField m_mag;
};

///@brief \f$ B^R = R_0\psi_Z /R\f$
struct BFieldR: public aCylindricalFunctor<BFieldR>
{
    BFieldR( const TokamakMagneticField& mag): m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        return  m_mag.R0()/R*m_mag.psipZ()(R,Z);
    }
    private:
    TokamakMagneticField m_mag;

};

///@brief \f$ B^Z = -R_0\psi_R /R\f$
struct BFieldZ: public aCylindricalFunctor<BFieldZ>
{
    BFieldZ( const TokamakMagneticField& mag): m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        return -m_mag.R0()/R*m_mag.psipR()(R,Z);
    }
    private:
    TokamakMagneticField m_mag;
};

///@brief \f$  B^{\theta} = B^R\partial_R\theta + B^Z\partial_Z\theta\f$
struct BFieldT: public aCylindricalFunctor<BFieldT>
{
    BFieldT( const TokamakMagneticField& mag):  m_R0(mag.R0()), m_fieldR(mag), m_fieldZ(mag){}
    double do_compute(double R, double Z) const
    {
        double r2 = (R-m_R0)*(R-m_R0) + Z*Z;
        return m_fieldR(R,Z)*(-Z/r2) + m_fieldZ(R,Z)*(R-m_R0)/r2;
    }
    private:
    double m_R0;
    BFieldR m_fieldR;
    BFieldZ m_fieldZ;
};

///@brief \f$ b^R = B^R/|B|\f$
struct BHatR: public aCylindricalFunctor<BHatR>
{
    BHatR( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag){ }
    double do_compute( double R, double Z) const
    {
        return  m_invB(R,Z)*m_mag.R0()/R*m_mag.psipZ()(R,Z);
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;

};

///@brief \f$ b^Z = B^Z/|B|\f$
struct BHatZ: public aCylindricalFunctor<BHatZ>
{
    BHatZ( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag){ }
    double do_compute( double R, double Z) const
    {
        return  -m_invB(R,Z)*m_mag.R0()/R*m_mag.psipR()(R,Z);
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;
};

///@brief \f$ \hat b^\varphi = B^\varphi/|B|\f$
struct BHatP: public aCylindricalFunctor<BHatP>
{
    BHatP( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag){ }
    double do_compute( double R, double Z) const
    {
        return m_invB(R,Z)*m_mag.R0()*m_mag.ipol()(R,Z)/R/R;
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;
};


/**
 * @brief Contravariant components of the unit vector field (0, 0, \f$\pm 1/R \f$)
 * and its Divergence and derivative (0,0)
 * in cylindrical coordinates.
 * @param sign indicate positive or negative unit vector
 * @return the tuple dg::geo::Constant(0), dg::geo::Constant(0), \f$ \pm 1/R \f$
 * @note This is equivalent to inserting a toroidal magnetic field into the \c dg::geo::createBHat function.
 */
inline CylindricalVectorLvl1 createEPhi( int sign ){
    if( sign > 0)
        return CylindricalVectorLvl1( Constant(0), Constant(0), [](double x, double y){ return 1./x;}, Constant(0), Constant(0));
    return CylindricalVectorLvl1( Constant(0), Constant(0), [](double x, double y){ return -1./x;}, Constant(0), Constant(0));
}
/**
 * @brief Approximate curvature vector field (CurvatureNablaBR, CurvatureNablaBZ, Constant(0))
 *
 * @param mag the tokamak magnetic field
 * @param sign indicate positive or negative unit vector in approximation
 * @return the tuple \c CurvatureNablaBR, \c CurvatureNablaBZ, \c dg::geo::Constant(0) constructed from \c mag
 * @note The contravariant components in cylindrical coordinates
 */
inline CylindricalVectorLvl0 createCurvatureNablaB( const TokamakMagneticField& mag, int sign){
    return CylindricalVectorLvl0( CurvatureNablaBR(mag, sign), CurvatureNablaBZ(mag, sign), Constant(0));
}
/**
 * @brief Approximate curvature vector field (CurvatureKappaR, CurvatureKappaZ, Constant(0))
 *
 * @param mag the tokamak magnetic field
 * @param sign indicate positive or negative unit vector in approximation
 * @return the tuple \c CurvatureKappaR, \c CurvatureKappaZ, \c dg::geo::Constant(0) constructed from \c mag
 * @note The contravariant components in cylindrical coordinates
 */
inline CylindricalVectorLvl0 createCurvatureKappa( const TokamakMagneticField& mag, int sign){
    return CylindricalVectorLvl0( CurvatureKappaR(mag, sign), CurvatureKappaZ(mag, sign), Constant(0));
}
/**
 * @brief True curvature vector field (TrueCurvatureKappaR, TrueCurvatureKappaZ, TrueCurvatureKappaP)
 *
 * @param mag the tokamak magnetic field
 * @return the tuple TrueCurvatureKappaR, TrueCurvatureKappaZ, TrueCurvatureKappaP constructed from mag
 * @note The contravariant components in cylindrical coordinates
 */
inline CylindricalVectorLvl0 createTrueCurvatureKappa( const TokamakMagneticField& mag){
    return CylindricalVectorLvl0( TrueCurvatureKappaR(mag), TrueCurvatureKappaZ(mag), TrueCurvatureKappaP(mag));
}
/**
 * @brief True curvature vector field (TrueCurvatureNablaBR, TrueCurvatureNablaBZ, TrueCurvatureNablaBP)
 *
 * @param mag the tokamak magnetic field
 * @return the tuple TrueCurvatureNablaBR, TrueCurvatureNablaBZ, TrueCurvatureNablaBP constructed from mag
 * @note The contravariant components in cylindrical coordinates
 */
inline CylindricalVectorLvl0 createTrueCurvatureNablaB( const TokamakMagneticField& mag){
    return CylindricalVectorLvl0( TrueCurvatureNablaBR(mag), TrueCurvatureNablaBZ(mag), TrueCurvatureNablaBP(mag));
}
/**
 * @brief Gradient Psip vector field (PsipR, PsipZ, 0)
 *
 * @param mag the tokamak magnetic field
 * @return the tuple PsipR, PsipZ, 0 constructed from mag
 * @note The contravariant components in cylindrical coordinates
 */
inline CylindricalVectorLvl0 createGradPsip( const TokamakMagneticField& mag){
    return CylindricalVectorLvl0( mag.psipR(), mag.psipZ(),Constant(0));
}


///@brief \f$ \partial_R b^R\f$
struct BHatRR: public aCylindricalFunctor<BHatRR>
{
    BHatRR( const TokamakMagneticField& mag): m_invB(mag), m_br(mag), m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        double psipZ = m_mag.psipZ()(R,Z);
        double psipRZ = m_mag.psipRZ()(R,Z);
        double binv = m_invB(R,Z);
        return -psipZ*m_mag.R0()*binv/R/R + psipRZ*binv*m_mag.R0()/R
            -psipZ*m_mag.R0()/R*binv*binv*m_br(R,Z);
    }
    private:
    InvB m_invB;
    BR m_br;
    TokamakMagneticField m_mag;
};
///@brief \f$ \partial_Z b^R\f$
struct BHatRZ: public aCylindricalFunctor<BHatRZ>
{
    BHatRZ( const TokamakMagneticField& mag): m_invB(mag), m_bz(mag), m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        double psipZ = m_mag.psipZ()(R,Z);
        double psipZZ = m_mag.psipZZ()(R,Z);
        double binv = m_invB(R,Z);
        return m_mag.R0()/R*( psipZZ*binv  -binv*binv*m_bz(R,Z)*psipZ );
    }
    private:
    InvB m_invB;
    BZ m_bz;
    TokamakMagneticField m_mag;
};
///@brief \f$ \partial_R b^Z\f$
struct BHatZR: public aCylindricalFunctor<BHatZR>
{
    BHatZR( const TokamakMagneticField& mag): m_invB(mag), m_br(mag), m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        double psipR = m_mag.psipR()(R,Z);
        double psipRR = m_mag.psipRR()(R,Z);
        double binv = m_invB(R,Z);
        return +psipR*m_mag.R0()*binv/R/R - psipRR*binv*m_mag.R0()/R
            +psipR*m_mag.R0()/R*binv*binv*m_br(R,Z);
    }
    private:
    InvB m_invB;
    BR m_br;
    TokamakMagneticField m_mag;
};
///@brief \f$ \partial_Z b^Z\f$
struct BHatZZ: public aCylindricalFunctor<BHatZZ>
{
    BHatZZ( const TokamakMagneticField& mag): m_invB(mag), m_bz(mag), m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        double psipR = m_mag.psipR()(R,Z);
        double psipRZ = m_mag.psipRZ()(R,Z);
        double binv = m_invB(R,Z);
        return -m_mag.R0()/R*( psipRZ*binv  -binv*binv*m_bz(R,Z)*psipR );
    }
    private:
    InvB m_invB;
    BZ m_bz;
    TokamakMagneticField m_mag;
};
///@brief \f$ \partial_R \hat b^\varphi\f$
struct BHatPR: public aCylindricalFunctor<BHatPR>
{
    BHatPR( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag), m_br(mag){ }
    double do_compute( double R, double Z) const
    {
        double binv = m_invB(R,Z);
        double ipol = m_mag.ipol()(R,Z);
        double ipolR = m_mag.ipolR()(R,Z);
        return -binv*binv*m_br(R,Z)*m_mag.R0()*ipol/R/R
            - 2./R/R/R*binv*m_mag.R0()*ipol
            + binv *m_mag.R0()/R/R*ipolR;
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;
    BR m_br;
};
///@brief \f$ \partial_Z \hat b^\varphi\f$
struct BHatPZ: public aCylindricalFunctor<BHatPZ>
{
    BHatPZ( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag), m_bz(mag){ }
    double do_compute( double R, double Z) const
    {
        double binv = m_invB(R,Z);
        double ipol = m_mag.ipol()(R,Z);
        double ipolZ = m_mag.ipolZ()(R,Z);
        return -binv*binv*m_bz(R,Z)*m_mag.R0()*ipol/R/R
            + binv *m_mag.R0()/R/R*ipolZ;
    }
    private:
    TokamakMagneticField m_mag;
    InvB m_invB;
    BZ m_bz;
};

///@brief \f$ \nabla\cdot\left( \frac{ \hat b }{\hat b^\varphi}\right)\f$
struct DivVVP: public aCylindricalFunctor<DivVVP>
{
    DivVVP( const TokamakMagneticField& mag): m_mag(mag),
        m_bhatP(mag){ }
    double do_compute( double R, double Z) const
    {
        double ipol = m_mag.ipol()(R,Z), ipolR = m_mag.ipolR()(R,Z),
               ipolZ  = m_mag.ipolZ()(R,Z);
        double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z);
        double bphi = m_bhatP(R,Z);
        return -(psipZ*(ipolR/R - 2.*ipol/R/R) - ipolZ/R*psipR)/
                     (ipol*ipol + psipR*psipR + psipZ*psipZ)/bphi/bphi;
    }
    private:
    TokamakMagneticField m_mag;
    BHatP m_bhatP;
};

/**
 * @brief Contravariant components of the magnetic unit vector field
 * and its Divergence and derivative in cylindrical coordinates.
 * @param mag the tokamak magnetic field
 * @return the tuple BHatR, BHatZ, BHatP, Divb, DivVVP constructed from mag
 */
inline CylindricalVectorLvl1 createBHat( const TokamakMagneticField& mag){
    return CylindricalVectorLvl1( BHatR(mag), BHatZ(mag), BHatP(mag),
            Divb(mag), DivVVP(mag)
           );
}

/**
 * @brief \f$ \sqrt{1. - \psi_p/ \psi_{p,O}} \f$
 *
 * @attention undefined if there is no O-point near [R_0 , 0], except for
 * \c description::centeredX when we take psipO = -10
 */
struct RhoP: public aCylindricalFunctor<RhoP>
{
    RhoP( const TokamakMagneticField& mag): m_mag(mag){
        double RO = m_mag.R0(), ZO = 0;
        try{
            findOpoint( mag.get_psip(), RO, ZO);
            m_psipmin = m_mag.psip()(RO, ZO);
        } catch ( dg::Error& err)
        {
            m_psipmin = 1.;
            if( mag.params().getDescription() == description::centeredX)
                m_psipmin = -10;
        }
    }
    double do_compute( double R, double Z) const
    {
        return sqrt( 1.-m_mag.psip()(R,Z)/m_psipmin ) ;
    }
    private:
    double m_psipmin;
    TokamakMagneticField m_mag;

};

///@brief Inertia factor \f$ \mathcal I_0 = B^2/(R_0^2|\nabla\psi_p|^2)\f$
struct Hoo : public dg::geo::aCylindricalFunctor<Hoo>
{
    Hoo( dg::geo::TokamakMagneticField mag): m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z), ipol = m_mag.ipol()(R,Z);
        double psip2 = psipR*psipR+psipZ*psipZ;
        if( psip2 == 0)
            psip2 = 1e-16;
        return (ipol*ipol + psip2)/R/R/psip2;
    }
    private:
    dg::geo::TokamakMagneticField m_mag;
};

///@brief Determine if poloidal field points towards or away from the nearest wall
///
///@attention Does not account for toroidal field direction
struct WallDirection : public dg::geo::aCylindricalFunctor<WallDirection>
{
    /**
     * @brief Allocate lines
     *
     * @param mag Use to construct magnetic field
     * @param vertical walls R_0, R_1 ...  ( can be arbitrary size)
     * @param horizontal walls Z_0, Z_1 ... ( can be arbitrary size)
     */
    WallDirection( dg::geo::TokamakMagneticField mag, std::vector<double>
            vertical, std::vector<double> horizontal) : m_vertical(vertical),
        m_horizontal(horizontal), m_BR( mag), m_BZ(mag){}
    /**
     * @brief Allocate lines
     *
     * @param mag Use to construct magnetic field
     * @param walls two vertical x0, x1 and two horizontal y0, y1 walls
     */
    WallDirection( dg::geo::TokamakMagneticField mag,
            dg::Grid2d walls) : m_vertical({walls.x0(), walls.x1()}),
        m_horizontal({walls.y0(), walls.y1()}), m_BR( mag), m_BZ(mag){}
    double do_compute ( double R, double Z) const
    {
        std::vector<double> v_dist(1,1e100), h_dist(1,1e100);
        for( auto v : m_vertical)
            v_dist.push_back( R-v );
        for( auto h : m_horizontal)
            h_dist.push_back( Z-h );
        double v_min = *std::min_element( v_dist.begin(), v_dist.end(),
                [](double a, double b){ return fabs(a) < fabs(b);} );
        double h_min = *std::min_element( h_dist.begin(), h_dist.end(),
                [](double a, double b){ return fabs(a) < fabs(b);} );
        if( fabs(v_min) < fabs(h_min) ) // if vertical R wall is closer
        {
            double br = m_BR( R,Z);
            return v_min*br < 0 ? +1 : -1;
        }
        else //horizontal Z wall is closer
        {
            double bz = m_BZ( R,Z);
            return h_min*bz < 0 ? +1 : -1;
        }
    }
    private:
    std::vector<double> m_vertical, m_horizontal;
    dg::geo::BFieldR m_BR;
    dg::geo::BFieldZ m_BZ;
};
///@}

} //namespace geo
} //namespace dg

