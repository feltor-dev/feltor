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

///@brief How flux-function is computed
enum class equilibrium
{
    solovev, //!< dg::geo::solovev::Psip
    taylor, //!< dg::geo::taylor::Psip
    //polynomial, ///!< dg::geo::polynomial::Psip
    guenther, //!< dg::geo::guenther::Psip
    toroidal, //!< dg::geo::toroidal::Psip
    circular //!< dg::geo::circular::Psip
};
///@brief How flux-function is modified
enum class modifier
{
    none, //!< no modification
    heaviside //!< Psip is dampened to a constant outside a critical value
};
///@brief How flux-function looks like
enum class form
{
    standardO, //!< closed flux surfaces centered around an O-point located near (R_0, 0); flux-aligned grids can be constructed
    standardX, //!< closed flux surfaces centered around an O-point located near (R_0, 0) and bordered by a separatrix with a single X-point; flux-aligned X-grids can be constructed
    square, //!< closed flux surfaces centered around an O-point and bordered by a square  with four X-points in the corners (mainly the Guenther field)
    centeredX //!< one X-point in the middle, no O-point, only open flux surfaces, X-grids cannot be constructed
};
///@cond
static const std::map<std::string, equilibrium> str2equilibrium{
    {"solovev", equilibrium::solovev},
    {"taylor", equilibrium::taylor},
    //{"polynomial": equilibrium::polynomial},
    {"guenther", equilibrium::guenther},
    {"toroidal", equilibrium::toroidal},
    {"circular", equilibrium::circular}
};
static const std::map<std::string, modifier> str2modifier{
    {"none", modifier::none},
    {"heaviside", modifier::heaviside}
};
static const std::map<std::string, form> str2form{
    {"standardO", form::standardO},
    {"standardX", form::standardX},
    {"square", form::square},
    {"centeredX", form::centeredX}
};
///@endcond

//Meta-data about magnetic fields
//
/**
 * @brief Meta-data about the magnetic field in particular the flux function
 *
 * The purpose of this is to give a unified set of parameters for
 * all equilibria that can be used to stear program execution.
 *
 * For example it is very hard to automatically detect if the construction
 * of a flux aligned X-grid is possible, but for a human it is very easy.
 * Here we give the \c form specifier that can be used in an if-else statement.
 */
struct MagneticFieldParameters
{
    double a, //!< The minor radius; the purpose of this parameter is not to be exact but to serve as a refernce of how to setup the size of a simulation box
           elongation, //!< (maximum Z - minimum Z of lcfs)/2a; 1 for a circle; the purpose of this parameter is not to be exact but more to be a reference of how to setup the aspect ratio of a simulation box
           triangularity; //!< (R_0 - R_X) /a;  The purpose of this parameter is to find the approximate location of R_X (if an X-point is present, Z_X is given by elongation) the exact location can be computed by the \c findXpoint function
    equilibrium equ; //!< the way the flux function is computed
    modifier mod; //!<  the way the flux function is modified
    form frm; //!< human readable descriptor of how the flux function looks
};

/**
* @brief A tokamak field as given by R0, Psi and Ipol

 This is the representation of toroidally axisymmetric magnetic fields that can be modeled in the form
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
            CylindricalFunctorsLvl1& ipol
            //, MagneticFieldParameters gp
            ): m_R0(R0), m_psip(psip), m_ipol(ipol) {}//, m_params(gp){}
    void set( double R0, const CylindricalFunctorsLvl2& psip, const
            CylindricalFunctorsLvl1& ipol
            //, MagneticFieldParameters gp
            )
    {
        m_R0=R0;
        m_psip=psip;
        m_ipol=ipol;
        //m_params = gp;
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
    //const MagneticFieldParameters& params() const{return m_params;}

    private:
    double m_R0;
    CylindricalFunctorsLvl2 m_psip;
    CylindricalFunctorsLvl1 m_ipol;
    //MagneticFieldParamters m_params;
};

///@cond
CylindricalFunctorsLvl1 periodify( const CylindricalFunctorsLvl1& in, double R0, double R1, double Z0, double Z1, bc bcx, bc bcy)
{
    return CylindricalFunctorsLvl1(
            Periodify( in.f(),   R0, R1, Z0, Z1, bcx, bcy),
            Periodify( in.dfx(), R0, R1, Z0, Z1, inverse(bcx), bcy),
            Periodify( in.dfy(), R0, R1, Z0, Z1, bcx, inverse(bcy)));
}
CylindricalFunctorsLvl2 periodify( const CylindricalFunctorsLvl2& in, double R0, double R1, double Z0, double Z1, bc bcx, bc bcy)
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
 * @brief Use dg::geo::Periodify to periodify every function the magnetic field
 *
 * Note that derivatives are periodified with dg::inverse boundary conditions
 * @param mag The magnetic field to periodify
 * @param R0 left boundary in R
 * @param R1 right boundary in R
 * @param Z0 lower boundary in Z
 * @param Z1 upper boundary in Z
 * @param bcx boundary condition in x (determines how function is periodified)
 * @param bcy boundary condition in y (determines how function is periodified)
 * @note So far this was only tested for Neumann boundary conditions. It is uncertain if Dirichlet boundary conditions work
 *
 * @return new periodified magnetic field
 */
TokamakMagneticField periodify( const TokamakMagneticField& mag, double R0, double R1, double Z0, double Z1, dg::bc bcx, dg::bc bcy)
{
    return TokamakMagneticField( mag.R0(),
            periodify( mag.get_psip(), R0, R1, Z0, Z1, bcx, bcy),
            //what if Dirichlet BC in the current? Won't that generate a NaN?
            periodify( mag.get_ipol(), R0, R1, Z0, Z1, bcx, bcy));
}

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
        double Rn;
        Rn = R/m_mag.R0();
        return -1./R/m_invB(R,Z) + m_invB(R,Z)/Rn/Rn*(m_mag.ipol()(R,Z)*m_mag.ipolR()(R,Z) + m_mag.psipR()(R,Z)*m_mag.psipRR()(R,Z) + m_mag.psipZ()(R,Z)*m_mag.psipRZ()(R,Z));
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
        double Rn;
        Rn = R/m_mag.R0();
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

///@brief \f$ b^\varphi = B^\varphi/|B|\f$
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
 * @brief Contravariant components of the magnetic unit vector field
 * in cylindrical coordinates.
 * @param mag the tokamak magnetic field
 * @return the tuple BHatR, BHatZ, BHatP constructed from mag
 */
inline CylindricalVectorLvl0 createBHat( const TokamakMagneticField& mag){
    return CylindricalVectorLvl0( BHatR(mag), BHatZ(mag), BHatP(mag));
}

/**
 * @brief Contravariant components of the unit vector field (0, 0, +/- 1/R)
 * in cylindrical coordinates.
 * @param sign indicate positive or negative unit vector
 * @return the tuple dg::geo::Constant(0), dg::geo::Constant(0), \f$ 1/R \f$
 * @note This is equivalent to inserting a toroidal magnetic field into the \c dg::geo::createBHat function.
 */
inline CylindricalVectorLvl0 createEPhi( int sign ){
    if( sign > 0)
        return CylindricalVectorLvl0( Constant(0), Constant(0), [](double x, double y){ return 1./x;});
    return CylindricalVectorLvl0( Constant(0), Constant(0), [](double x, double y){ return -1./x;});
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

//Necessary to analytically compute Laplacians:
///@brief \f$ \nabla_\parallel b^R \f$
struct GradBHatR: public aCylindricalFunctor<GradBHatR>
{
    GradBHatR( const TokamakMagneticField& mag): m_bhatR(mag), m_divb(mag), m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        double ipol = m_mag.ipol()(R,Z);
        double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z);
        double psipZZ = m_mag.psipZZ()(R,Z), psipRZ = m_mag.psipRZ()(R,Z);
        return  m_divb(R,Z)*m_bhatR(R,Z) +
                ( psipZ*(psipRZ-psipZ/R) - psipZZ*psipR  )/
                    (ipol*ipol + psipR*psipR + psipZ*psipZ);
    }
    private:
    BHatR m_bhatR;
    Divb m_divb;
    TokamakMagneticField m_mag;
};
///@brief \f$ \nabla_\parallel b^Z \f$
struct GradBHatZ: public aCylindricalFunctor<GradBHatZ>
{
    GradBHatZ( const TokamakMagneticField& mag): m_bhatZ(mag), m_divb(mag), m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        double ipol = m_mag.ipol()(R,Z);
        double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z);
        double psipRR = m_mag.psipRR()(R,Z), psipRZ = m_mag.psipRZ()(R,Z);

        return  m_divb(R,Z)*m_bhatZ(R,Z) +
                (psipR*(psipRZ+psipZ/R) - psipRR*psipZ)/
                    (ipol*ipol + psipR*psipR + psipZ*psipZ);
    }
    private:
    BHatZ m_bhatZ;
    Divb m_divb;
    TokamakMagneticField m_mag;
};
///@brief \f$ \nabla_\parallel b^\varphi \f$
struct GradBHatP: public aCylindricalFunctor<GradBHatP>
{
    GradBHatP( const TokamakMagneticField& mag): m_bhatP(mag), m_divb(mag), m_mag(mag){}
    double do_compute( double R, double Z) const
    {
        double ipol = m_mag.ipol()(R,Z), ipolR = m_mag.ipolR()(R,Z), ipolZ  = m_mag.ipolZ()(R,Z);
        double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z);

        return  m_divb(R,Z)*m_bhatP(R,Z) +
             (psipZ*(ipolR/R - 2.*ipol/R/R) - ipolZ/R*psipR)/
                    (ipol*ipol + psipR*psipR + psipZ*psipZ);
    }
    private:
    BHatP m_bhatP;
    Divb m_divb;
    TokamakMagneticField m_mag;
};

///@brief \f$ \sqrt{\psi_p/ \psi_{p,\min}} \f$
struct RhoP: public aCylindricalFunctor<RhoP>
{
    RhoP( const TokamakMagneticField& mag): m_mag(mag){
        double RO = m_mag.R0(), ZO = 0;
        findOpoint( mag.get_psip(), RO, ZO);
        m_psipmin = m_mag.psip()(RO, ZO);
    }
    double do_compute( double R, double Z) const
    {
        return sqrt( 1.-m_mag.psip()(R,Z)/m_psipmin ) ;
    }
    private:
    double m_psipmin;
    TokamakMagneticField m_mag;

};

///@brief Inertia factor \f$ \mathcal I_0 \f$
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
///@}

} //namespace geo
} //namespace dg

