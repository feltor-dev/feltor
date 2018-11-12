#pragma once
//#include "solovev.h"
#include "fluxfunctions.h"
#include "solovev_parameters.h"

/*!@file
 *
 * Initialize and Damping objects
 */
namespace dg
{
namespace geo
{
///@addtogroup profiles
///@{
//
/**
 * @brief Returns zero outside psipmax and inside psipmin, otherwise 1
     \f[ \begin{cases}
        1  \text{ if } \psi_{p,min} < \psi_p(R,Z) < \psi_{p,max}\\
        0  \text{ else}
     \end{cases}\f]
 */
struct Iris : public aCloneableCylindricalFunctor<Iris>
{
    Iris( CylindricalFunctor psi, double psi_min, double psi_max ):
        psip_(psi), psipmin_(psi_min), psipmax_(psi_max) { }
    private:
    double do_compute(double R, double Z)const
    {
        double psip = psip_(R,Z);
        if( psip > psipmax_) return 0.;
        if( psip < psipmin_) return 0.;
        return 1.;
    }
    CylindricalFunctor psip_;
    double psipmin_, psipmax_;
};
/**
 * @brief Returns zero outside psipmaxcut otherwise 1
     \f[ \begin{cases}
        0  \text{ if } \psi_p(R,Z) > \psi_{p,maxcut} \\
        1  \text{ else}
     \end{cases}\f]
 */
struct Pupil : public aCloneableCylindricalFunctor<Pupil>
{
    Pupil( CylindricalFunctor psi, double psipmaxcut):
        psip_(psi), psipmaxcut_(psipmaxcut) { }
    private:
    double do_compute(double R, double Z)const
    {
        if( psip_(R,Z) > psipmaxcut_) return 0.;
        return 1.;
    }
    CylindricalFunctor psip_;
    double psipmaxcut_;
};
/**
 * @brief Returns psi inside psipmax and psipmax outside psipmax
     \f[ \begin{cases}
        \psi_{p,max}  \text{ if } \psi_p(R,Z) > \psi_{p,max} \\
        \psi_p(R,Z) \text{ else}
     \end{cases}\f]
 */
struct PsiPupil : public aCloneableCylindricalFunctor<PsiPupil>
{
    PsiPupil(CylindricalFunctor psi, double psipmax):
        psipmax_(psipmax), psip_(psi) { }
    private:
    double do_compute(double R, double Z)const
    {
        if( psip_(R,Z) > psipmax_) return psipmax_;
        return  psip_(R,Z);
    }
    double psipmax_;
    CylindricalFunctor psip_;
};
/**
 * @brief Sets values to one outside psipmaxcut, zero else
     \f[ \begin{cases}
        1  \text{ if } \psi_p(R,Z) > \psi_{p,maxlim} \\
        0  \text{ else}
     \end{cases}\f]
 *
 */
struct PsiLimiter : public aCloneableCylindricalFunctor<PsiLimiter>
{
    PsiLimiter( CylindricalFunctor psi, double psipmaxlim):
        psipmaxlim_(psipmaxlim), psip_(psi) { }

    private:
    double do_compute(double R, double Z)const
    {
        if( psip_(R,Z) > psipmaxlim_) return 1.;
        return 0.;
    }
    double psipmaxlim_;
    CylindricalFunctor psip_;
};

/*!@brief Cut everything below X-point
 \f[ \begin{cases}
 1 \text{ if } Z > Z_X \\
 0 \text{ else }
 \end{cases}
 */
struct ZCutter
{
    ZCutter(double ZX): Z_X(ZX){}
    double operator()(double R, double Z) const {
        if( Z> Z_X)
            return 1;
        return 0;
    }
    double operator()(double R, double Z, double phi) const{
        return this->operator()(R,Z);
    }
    private:
    double Z_X;
};


/**
 * @brief Damps the outer boundary in a zone
 * from psipmaxcut to psipmaxcut+ 4*alpha with a normal distribution
 * Returns 1 inside, zero outside and a gaussian within
     \f[ \begin{cases}
 1 \text{ if } \psi_p(R,Z) < \psi_{p,max,cut}\\
 0 \text{ if } \psi_p(R,Z) > (\psi_{p,max,cut} + 4\alpha) \\
 \exp\left( - \frac{(\psi_p - \psi_{p,max,cut})^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f]
 *
 */
struct GaussianDamping : public aCloneableCylindricalFunctor<GaussianDamping>
{
    GaussianDamping( CylindricalFunctor psi, double psipmaxcut, double alpha):
        m_psip(psi), m_psipmaxcut(psipmaxcut), m_alpha(alpha) { }
    private:
    double do_compute(double R, double Z)const
    {
        double psip = m_psip(R,Z);
        if( psip > m_psipmaxcut + 4.*m_alpha) return 0.;
        if( psip < m_psipmaxcut) return 1.;
        return exp( -( psip-m_psipmaxcut)*( psip-m_psipmaxcut)/2./m_alpha/m_alpha);
    }
    CylindricalFunctor m_psip;
    double m_psipmaxcut, m_alpha;
};

/**
 * @brief Damps the inner boundary in a zone
 * from psipmax to psipmax+ 4*alpha with a normal distribution
 * Returns 1 inside, zero outside and a gaussian within
     \f[ \begin{cases}
 1 \text{ if } \psi_p(R,Z) < (\psi_{p,max} - 4\alpha)\\
 0 \text{ if } \psi_p(R,Z) > \psi_{p,max} \\
 \exp\left( - \frac{(\psi_p - \psi_{p,max} + 4\alpha)^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f]
 *
 */
struct GaussianProfDamping : public aCloneableCylindricalFunctor<GaussianProfDamping>
{
    GaussianProfDamping( CylindricalFunctor psi, double psipmax, double alpha):
        m_damp( psi, psipmax-4*alpha, alpha) { }
    private:
    double do_compute(double R, double Z)const
    {
        return m_damp(R,Z);
    }
    GaussianDamping m_damp;
};

/**
 * @brief Damps the inner boundary in a zone
 * from psipmax to psipmax- 4*alpha with a normal distribution
 * Returns 1 inside, zero outside and a gaussian within
 * Additionally cuts if Z < Z_xpoint
     \f[ \begin{cases}
 1 \text{ if } \psi_p(R,Z) < (\psi_{p,max} - 4\alpha) \\
 0 \text{ if } \psi_p(R,Z) > \psi_{p,max} || Z < -1.1\varepsilon a  \\
 \exp\left( - \frac{(\psi_p - \psi_{p,max} + 4\alpha)^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f]
 *
 */
struct GaussianProfXDamping : public aCloneableCylindricalFunctor<GaussianProfXDamping>
{
    GaussianProfXDamping( CylindricalFunctor psi, dg::geo::solovev::Parameters gp):
        m_cut ( gp.elongation*gp.a), m_damp( psi, gp.psipmax-4*gp.alpha, gp.alpha) { }
    private:
    double do_compute(double R, double Z)const
    {
        return m_damp(R,Z)*m_cut(R,Z);
    }
    ZCutter m_cut;
    GaussianDamping m_damp;
};

/**
 * @brief
 * A tanh profile shifted to \f$ \psi_{p,max}-3\alpha\f$
 \f[ 0.5\left( 1 + \tanh\left( -\frac{\psi_p(R,Z) - \psi_{p,max} + 3\alpha}{\alpha}\right)\right) \f]

 Similar to GaussianProfDamping is zero outside given \c psipmax and
 one inside of it.
 */
struct TanhDamping : public aCloneableCylindricalFunctor<TanhDamping>
{
    TanhDamping(CylindricalFunctor psi, double psipmax, double alpha):
            m_psipmin(psipmax), m_alpha(alpha), m_psip(psi) { }
    private:
    double do_compute(double R, double Z)const
    {
        return 0.5*(1.+tanh(-(m_psip(R,Z)-m_psipmin + 3.*m_alpha)/m_alpha) );
    }
    double m_psipmin, m_alpha;
    CylindricalFunctor m_psip;
};
/**
 * @brief
 * A tanh profile shifted to \f$ \psi_{p,max}-3\alpha\f$, 0 if below X-point
 \f[ 0.5\left( 1 + \tanh\left( -\frac{\psi_p(R,Z) - \psi_{p,max} + 3\alpha}{\alpha}\right)\right)\\
 0 \text{if} Z < -1.1\varepsilon a \f]

 Similar to GaussianProfDamping is zero outside given \c psipmax and
 one inside of it.
 */
struct TanhXDamping : public aCloneableCylindricalFunctor<TanhXDamping>
{
    TanhXDamping(CylindricalFunctor psi, dg::geo::solovev::Parameters gp):
            m_source(psi, gp.psipmax, gp.alpha), m_cut( gp.elongation*gp.a) { }
    private:
    double do_compute(double R, double Z)const
    {
        return m_source(R,Z)*m_cut(R,Z);
    }
    TanhDamping m_source;
    ZCutter m_cut;
};

/**
 * @brief Density profile with variable peak amplitude and background amplitude
 *\f[ N(R,Z)=\begin{cases}
 A_{bg} + A_{peak}\frac{\psi_p(R,Z)} {\psi_p(R_0, 0)} \text{ if }\psi_p < \psi_{p,max} \\
 A_{bg} \text{ else }
 \end{cases}
\f]
 */
struct Nprofile : public aCloneableCylindricalFunctor<Nprofile>
{
    Nprofile( CylindricalFunctor psi, double R_0, double psipmax, double bgprofamp, double peakamp):
         m_bgamp(bgprofamp), m_namp( peakamp), m_norm( psi(R_0, 0.)), m_psipmax(psipmax),
         m_psip(psi) {
         }
    private:
    double do_compute(double R, double Z)const
    {
        double psip = m_psip(R,Z);
        if (psip<m_psipmax)
            return m_bgamp +(psip/m_norm*m_namp);
        return m_bgamp;
    }
    double m_bgamp, m_namp, m_norm, m_psipmax;
    CylindricalFunctor m_psip;
};

/**
 * @brief Zonal flow field
     \f[ N(R,Z)=\begin{cases}
    A |\sin(2\pi k_\psi \psi_p(R,Z) )| \text{ if }\psi_p(R,Z) < \psi_{p,max} \\
    0 \text{ else }
 \end{cases}
   \f]
 */
struct ZonalFlow : public aCloneableCylindricalFunctor<ZonalFlow>
{
    ZonalFlow( CylindricalFunctor psi,  double psipmax, double amplitude, double k_psi):
        m_amp(amplitude), m_k(k_psi), m_psipmax(psipmax),
        m_psip(psi) { }
    private:
    double do_compute(double R, double Z)const
    {
        double psip = m_psip(R,Z);
        if (psip<m_psipmax)
            return (m_amp*fabs(sin(2.*M_PI*psip*m_k)));
        return 0.;
    }
    double m_amp, m_k, m_psipmax;
    CylindricalFunctor m_psip;
};


///@}
}//namespace functors
}//namespace dg

