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
        1  \text{ if } \psi_{p,min} < \psi_p < \psi_{p,max}\\
        0  \text{ else}
     \end{cases}\f]
 */
struct Iris
{
    Iris( double psi_min, double psi_max ):
        m_psipmin(psi_min), m_psipmax(psi_max) { }
    DG_DEVICE
    double operator()(double psip ) const
    {
        if( psip > m_psipmax) return 0.;
        if( psip < m_psipmin) return 0.;
        return 1.;
    }
    private:
    double m_psipmin, m_psipmax;
};
/**
 * @brief Returns zero outside psipmaxcut otherwise 1
     \f[ \begin{cases}
        0  \text{ if } \psi_p > \psi_{p,maxcut} \\
        1  \text{ else}
     \end{cases}\f]
 */
struct Pupil
{
    Pupil( double psipmaxcut): m_psipmaxcut(psipmaxcut) { }
DG_DEVICE
    double operator()(double psip)const
    {
        if( psip > m_psipmaxcut) return 0.;
        return 1.;
    }
    private:
    double m_psipmaxcut;
};
/**
 * @brief Returns psi inside psipmax and psipmax outside psipmax
     \f[ \begin{cases}
        \psi_{p,max}  \text{ if } \psi_p > \psi_{p,max} \\
        \psi_p \text{ else}
     \end{cases}\f]
 */
struct PsiPupil
{
    PsiPupil( double psipmax): m_psipmax(psipmax) { }
DG_DEVICE
    double operator()(double psip)const
    {
        if( psip_(R,Z) > psipmax_) return psipmax_;
        return  psip_(R,Z);
    }
    private:
    double psipmax_;
};
/**
 * @brief Sets values to one outside psipmaxcut, zero else
     \f[ \begin{cases}
        1  \text{ if } \psi_p(R,Z) > \psi_{p,maxlim} \\
        0  \text{ else}
     \end{cases}\f]
 *
 */
struct PsiLimiter
{
    PsiLimiter(double psipmaxlim):
        psipmaxlim_(psipmaxlim), psip_(psi) { }

    double operator()(double psip)const
    {
        if( psip_(R,Z) > psipmaxlim_) return 1.;
        return 0.;
    }
    private:
    double psipmaxlim_;
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
struct GaussianDamping
{
    GaussianDamping( double psipmaxcut, double alpha):
        psip_(psi), psipmaxcut_(psipmaxcut), alpha_(alpha) { }
    double operator()(double psip)const
    {
        if( psip_(R,Z) > psipmaxcut_ + 4.*alpha_) return 0.;
        if( psip_(R,Z) < psipmaxcut_) return 1.;
        return exp( -( psip_(R,Z)-psipmaxcut_)*( psip_(R,Z)-psipmaxcut_)/2./alpha_/alpha_);
    }
    private:
    double psipmaxcut_, alpha_;
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
struct GaussianProfDamping
{
    GaussianProfDamping( double psipmax, double alpha):
        psip_(psi), psipmax_(psipmax), alpha_(alpha) { }
    double operator()(double psip)const
    {
        if( psip_(R,Z) > psipmax_ ) return 0.;
        if( psip_(R,Z) < (psipmax_-4.*alpha_)) return 1.;
        return exp( -( psip_(R,Z)-(psipmax_-4.*alpha_))*( psip_(R,Z)-(psipmax_-4.*alpha_))/2./alpha_/alpha_);
    }
    private:
    double psipmax_, alpha_;
};
/**
 * @brief Damps the inner boundary in a zone
 * from psipmax to psipmax+ 4*alpha with a normal distribution
 * Returns 1 inside, zero outside and a gaussian within
 * Additionally cuts if Z < Z_xpoint
     \f[ \begin{cases}
 1 \text{ if } \psi_p(R,Z) < (\psi_{p,max} - 4\alpha) \\
 0 \text{ if } \psi_p(R,Z) > \psi_{p,max} || Z < -1.1\varepsilon a  \\
 \exp\left( - \frac{(\psi_p - \psi_{p,max})^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f]
 *
 */
struct GaussianProfXDamping
{
    GaussianProfXDamping( dg::geo::solovev::Parameters gp):
        gp_(gp),
        psip_(psi) { }
    double operator()(double psip)const
    {
        if( psip_(R,Z) > gp_.psipmax || Z<-1.1*gp_.elongation*gp_.a)
            return 0.;
        if( psip_(R,Z) < (gp_.psipmax-4.*gp_.alpha))
            return 1.;
        return exp( -( psip_(R,Z)-(gp_.psipmax-4.*gp_.alpha))*(
            psip_(R,Z) -(gp_.psipmax-4.*gp_.alpha) )/2./gp_.alpha/gp_.alpha);
    }
    private:
    dg::geo::solovev::Parameters gp_;
};

/**
 * @brief A tanh profile shifted to \f$ \psi_{p,min}-3\alpha\f$
 \f[ 0.5\left( 1 + \tanh\left( -\frac{\psi_p - \psi_{p,min} + 3\alpha}{\alpha}\right)\right) \f]
 */
struct TanhSource
{
    TanhSource( double psipmin, double alpha):
            psipmin_(psipmin), alpha_(alpha), psip_(psi) { }
DG_DEVICE
    double operator()(double psip)const
    {
        return 0.5*(1.+tanh(-(psip -psipmin_ + 3.*alpha_)/alpha_) );
    }
    private:
    double psipmin_, alpha_;
};

/**
 * @brief Returns density profile with variable peak amplitude and background amplitude
     \f[ N(R,Z)=\begin{cases}
    A \frac{\psi_p(R,Z)} {\psi_p(R_0, 0)} \text{ if }\psi_p < 0 \\
    0 \text{ else }
 \end{cases}
   \f]
 */
struct Nprofile
{
    Nprofile( double amp, double psip0):
        m_amp( amp), m_normalize( psip0) { }
    DG_DEVICE
    double operator()(double psip)const
    {
        if ( psip < 0)
            return m_amp*m_psip/m_normalize;
        else
            return 0;
    }
    private:
    double m_amp, m_nomalize;
};

/**
 * @brief returns zonal flow field
     \f[ N(R,Z)=\begin{cases}
    A |\sin( 2\pi k_\psi\psi_p )| \text{ if }\psi_p < 0 \\
    0 \text{ else }
 \end{cases}
   \f]
 */
struct ZonalFlow
{
    ZonalFlow( double amp, double k_psi):
        m_amp(amp), m_k(k_psi)
        { }
    DG_DEVICE
    double operator( )( double psip) const
    {
        if ( psip < 0)
            return m_amp*fabs( sin(2.*M_PI*psip*m_k) );
        return 0.;
    }
    private:
    double m_amp, m_k;
};


///@}
}//namespace functors
}//namespace dg

