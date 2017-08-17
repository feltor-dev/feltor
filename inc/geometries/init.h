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
struct Iris : public aCloneableBinaryFunctor<Iris>
{
    Iris( const aBinaryFunctor& psi, double psi_min, double psi_max ): 
        psip_(psi), psipmin_(psi_min), psipmax_(psi_max) { }
    private:
    double do_compute(double R, double Z)const
    {
        if( psip_.get()(R,Z) > psipmax_) return 0.;
        if( psip_.get()(R,Z) < psipmin_) return 0.;
        return 1.;
    }
    Handle<aBinaryFunctor> psip_;
    double psipmin_, psipmax_;
};
/**
 * @brief Returns zero outside psipmaxcut otherwise 1
     \f[ \begin{cases}
        0  \text{ if } \psi_p(R,Z) > \psi_{p,maxcut} \\
        1  \text{ else}
     \end{cases}\f]
 */ 
struct Pupil : public aCloneableBinaryFunctor<Pupil>
{
    Pupil( const aBinaryFunctor& psi, double psipmaxcut): 
        psip_(psi), psipmaxcut_(psipmaxcut) { }
    private:
    double do_compute(double R, double Z)const
    {
        if( psip_.get()(R,Z) > psipmaxcut_) return 0.;
        return 1.;
    }
    Handle<aBinaryFunctor> psip_;
    double psipmaxcut_;
};
/**
 * @brief Returns psi inside psipmax and psipmax outside psipmax
     \f[ \begin{cases}
        \psi_{p,max}  \text{ if } \psi_p(R,Z) > \psi_{p,max} \\
        \psi_p(R,Z) \text{ else}
     \end{cases}\f]
 */ 
struct PsiPupil : public aCloneableBinaryFunctor<PsiPupil>
{
    PsiPupil(const aBinaryFunctor& psi, double psipmax): 
        psipmax_(psipmax), psip_(psi) { } 
    private:
    double do_compute(double R, double Z)const
    {
        if( psip_.get()(R,Z) > psipmax_) return psipmax_;
        return  psip_.get()(R,Z);
    }
    double psipmax_;
    Handle<aBinaryFunctor> psip_;
};
/**
 * @brief Sets values to one outside psipmaxcut, zero else
     \f[ \begin{cases}
        1  \text{ if } \psi_p(R,Z) > \psi_{p,maxlim} \\
        0  \text{ else}
     \end{cases}\f]
 *
 */ 
struct PsiLimiter : public aCloneableBinaryFunctor<PsiLimiter>
{
    PsiLimiter( const aBinaryFunctor& psi, double psipmaxlim): 
        psipmaxlim_(psipmaxlim), psip_(psi) { }

    private:
    double do_compute(double R, double Z)const
    {
        if( psip_.get()(R,Z) > psipmaxlim_) return 1.;
        return 0.;
    }
    double psipmaxlim_;
    Handle<aBinaryFunctor> psip_;
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
struct GaussianDamping : public aCloneableBinaryFunctor<GaussianDamping>
{
    GaussianDamping( const aBinaryFunctor& psi, double psipmaxcut, double alpha):
        psip_(psi), psipmaxcut_(psipmaxcut), alpha_(alpha) { }
    private:
    double do_compute(double R, double Z)const
    {
        if( psip_.get()(R,Z) > psipmaxcut_ + 4.*alpha_) return 0.;
        if( psip_.get()(R,Z) < psipmaxcut_) return 1.;
        return exp( -( psip_.get()(R,Z)-psipmaxcut_)*( psip_.get()(R,Z)-psipmaxcut_)/2./alpha_/alpha_);
    }
    Handle<aBinaryFunctor> psip_;
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
struct GaussianProfDamping : public aCloneableBinaryFunctor<GaussianProfDamping>
{
    GaussianProfDamping( const aBinaryFunctor& psi, double psipmax, double alpha):
        psip_(psi), psipmax_(psipmax), alpha_(alpha) { }
    private:
    double do_compute(double R, double Z)const
    {
        if( psip_.get()(R,Z) > psipmax_ ) return 0.;
        if( psip_.get()(R,Z) < (psipmax_-4.*alpha_)) return 1.;
        return exp( -( psip_.get()(R,Z)-(psipmax_-4.*alpha_))*( psip_.get()(R,Z)-(psipmax_-4.*alpha_))/2./alpha_/alpha_);
    }
    Handle<aBinaryFunctor> psip_;
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
struct GaussianProfXDamping : public aCloneableBinaryFunctor<GaussianProfXDamping>
{
    GaussianProfXDamping( const aBinaryFunctor& psi, dg::geo::solovev::GeomParameters gp):
        gp_(gp),
        psip_(psi) { }
    private:
    double do_compute(double R, double Z)const
    {
        if( psip_.get()(R,Z) > gp_.psipmax || Z<-1.1*gp_.elongation*gp_.a) return 0.;
        if( psip_.get()(R,Z) < (gp_.psipmax-4.*gp_.alpha)) return 1.;
        return exp( -( psip_.get()(R,Z)-(gp_.psipmax-4.*gp_.alpha))*( psip_.get()(R,Z)-(gp_.psipmax-4.*gp_.alpha))/2./gp_.alpha/gp_.alpha);
    }
    dg::geo::solovev::GeomParameters gp_;
    Handle<aBinaryFunctor> psip_;
};

/**
 * @brief source for quantities N ... dtlnN = ...+ source/N
 * Returns a tanh profile shifted to \f$ \psi_{p,min}-3\alpha\f$
 \f[ 0.5\left( 1 + \tanh\left( -\frac{\psi_p(R,Z) - \psi_{p,min} + 3\alpha}{\alpha}\right)\right) \f]
 */
struct TanhSource : public aCloneableBinaryFunctor<TanhSource>
{
    TanhSource(const aBinaryFunctor& psi, double psipmin, double alpha):
            psipmin_(psipmin), alpha_(alpha), psip_(psi) { }
    private:
    double do_compute(double R, double Z)const
    {
        return 0.5*(1.+tanh(-(psip_.get()(R,Z)-psipmin_ + 3.*alpha_)/alpha_) );
    }
    double psipmin_, alpha_; 
    Handle<aBinaryFunctor> psip_;
};

// struct Gradient : public aCloneableBinaryFunctor<Gradient>
// {
//     Gradient(  eule::Parameters p, GeomParameters gp):
//         p_(p),
//         gp_(gp),
//         psip_(gp) {
//     }
//     private:
//     double do_compute(double R, double Z)
//     {
//         if( psip_.get()(R,Z) < (gp_.psipmin)) return p_.nprofileamp+p_.bgprofamp;
//         if( psip_.get()(R,Z) < 0.) return p_.nprofileamp+p_.bgprofamp-(gp_.psipmin-psip_.get()(R,Z))*(p_.nprofileamp/gp_.psipmin);
//         return p_.bgprofamp;
//     }
//     eule::Parameters p_;
//     GeomParameters gp_;
//     Handle<aBinaryFunctor> psip_;
// };

/**
 * @brief Returns density profile with variable peak amplitude and background amplitude 
     *\f[ N(R,Z)=\begin{cases}
 A_{bg} + A_{peak}\frac{\psi_p(R,Z)} {\psi_p(R_0, 0)} \text{ if }\psi_p < \psi_{p,max} \\
 A_{bg} \text{ else } 
 \end{cases}
   \f]
 */ 
struct Nprofile : public aCloneableBinaryFunctor<Nprofile>
{
     Nprofile( double bgprofamp, double peakamp, dg::geo::solovev::GeomParameters gp, const aBinaryFunctor& psi):
         bgamp(bgprofamp), namp( peakamp),
         gp_(gp),
         psip_(psi) { }
    private:
    double do_compute(double R, double Z)const
    {
        if (psip_.get()(R,Z)<gp_.psipmax) return bgamp +(psip_.get()(R,Z)/psip_.get()(gp_.R_0,0.0)*namp);
	if( psip_.get()(R,Z) > gp_.psipmax || Z<-1.1*gp_.elongation*gp_.a) return bgamp;
        return bgamp;
    }
    double bgamp, namp;
    dg::geo::solovev::GeomParameters gp_;
    Handle<aBinaryFunctor> psip_;
};

/**
 * @brief returns zonal flow field 
     \f[ N(R,Z)=\begin{cases}
    A_{bg} |\cos(2\pi\psi_p(R,Z) k_\psi)| \text{ if }\psi_p < \psi_{p,max} \\
    0 \text{ else } 
 \end{cases}
   \f]
 */ 
struct ZonalFlow : public aCloneableBinaryFunctor<ZonalFlow>
{
    ZonalFlow(  double amplitude, double k_psi, dg::geo::solovev::GeomParameters gp, const aBinaryFunctor& psi):
        amp_(amplitude), k_(k_psi),
        gp_(gp),
        psip_(psi) { }
    private:
    double do_compute(double R, double Z)const
    {
      if (psip_.get()(R,Z)<gp_.psipmax) 
          return (amp_*fabs(cos(2.*M_PI*psip_.get()(R,Z)*k_)));
      return 0.;

    }
    double amp_, k_;
    dg::geo::solovev::GeomParameters gp_;
    Handle<aBinaryFunctor> psip_;
};


///@}
}//namespace functors
}//namespace dg

