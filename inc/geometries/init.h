#pragma once
//#include "solovev.h"
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
/**
 * @brief Returns zero outside psipmax and inside psipmin, otherwise 1
     \f[ \begin{cases}
        1  \text{ if } \psi_{p,min} < \psi_p(R,Z) < \psi_{p,max}\\
        0  \text{ else}
     \end{cases}\f]
     @tparam Psi models aBinaryOperator
 */ 
template<class Psi>
struct Iris
{
    Iris( Psi psi, double psi_min, double psi_max ): 
        psip_(psi), psipmin_(psi_min), psipmax_(psi_max) { }

    /**
     * @brief 
     \f[ \begin{cases}
        1  \text{ if } \psi_{p,min} < \psi_p(R,Z) < \psi_{p,max}\\
        0  \text{ else}
     \end{cases}\f]
     */
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > psipmax_) return 0.;
        if( psip_(R,Z) < psipmin_) return 0.;
        return 1.;
    }
    /**
    * @brief == operator()(R,Z)
    */
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    Psi psip_;
    double psipmin_, psipmax_;
};
/**
 * @brief Returns zero outside psipmaxcut otherwise 1
     \f[ \begin{cases}
        0  \text{ if } \psi_p(R,Z) > \psi_{p,maxcut} \\
        1  \text{ else}
     \end{cases}\f]
 */ 
template<class Psi>
struct Pupil
{
    Pupil( Psi psi, double psipmaxcut): 
        psip_(psi), psipmaxcut_(psipmaxcut) { }
    /**
     * @brief 
     \f[ \begin{cases}
        0  \text{ if } \psi_p(R,Z) > \psi_{p,maxcut} \\
        1  \text{ else}
     \end{cases}\f]
     */
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > psipmaxcut_) return 0.;
        return 1.;
    }
    /**
    * @brief == operator()(R,Z)
    */
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    Psi psip_;
    double psipmaxcut_;
};
/**
 * @brief Returns psi inside psipmax and psipmax outside psipmax
     \f[ \begin{cases}
        \psi_{p,max}  \text{ if } \psi_p(R,Z) > \psi_{p,max} \\
        \psi_p(R,Z) \text{ else}
     \end{cases}\f]
 */ 
template<class Psi>
struct PsiPupil
{
    PsiPupil( Psi psi, double psipmax): 
        psipmax_(psipmax), psip_(psi) { } 
    /**
     * @brief 
     \f[ \begin{cases}
        \psi_{p,max}  \text{ if } \psi_p(R,Z) > \psi_{p,max} \\
        \psi_p(R,Z) \text{ else}
     \end{cases}\f]
     */
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > psipmax_) return psipmax_;
        return  psip_(R,Z);
    }
    /**
    * @brief == operator()(R,Z)
    */
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    double psipmax_;
    Psi psip_;
};
/**
 * @brief Sets values to one outside psipmaxcut, zero else
     \f[ \begin{cases}
        1  \text{ if } \psi_p(R,Z) > \psi_{p,maxlim} \\
        0  \text{ else}
     \end{cases}\f]
 *
 */ 
template<class Psi>
struct PsiLimiter
{
    PsiLimiter( Psi psi, double psipmaxlim): 
        psipmaxlim_(psipmaxlim), psip_(psi) { }

    /**
     * @brief 
     \f[ \begin{cases}
        1  \text{ if } \psi_p(R,Z) > \psi_{p,maxlim} \\
        0  \text{ else}
     \end{cases}\f]
     */
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > psipmaxlim_) return 1.;
        return 0.;
    }
    /**
    * @brief == operator()(R,Z)
    */
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    double psipmaxlim_;
    Psi psip_;
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
template< class Psi>
struct GaussianDamping
{
    GaussianDamping( Psi psi, double psipmaxcut, double alpha):
        psip_(psi), psipmaxcut_(psipmaxcut), alpha_(alpha) { }
    /**
     * @brief 
     \f[ \begin{cases}
 1 \text{ if } \psi_p(R,Z) < \psi_{p,max,cut}\\
 0 \text{ if } \psi_p(R,Z) > (\psi_{p,max,cut} + 4\alpha) \\
 \exp\left( - \frac{(\psi_p - \psi_{p,max,cut})^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f]
     */
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > psipmaxcut_ + 4.*alpha_) return 0.;
        if( psip_(R,Z) < psipmaxcut_) return 1.;
        return exp( -( psip_(R,Z)-psipmaxcut_)*( psip_(R,Z)-psipmaxcut_)/2./alpha_/alpha_);
    }
    /**
    * @brief == operator()(R,Z)
    */
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    Psi psip_;
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
template< class Psi>
struct GaussianProfDamping
{
    GaussianProfDamping( Psi psi, double psipmax, double alpha):
        psip_(psi), psipmax_(psipmax), alpha_(alpha) { }
    /**
     * @brief 
     \f[ \begin{cases}
 1 \text{ if } \psi_p(R,Z) < (\psi_{p,max} - 4\alpha)\\
 0 \text{ if } \psi_p(R,Z) > \psi_{p,max} \\
 \exp\left( - \frac{(\psi_p - \psi_{p,max} + 4\alpha)^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f]
     */
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > psipmax_ ) return 0.;
        if( psip_(R,Z) < (psipmax_-4.*alpha_)) return 1.;
        return exp( -( psip_(R,Z)-(psipmax_-4.*alpha_))*( psip_(R,Z)-(psipmax_-4.*alpha_))/2./alpha_/alpha_);
    }
    /**
    * @brief == operator()(R,Z)
    */
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    Psi psip_;
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
template <class Psi>
struct GaussianProfXDamping
{
    GaussianProfXDamping( Psi psi, dg::geo::solovev::GeomParameters gp):
        gp_(gp),
        psip_(psi) {
        }
    /**
     * @brief 
     \f[ \begin{cases}
 1 \text{ if } \psi_p(R,Z) < (\psi_{p,max} - 4\alpha) \\
 0 \text{ if } \psi_p(R,Z) > \psi_{p,max} \text{ or } Z < -1.1\varepsilon a  \\
 \exp\left( - \frac{(\psi_p - \psi_{p,max})^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f]
     */
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmax || Z<-1.1*gp_.elongation*gp_.a) return 0.;
        if( psip_(R,Z) < (gp_.psipmax-4.*gp_.alpha)) return 1.;
        return exp( -( psip_(R,Z)-(gp_.psipmax-4.*gp_.alpha))*( psip_(R,Z)-(gp_.psipmax-4.*gp_.alpha))/2./gp_.alpha/gp_.alpha);
    }
    /**
    * @brief == operator()(R,Z)
    */
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    dg::geo::solovev::GeomParameters gp_;
    Psi psip_;
};

/**
 * @brief source for quantities N ... dtlnN = ...+ source/N
 * Returns a tanh profile shifted to \f$ \psi_{p,min}-3\alpha\f$
 \f[ 0.5\left( 1 + \tanh\left( -\frac{\psi_p(R,Z) - \psi_{p,min} + 3\alpha}{\alpha}\right)\right) \f]
 */
template<class Psi>
struct TanhSource
{
        TanhSource(Psi psi, double psipmin, double alpha):
            psipmin_(psipmin), alpha_(alpha), psip_(psi) { }
    /**
     * @brief \f[ 0.5\left( 1 + \tanh\left( -\frac{\psi_p(R,Z) - \psi_{p,min} + 3\alpha}{\alpha}\right)\right)
   \f]
     */
    double operator( )(double R, double Z)
    {
        return 0.5*(1.+tanh(-(psip_(R,Z)-psipmin_ + 3.*alpha_)/alpha_) );
    }
    /**
    * @brief == operator()(R,Z)
    */
    double operator( )(double R, double Z, double phi)
    {
        return 0.5*(1.+tanh(-(psip_(R,Z,phi)-psipmin_ + 3.*alpha_)/alpha_) );
    }
    private:
    double psipmin_, alpha_; 
    Psi psip_;
};

// struct Gradient
// {
//     Gradient(  eule::Parameters p, GeomParameters gp):
//         p_(p),
//         gp_(gp),
//         psip_(gp) {
//     }
//     double operator( )(double R, double Z)
//     {
//         if( psip_(R,Z) < (gp_.psipmin)) return p_.nprofileamp+p_.bgprofamp;
//         if( psip_(R,Z) < 0.) return p_.nprofileamp+p_.bgprofamp-(gp_.psipmin-psip_(R,Z))*(p_.nprofileamp/gp_.psipmin);
//         return p_.bgprofamp;
//     }
//     double operator( )(double R, double Z, double phi)
//     {
//         return (*this)(R,Z);
// 
//     }
//     private:
//     eule::Parameters p_;
//     GeomParameters gp_;
//     Psi psip_;
// };

/**
 * @brief Returns density profile with variable peak amplitude and background amplitude 
     *\f[ N(R,Z)=\begin{cases}
 A_{bg} + A_{peak}\frac{\psi_p(R,Z)} {\psi_p(R_0, 0)} \text{ if }\psi_p < \psi_{p,max} \\
 A_{bg} \text{ else } 
 \end{cases}
   \f]
   @tparam Psi models aBinaryOperator
 */ 
template<class Psi>
struct Nprofile
{
     Nprofile( double bgprofamp, double peakamp, dg::geo::solovev::GeomParameters gp, Psi psi):
         bgamp(bgprofamp), namp( peakamp),
         gp_(gp),
         psip_(psi) { }
    /**
     * @brief \f[ N(R,Z)=\begin{cases}
 A_{bg} + A_{peak}\frac{\psi_p(R,Z)} {\psi_p(R_0, 0)} \text{ if }\psi_p < \psi_{p,max} \\
 A_{bg} \text{ else } 
 \end{cases}
   \f]
     */
   double operator( )(double R, double Z)
    {
        if (psip_(R,Z)<gp_.psipmax) return bgamp +(psip_(R,Z)/psip_(gp_.R_0,0.0)*namp);
	if( psip_(R,Z) > gp_.psipmax || Z<-1.1*gp_.elongation*gp_.a) return bgamp;
        return bgamp;
    }
    /**
    * @brief == operator()(R,Z)
    */
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    double bgamp, namp;
    dg::geo::solovev::GeomParameters gp_;
    Psi psip_;
};

/**
 * @brief returns zonal flow field 
     \f[ N(R,Z)=\begin{cases}
    A_{bg} |\cos(2\pi\psi_p(R,Z) k_\psi)| \text{ if }\psi_p < \psi_{p,max} \\
    0 \text{ else } 
 \end{cases}
   \f]
 */ 
template<class Psi>
struct ZonalFlow
{
    ZonalFlow(  double amplitude, double k_psi, dg::geo::solovev::GeomParameters gp, Psi psi):
        amp_(amplitude), k_(k_psi),
        gp_(gp),
        psip_(psi) { }
    /**
     * @brief \f[ N(R,Z)=\begin{cases}
 A_{bg} |\cos(2\pi\psi_p(R,Z) k_\psi)| \text{ if }\psi_p < \psi_{p,max} \\
  0 \text{ else } 
 \end{cases}
   \f]
     */
    double operator() (double R, double Z)
    {
      if (psip_(R,Z)<gp_.psipmax) 
          return (amp_*fabs(cos(2.*M_PI*psip_(R,Z)*k_)));
      return 0.;

    }
    /**
    * @brief == operator()(R,Z)
    */
    double operator() (double R, double Z,double phi)
    {
        return (*this)(R,Z);
    }
    private:
    double amp_, k_;
    dg::geo::solovev::GeomParameters gp_;
    Psi psip_;
};


///@}
}//namespace functors
}//namespace dg

