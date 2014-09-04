#pragma once
#include "geometry.h"
/*!@file
 *
 * Initialize and Damping objects
 */
namespace solovev
{
///@addtogroup geom
///@{
/**
 * @brief Sets values to zero outside psipmax and inside psipmin
 */ 
struct Iris
{
    Iris( GeomParameters gp ): 
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmax) return 0.;
        if( psip_(R,Z) < gp_.psipmin) return 0.;
        return 1.;
    }
    double operator( )(double R, double Z, double phi)
    {
        if( psip_(R,Z,phi) > gp_.psipmax) return 0.;
        if( psip_(R,Z,phi) < gp_.psipmin) return 0.;
        return 1.;
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/**
 * @brief Sets values to zero outside psipmax 
 */ 
struct Pupil
{
    Pupil( GeomParameters gp): 
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmaxcut) return 0.;
        return 1.;
    }
    double operator( )(double R, double Z, double phi)
    {
        if( psip_(R,Z,phi) > gp_.psipmaxcut) return 0.;
        return 1.;
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/**
 * @brief Sets values to zero outside psipmax 
 */ 
struct PsiLimiter
{
    PsiLimiter( GeomParameters gp): 
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmaxcut) return 1.;
        return 0.;
    }
    double operator( )(double R, double Z, double phi)
    {
        if( psip_(R,Z,phi) > gp_.psipmaxcut) return 1.;
        return 0.;
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/**
 * @brief Damps the outer boundary in a zone from psipmax to psipmax+ 4*alpha with a normal distribution
 */ 
struct GaussianDamping
{
    GaussianDamping( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmax + 4.*gp_.alpha) return 0.;
        if( psip_(R,Z) < (gp_.psipmax)) return 1.;
        return exp( -( psip_(R,Z)-gp_.psipmax)*( psip_(R,Z)-gp_.psipmax)/2./gp_.alpha/gp_.alpha);
    }
    double operator( )(double R, double Z, double phi)
    {
        if( psip_(R,Z,phi) > gp_.psipmax + 4.*gp_.alpha) return 0.;
        if( psip_(R,Z,phi) < (gp_.psipmax)) return 1.;
        return exp( -( psip_(R,Z,phi)-gp_.psipmax)*( psip_(R,Z,phi)-gp_.psipmax)/2./gp_.alpha/gp_.alpha);

    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/**
 * @brief Damps lnN quantitie with tanh
 */ 
struct TanhDampingProf
{
        TanhDampingProf( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        return 0.5*(1.+tanh(-(psip_(R,Z)-gp_.psipmax + 3.*gp_.alpha)/gp_.alpha) );
    }
    double operator( )(double R, double Z, double phi)
    {
        return 0.5*(1.+tanh(-(psip_(R,Z,phi)-gp_.psipmax + 3.*gp_.alpha)/gp_.alpha) );
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/*damps from psi_max on outwards*/
struct TanhDampingOut
{
        TanhDampingOut( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        return 0.5*(1.+tanh(-(psip_(R,Z)-gp_.psipmaxcut - 3.*gp_.alpha)/gp_.alpha) );
    }
    double operator( )(double R, double Z, double phi)
    {
        return 0.5*(1.+tanh(-(psip_(R,Z,phi)-gp_.psipmaxcut - 3.*gp_.alpha)/gp_.alpha) );
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
struct TanhDampingIn
{
        TanhDampingIn( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        return 0.5*(1.+tanh(-(psip_(R,Z)-gp_.psipmaxcut + 3.*gp_.alpha)/gp_.alpha) );
    }
    double operator( )(double R, double Z, double phi)
    {
        return 0.5*(1.+tanh(-(psip_(R,Z,phi)-gp_.psipmaxcut + 3.*gp_.alpha)/gp_.alpha) );
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/*increases from psi_maxlap on*/
struct TanhDampingInv
{
        TanhDampingInv( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        return 1.-0.5*(1.+tanh(-(psip_(R,Z)-gp_.psipmaxlap - 3.*gp_.alpha)/gp_.alpha) );
    }
    double operator( )(double R, double Z, double phi)
    {
        return 1.-0.5*(1.+tanh(-(psip_(R,Z,phi)-gp_.psipmaxlap - 3.*gp_.alpha)/gp_.alpha) );
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};

/**
 * @brief source for quantities N ... dtlnN = ...+ source/N
 */
struct TanhSource
{
        TanhSource( GeomParameters gp, double amp):
        gp_(gp),
        amp_(amp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        return amp_*0.5*(1.+tanh(-(psip_(R,Z)-gp_.psipmin + 3.*gp_.alpha)/gp_.alpha) );
    }
    double operator( )(double R, double Z, double phi)
    {
        return amp_*0.5*(1.+tanh(-(psip_(R,Z,phi)-gp_.psipmin + 3.*gp_.alpha)/gp_.alpha) );
    }
    private:
    GeomParameters gp_;
    double amp_;
    Psip psip_;
};
/**
 * @brief Computes the background gradient for the logarithmic densities on n>=1
 */ 
struct Gradient
{
    Gradient( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
   double operator( )(double R, double Z)
    {
        if( psip_(R,Z) < (gp_.psipmin)) return exp(gp_.lnN_inner*log(10)); 
        if( psip_(R,Z) < 0.) return -1./gp_.psipmin*(psip_(R,Z) -gp_.psipmin +exp(gp_.lnN_inner*log(10))*(- psip_(R,Z)));
        return 1.;
    }
    double operator( )(double R, double Z, double phi)
    {
        if( psip_(R,Z,phi) < (gp_.psipmin)) return exp(gp_.lnN_inner*log(10)); 
        if( psip_(R,Z,phi) < 0.) return -1./gp_.psipmin*(psip_(R,Z,phi) -gp_.psipmin +exp(gp_.lnN_inner*log(10))*(- psip_(R,Z,phi)));
        return 1.;
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/**
 * @brief Returns density profile with variable peak amplitude and background amplitude 
 */ 
struct Nprofile
{
     Nprofile( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
   double operator( )(double R, double Z)
    {
        if (psip_(R,Z)<0.) return gp_.bgprofamp +(psip_(R,Z)/psip_(gp_.R_0,0.0)*gp_.nprofileamp);
        return gp_.bgprofamp;
    }
    double operator( )(double R, double Z, double phi)
    {
        if (psip_(R,Z,phi)<0.) return gp_.bgprofamp+(psip_(R,Z,phi)/psip_(gp_.R_0,0.0,0.0)*gp_.nprofileamp);
        return gp_.bgprofamp;
    }
    private:
    GeomParameters gp_;    
    Psip psip_;
};   

/**
 * @brief returns zonal flow field 
 */ 
struct ZonalFlow
{
    ZonalFlow(GeomParameters gp,  double amp):
        gp_(gp),
        amp_(amp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
    }
    double operator() (double R, double Z) 
    {
      if (psip_(R,Z)<0.) return (amp_*abs(cos(2.*M_PI*psip_(R,Z)*gp_.k_psi)));
      return 0.;
      
    }
    double operator() (double R, double Z,double phi) 
    {
        if (psip_(R,Z,phi)<0.) return ( amp_*abs(cos(2.*M_PI*psip_(R,Z,phi)*gp_.k_psi)));
        return 0.;
    }
    private:
    GeomParameters gp_;
    double amp_;
    Psip psip_;
};

/**
 * @brief testfunction to test the parallel derivative \f[ f = \psi_p(R,Z) \sin{(\varphi)}\f]
 */ 
struct TestFunction
{
    TestFunction(Psip psip) : psip_(psip){}
    double operator()( double R, double Z, double phi)
    {
        return psip_(R,Z,phi)*sin(phi);
    }
    private:
    Psip psip_;
};
/**
 * @brief analyitcal solution of the parallel derivative of the testfunction
 *  \f[ \nabla_\parallel f = \psi_p(R,Z) b^\varphi \cos{(\varphi)}\f]
 */ 
struct DeriTestFunction
{
    DeriTestFunction(GeomParameters gp, Psip psip,PsipR psipR, PsipZ psipZ, Ipol ipol, InvB invB) :gp_(gp), psip_(psip), psipR_(psipR), psipZ_(psipZ),ipol_(ipol), invB_(invB) {}
    double operator()( double R, double Z, double phi)
    {
        return  gp_.R_0*psip_(R,Z,phi)*ipol_(R,Z,phi)*cos(phi)*invB_(R,Z,phi)/R/R;
    }
    private:
    GeomParameters gp_;
    Psip psip_;
    PsipR psipR_;
    PsipZ psipZ_;
    Ipol ipol_;
    InvB invB_;
};
///@} 
}//namespace solovev
