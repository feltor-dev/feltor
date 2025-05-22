#pragma once
#include "fluxfunctions.h"

///@cond
namespace dg
{
namespace geo
{

////////////////////////////////////////////for grid generation/////////////////
namespace flux{

/**
 * @brief
 * \f[  d R/d \theta =   B^R/B^\theta \f],
 * \f[  d Z/d \theta =   B^Z/B^\theta \f],
 * \f[  d y/d \theta =   B^y/B^\theta\f]
 */
struct FieldRZYT
{
    //x0 and y0 sould be O-point to define angle with respect to O-point
    FieldRZYT( const CylindricalFunctorsLvl1& psip, const CylindricalFunctorsLvl1& ipol, double R0, double Z0): R_0_(R0), Z_0_(Z0), psip_(psip), ipol_(ipol){}
    void operator()(double, const std::array<double,3>& y, std::array<double,3>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double ipol=ipol_.f()(y[0], y[1]);
        yp[0] =  psipZ;//fieldR
        yp[1] = -psipR;//fieldZ
        yp[2] =ipol/y[0];
        double r2 = (y[0]-R_0_)*(y[0]-R_0_) + (y[1]-Z_0_)*(y[1]-Z_0_);
        double fieldT = psipZ*(y[1]-Z_0_)/r2 + psipR*(y[0]-R_0_)/r2;
        yp[0] /=  fieldT;
        yp[1] /=  fieldT;
        yp[2] /=  fieldT;
    }
  private:
    double R_0_, Z_0_;
    CylindricalFunctorsLvl1 psip_, ipol_;
};

struct FieldRZYZ
{
    FieldRZYZ( const CylindricalFunctorsLvl1& psip, const CylindricalFunctorsLvl1& ipol):psip_(psip), ipol_(ipol) {}
    void operator()(double, const std::array<double,3>& y, std::array<double,3>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double ipol=ipol_.f()(y[0], y[1]);
        yp[0] =  psipZ;//fieldR
        yp[1] = -psipR;//fieldZ
        yp[2] =   ipol/y[0]; //fieldYbar
        yp[0] /=  yp[1];
        yp[2] /=  yp[1];
        yp[1] =  1.;
    }
  private:
    CylindricalFunctorsLvl1 psip_, ipol_;
};

/**
 * @brief
 * \f[  d R/d y =   B^R/B^y \f],
 * \f[  d Z/d y =   B^Z/B^y \f],
 */
struct FieldRZY
{
    FieldRZY( const CylindricalFunctorsLvl2& psip, const CylindricalFunctorsLvl1& ipol): f_(1.), psip_(psip), ipol_(ipol){}
    void set_f(double f){ f_ = f;}
    void operator()(double, const std::array<double,2>& y, std::array<double,2>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double ipol=ipol_.f()(y[0], y[1]);
        double fnorm = y[0]/ipol/f_;
        yp[0] =  (psipZ)*fnorm;
        yp[1] = -(psipR)*fnorm;
    }
  private:
    double f_;
    CylindricalFunctorsLvl1 psip_,ipol_;
};

/**
 * @brief
 * \f[  d R/d y   =   B^R/B^y = \frac{q R}{I} \frac{\partial \psi_p}{\partial Z} \f],
 * \f[  d Z/d y   =   B^Z/B^y =    -\frac{q R}{I} \frac{\partial \psi_p}{\partial R} \f],
 * \f[  d y_R/d y =  \frac{q( \psi_p) R}{I( \psi_p)}\left[\frac{\partial^2 \psi_p}{\partial R \partial Z} y_R
    -\frac{\partial^2 \psi_p}{\partial^2 R} y_Z)\right] +
    \frac{\partial \psi_p}{\partial R} \left(\frac{1}{I(\psi_p)} \frac{\partial I(\psi_p)}{\partial \psi_p} -\frac{1}{q(\psi_p)} \frac{\partial q(\psi_p)}{\partial \psi_p}\right)-\frac{1}{R} \f],
 * \f[  d y_Z/d y =   - \frac{q( \psi_p) R}{I( \psi_p)}\left[\frac{\partial^2 \psi_p}{\partial Z^2} y_R\right)
    -\frac{\partial^2 \psi_p}{\partial R \partial Z} y_Z\right]+
    \frac{\partial \psi_p}{\partial Z} \left(\frac{1}{I(\psi_p)} \frac{\partial I(\psi_p)}{\partial \psi_p} -\frac{1}{q(\psi_p)} \frac{\partial q(\psi_p)}{\partial \psi_p}\right)\f],
 */
struct FieldRZYRYZY
{
    FieldRZYRYZY(const CylindricalFunctorsLvl2& psip, const CylindricalFunctorsLvl1& ipol): f_(1.), f_prime_(1), psip_(psip), ipol_(ipol){}
    void set_f( double new_f){ f_ = new_f;}
    void set_fp( double new_fp){ f_prime_ = new_fp;}
    void initialize( double R0, double Z0, double& yR, double& yZ)
    {
        double psipR = psip_.dfx()(R0, Z0), psipZ = psip_.dfy()(R0,Z0);
        double psip2 = (psipR*psipR+ psipZ*psipZ);
        double fnorm =R0/ipol_.f()(R0,Z0)/f_; //=Rq/I
        yR = -psipZ/psip2/fnorm;
        yZ = +psipR/psip2/fnorm;
    }
    void derive( double R0, double Z0, double& xR, double& xZ)
    {
        xR = +f_*psip_.dfx()(R0, Z0);
        xZ = +f_*psip_.dfy()(R0, Z0);
    }

    void operator()(double, const std::array<double,4>& y, std::array<double,4>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double psipRR = psip_.dfxx()(y[0], y[1]), psipRZ = psip_.dfxy()(y[0],y[1]), psipZZ = psip_.dfyy()(y[0],y[1]);
        double ipol=ipol_.f()(y[0], y[1]);
        double ipolR=ipol_.dfx()(y[0], y[1]);
        double ipolZ=ipol_.dfy()(y[0], y[1]);
        double fnorm =y[0]/ipol/f_; //=R/(I/q)

        yp[0] = -(psipZ)*fnorm;
        yp[1] = +(psipR)*fnorm;
        yp[2] = (+psipRZ*y[2]- psipRR*y[3])*fnorm + f_prime_/f_*psipR + ipolR/ipol - 1./y[0];
        yp[3] = (-psipRZ*y[3]+ psipZZ*y[2])*fnorm + f_prime_/f_*psipZ + ipolZ/ipol;

    }
  private:
    double f_, f_prime_;
    CylindricalFunctorsLvl2 psip_;
    CylindricalFunctorsLvl1 ipol_;
};

}//namespace flux
namespace ribeiro{

struct FieldRZYT
{
    //x0 and y0 sould be O-point to define angle with respect to O-point
    FieldRZYT( const CylindricalFunctorsLvl1& psip, double R0, double Z0, const CylindricalSymmTensorLvl1& chi = CylindricalSymmTensorLvl1() ): R_0_(R0), Z_0_(Z0), psip_(psip), chi_(chi){}
    void operator()(double, const std::array<double,3>& y, std::array<double,3>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double psip2 =   chi_.xx()(y[0], y[1])*psipR*psipR
                    + 2.*chi_.xy()(y[0], y[1])*psipR*psipZ
                       + chi_.yy()(y[0], y[1])*psipZ*psipZ;
        yp[0] = -psipZ;//fieldR
        yp[1] = +psipR;//fieldZ
        //yp[2] = 1; //volume
        //yp[2] = sqrt(psip2); //equalarc
        yp[2] = psip2; //ribeiro
        //yp[2] = psip2*sqrt(psip2); //separatrix
        double r2 = (y[0]-R_0_)*(y[0]-R_0_) + (y[1]-Z_0_)*(y[1]-Z_0_);
        double fieldT = psipZ*(y[1]-Z_0_)/r2 + psipR*(y[0]-R_0_)/r2;
        yp[0] /=  fieldT;
        yp[1] /=  fieldT;
        yp[2] /=  fieldT;
    }
  private:
    double R_0_, Z_0_;
    CylindricalFunctorsLvl1 psip_;
    CylindricalSymmTensorLvl1 chi_;
};

struct FieldRZYZ
{
    FieldRZYZ( const CylindricalFunctorsLvl1& psip, const CylindricalSymmTensorLvl1& chi = CylindricalSymmTensorLvl1() ): psip_(psip), chi_(chi){}
    void operator()(double, const std::array<double,3>& y, std::array<double,3>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double psip2 =   chi_.xx()(y[0], y[1])*psipR*psipR
                    + 2.*chi_.xy()(y[0], y[1])*psipR*psipZ
                       + chi_.yy()(y[0], y[1])*psipZ*psipZ;
        yp[0] = -psipZ;//fieldR
        yp[1] =  psipR;//fieldZ
        //yp[2] = 1.0; //volume
        //yp[2] = sqrt(psip2); //equalarc
        yp[2] = psip2; //ribeiro
        //yp[2] = psip2*sqrt(psip2); //separatrix
        yp[0] /=  yp[1];
        yp[2] /=  yp[1];
        yp[1] =  1.;
    }
  private:
    CylindricalFunctorsLvl1 psip_;
    CylindricalSymmTensorLvl1 chi_;
};

struct FieldRZY
{
    FieldRZY( const CylindricalFunctorsLvl1& psip, const CylindricalSymmTensorLvl1& chi = CylindricalSymmTensorLvl1() ): f_(1.), psip_(psip), chi_(chi){}
    void set_f(double f){ f_ = f;}
    void operator()(double, const std::array<double,2>& y, std::array<double,2>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double psip2 =   chi_.xx()(y[0], y[1])*psipR*psipR
                    + 2.*chi_.xy()(y[0], y[1])*psipR*psipZ
                       + chi_.yy()(y[0], y[1])*psipZ*psipZ;
        //yp[0] = +psipZ/f_;//volume
        //yp[1] = -psipR/f_;//volume
        //yp[0] = +psipZ/sqrt(psip2)/f_;//equalarc
        //yp[1] = -psipR/sqrt(psip2)/f_;//equalarc
        yp[0] = -psipZ/psip2/f_;//ribeiro
        yp[1] = +psipR/psip2/f_;//ribeiro
        //yp[0] = +psipZ/psip2/sqrt(psip2)/f_;//separatrix
        //yp[1] = -psipR/psip2/sqrt(psip2)/f_;//separatrix
    }
  private:
    double f_;
    CylindricalFunctorsLvl1 psip_;
    CylindricalSymmTensorLvl1 chi_;
};


struct FieldRZYRYZY
{
    FieldRZYRYZY( const CylindricalFunctorsLvl2& psip):psip_(psip){ f_ = f_prime_ = 1.;}
    void set_f( double new_f){ f_ = new_f;}
    void set_fp( double new_fp){ f_prime_ = new_fp;}
    void initialize( double R0, double Z0, double& yR, double& yZ)
    {
        yR = -f_*psip_.dfy()(R0, Z0);
        yZ = +f_*psip_.dfx()(R0, Z0);
    }
    void derive( double R0, double Z0, double& xR, double& xZ)
    {
        xR = +f_*psip_.dfx()(R0, Z0);
        xZ = +f_*psip_.dfy()(R0, Z0);
    }

    void operator()(double, const std::array<double,4>& y, std::array<double,4>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double psipRR = psip_.dfxx()(y[0], y[1]), psipRZ = psip_.dfxy()(y[0],y[1]), psipZZ = psip_.dfyy()(y[0],y[1]);
        double psip2 = (psipR*psipR+ psipZ*psipZ);

        yp[0] =  -psipZ/f_/psip2;
        yp[1] =  +psipR/f_/psip2;
        yp[2] =  ( + psipRZ*y[2] - psipRR*y[3] )/f_/psip2
            + f_prime_/f_* psipR + 2.*(psipR*psipRR + psipZ*psipRZ)/psip2 ;
        yp[3] =  (-psipRZ*y[3] + psipZZ*y[2])/f_/psip2
            + f_prime_/f_* psipZ + 2.*(psipR*psipRZ + psipZ*psipZZ)/psip2;
    }
  private:
    double f_, f_prime_;
    CylindricalFunctorsLvl2 psip_;
};
}//namespace ribeiro
namespace equalarc{


struct FieldRZYT
{
    //x0 and y0 sould be O-point to define angle with respect to O-point
    FieldRZYT( const CylindricalFunctorsLvl1& psip, double R0, double Z0, const CylindricalSymmTensorLvl1& chi = CylindricalSymmTensorLvl1() ): R_0_(R0), Z_0_(Z0), psip_(psip), chi_(chi){}
    void operator()(double, const std::array<double,3>& y, std::array<double,3>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double psip2 =   chi_.xx()(y[0], y[1])*psipR*psipR
                    + 2.*chi_.xy()(y[0], y[1])*psipR*psipZ
                       + chi_.yy()(y[0], y[1])*psipZ*psipZ;
        yp[0] = -psipZ;//fieldR
        yp[1] = +psipR;//fieldZ
        //yp[2] = 1; //volume
        yp[2] = sqrt(psip2); //equalarc
        //yp[2] = psip2; //ribeiro
        //yp[2] = psip2*sqrt(psip2); //separatrix
        double r2 = (y[0]-R_0_)*(y[0]-R_0_) + (y[1]-Z_0_)*(y[1]-Z_0_);
        double fieldT = psipZ*(y[1]-Z_0_)/r2 + psipR*(y[0]-R_0_)/r2; //fieldT
        yp[0] /=  fieldT;
        yp[1] /=  fieldT;
        yp[2] /=  fieldT;
    }
  private:
    double R_0_, Z_0_;
    CylindricalFunctorsLvl1 psip_;
    CylindricalSymmTensorLvl1 chi_;
};

struct FieldRZYZ
{
    FieldRZYZ( const CylindricalFunctorsLvl1& psip, const CylindricalSymmTensorLvl1& chi = CylindricalSymmTensorLvl1() ): psip_(psip), chi_(chi){}
    void operator()(double, const std::array<double,3>& y, std::array<double,3>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double psip2 =   chi_.xx()(y[0], y[1])*psipR*psipR
                    + 2.*chi_.xy()(y[0], y[1])*psipR*psipZ
                       + chi_.yy()(y[0], y[1])*psipZ*psipZ;
        yp[0] = -psipZ;//fieldR
        yp[1] = +psipR;//fieldZ
        //yp[2] = 1.0; //volume
        yp[2] = sqrt(psip2); //equalarc
        //yp[2] = psip2; //ribeiro
        //yp[2] = psip2*sqrt(psip2); //separatrix
        yp[0] /=  yp[1];
        yp[2] /=  yp[1];
        yp[1] =  1.;
    }
  private:
    CylindricalFunctorsLvl1 psip_;
    CylindricalSymmTensorLvl1 chi_;
};

struct FieldRZY
{
    FieldRZY( const CylindricalFunctorsLvl1& psip, const CylindricalSymmTensorLvl1& chi = CylindricalSymmTensorLvl1() ): f_(1.), psip_(psip), chi_(chi){}
    void set_f(double f){ f_ = f;}
    void operator()(double, const std::array<double,2>& y, std::array<double,2>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double psip2 =   chi_.xx()(y[0], y[1])*psipR*psipR
                    + 2.*chi_.xy()(y[0], y[1])*psipR*psipZ
                       + chi_.yy()(y[0], y[1])*psipZ*psipZ;
        //yp[0] = +psipZ/f_;//volume
        //yp[1] = -psipR/f_;//volume
        yp[0] = -psipZ/sqrt(psip2)/f_;//equalarc
        yp[1] = +psipR/sqrt(psip2)/f_;//equalarc
        //yp[0] = -psipZ/psip2/f_;//ribeiro
        //yp[1] = +psipR/psip2/f_;//ribeiro
        //yp[0] = +psipZ/psip2/sqrt(psip2)/f_;//separatrix
        //yp[1] = -psipR/psip2/sqrt(psip2)/f_;//separatrix
    }
  private:
    double f_;
    CylindricalFunctorsLvl1 psip_;
    CylindricalSymmTensorLvl1 chi_;
};


struct FieldRZYRYZY
{
    FieldRZYRYZY( const CylindricalFunctorsLvl2& psip):  psip_(psip){ f_ = f_prime_ = 1.;}
    void set_f( double new_f){ f_ = new_f;}
    void set_fp( double new_fp){ f_prime_ = new_fp;}
    void initialize( double R0, double Z0, double& yR, double& yZ)
    {
        double psipR = psip_.dfx()(R0, Z0), psipZ = psip_.dfy()(R0,Z0);
        double psip2 = (psipR*psipR+ psipZ*psipZ);
        yR = -f_*psipZ/sqrt(psip2);
        yZ = +f_*psipR/sqrt(psip2);
    }
    void derive( double R0, double Z0, double& xR, double& xZ)
    {
        xR = +f_*psip_.dfx()(R0, Z0);
        xZ = +f_*psip_.dfy()(R0, Z0);
    }

    void operator()(double, const std::array<double,4>& y, std::array<double,4>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double psipRR = psip_.dfxx()(y[0], y[1]), psipRZ = psip_.dfxy()(y[0],y[1]), psipZZ = psip_.dfyy()(y[0],y[1]);
        double psip2 = (psipR*psipR+ psipZ*psipZ);

        yp[0] =  -psipZ/f_/sqrt(psip2);
        yp[1] =  +psipR/f_/sqrt(psip2);
        yp[2] =  ( +psipRZ*y[2] - psipRR *y[3])/f_/sqrt(psip2)
            + f_prime_/f_* psipR + (psipR*psipRR + psipZ*psipRZ)/psip2 ;
        yp[3] =  ( -psipRZ*y[3] + psipZZ*y[2])/f_/sqrt(psip2)
            + f_prime_/f_* psipZ + (psipR*psipRZ + psipZ*psipZZ)/psip2;
    }
  private:
    double f_, f_prime_;
    CylindricalFunctorsLvl2 psip_;
};

}//namespace equalarc

struct FieldRZtau
{
    FieldRZtau(const CylindricalFunctorsLvl1& psip, const CylindricalSymmTensorLvl1& chi = CylindricalSymmTensorLvl1()): psip_(psip), chi_(chi){}
    void operator()(double, const std::array<double,2>& y, std::array<double,2>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0],y[1]);
        double chiRR = chi_.xx()(y[0], y[1]),
               chiRZ = chi_.xy()(y[0], y[1]),
               chiZZ = chi_.yy()(y[0], y[1]);
        double psip2 =   chiRR*psipR*psipR + 2.*chiRZ*psipR*psipZ + chiZZ*psipZ*psipZ;
        yp[0] =  (chiRR*psipR + chiRZ*psipZ)/psip2;
        yp[1] =  (chiRZ*psipR + chiZZ*psipZ)/psip2;
    }
  private:
    CylindricalFunctorsLvl1 psip_;
    CylindricalSymmTensorLvl1 chi_;
};

struct HessianRZtau
{
    HessianRZtau( const CylindricalFunctorsLvl2& psip): norm_(false), quad_(1), psip_(psip){}
    // if true goes into positive Z - direction and X else
    void set_quadrant( int quadrant) {quad_ = quadrant;}
    void set_norm( bool normed) {norm_ = normed;}
    void operator()(double, const std::array<double,2>& y, std::array<double,2>& yp) const
    {
        double psipRZ = psip_.dfxy()(y[0], y[1]);
        if( psipRZ == 0)
        {
            if(      quad_ == 0) { yp[0] = 1; yp[1] = 0; }
            else if( quad_ == 1) { yp[0] = 0; yp[1] = 1; }
            else if( quad_ == 2) { yp[0] = -1; yp[1] = 0; }
            else if( quad_ == 3) { yp[0] = 0; yp[1] = -1; }
        }
        else
        {
            double psipRR = psip_.dfxx()(y[0], y[1]), psipZZ = psip_.dfyy()(y[0],y[1]);
            double T = psipRR + psipZZ;
            double D = psipZZ*psipRR - psipRZ*psipRZ;
            double L1 = 0.5*T+sqrt( 0.25*T*T-D); // > 0
            double L2 = 0.5*T-sqrt( 0.25*T*T-D); // < 0;  D = L1*L2
            if      ( quad_ == 0){ yp[0] =  L1 - psipZZ; yp[1] = psipRZ;}
            else if ( quad_ == 1){ yp[0] = -psipRZ; yp[1] = psipRR - L2;}
            else if ( quad_ == 2){ yp[0] =  psipZZ - L1; yp[1] = -psipRZ;}
            else if ( quad_ == 3){ yp[0] = +psipRZ; yp[1] = L2 - psipRR;}
        }
        if( norm_)
        {
            double vgradpsi = yp[0]*psip_.dfx()(y[0],y[1]) + yp[1]*psip_.dfy()(y[0],y[1]);
            yp[0] /= vgradpsi, yp[1] /= vgradpsi;
        }
        else
        {
            double norm = sqrt(yp[0]*yp[0]+yp[1]*yp[1]);
            yp[0]/= norm, yp[1]/= norm;
        }

    }
  private:
    bool norm_;
    int quad_;
    CylindricalFunctorsLvl2 psip_;
};

struct MinimalCurve
{
    MinimalCurve(const CylindricalFunctorsLvl1& psip): norm_(false),
        psip_(psip){}
    void set_norm( bool normed) {norm_ = normed;}
    void operator()(double, const std::array<double,4>& y, std::array<double,4>& yp) const
    {
        double psipR = psip_.dfx()(y[0], y[1]), psipZ = psip_.dfy()(y[0], y[1]);
        yp[0] = y[2];
        yp[1] = y[3];
        //double psipRZ = psipRZ_(y[0], y[1]), psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0], y[1]), psipRR=psipRR_(y[0], y[1]), psipZZ=psipZZ_(y[0], y[1]);
        //double D2 = psipRR*y[2]*y[2] + 2.*psipRZ*y[2]*y[3] + psipZZ*y[3]*y[3];
        //double grad2 = psipR*psipR+psipZ*psipZ;
        //yp[2] = D2/(1.+grad2) * psipR ;
        //yp[3] = D2/(1.+grad2) * psipZ ;
        if( psip_.f()(y[0], y[1]) < 0)
        {
            yp[2] = -10.*psipR;
            yp[3] = -10.*psipZ;
        }
        else
        {
            yp[2] = 10.*psipR;
            yp[3] = 10.*psipZ;
        }

        if( norm_)
        {
            double vgradpsi = y[2]*psipR + y[3]*psipZ;
            yp[0] /= vgradpsi, yp[1] /= vgradpsi, yp[2] /= vgradpsi, yp[3] /= vgradpsi;
        }
    }
  private:
    bool norm_;
    CylindricalFunctorsLvl1 psip_;
};

////////////////////////////////////////////////////////////////////////////////


namespace detail
{
//compute psi(x) and f(x) for given discretization of x and a fpsiMinv functor
//doesn't integrate over the x-point
//returns psi_1
template <class FieldFinv>
void construct_psi_values( FieldFinv fpsiMinv,
        const double psi_0, const double psi_1, const double x_0, const thrust::host_vector<double>& x_vec, const double x_1,
        thrust::host_vector<double>& psi_x,
        thrust::host_vector<double>& f_x_, bool verbose = false)
{
    f_x_.resize( x_vec.size()), psi_x.resize( x_vec.size());
    double begin(psi_0), end(begin), temp(begin);
    unsigned N = 1;
    double eps = 1e10, eps_old=2e10;
    if(verbose)std::cout << "In psi function:\n";
    double x0=x_0, x1 = psi_1>psi_0? x_vec[0]:-x_vec[0];
    dg::SinglestepTimeloop<double> odeint( dg::RungeKutta<double>(
                "Feagin-17-8-10",0.), fpsiMinv);
    while( (eps <  eps_old || eps > 1e-8) && eps > 1e-14) //1e-8 < eps < 1e-14
    {
        eps_old = eps;
        x0 = x_0, x1 = x_vec[0];
        if( psi_1<psi_0) x1*=-1;
        odeint.integrate_steps( x0, begin, x1, end, N);
        psi_x[0] = end; fpsiMinv(0.,end,temp); f_x_[0] = temp;
        for( unsigned i=1; i<x_vec.size(); i++)
        {
            temp = end;
            x0 = x_vec[i-1], x1 = x_vec[i];
            if( psi_1<psi_0) x0*=-1, x1*=-1;
            odeint.integrate_steps( x0, temp, x1, end, N);
            psi_x[i] = end; fpsiMinv(0.,end,temp); f_x_[i] = temp;
        }
        temp = end;
        odeint.integrate_steps(x1, temp, psi_1>psi_0?x_1:-x_1, end,N);
        double psi_1_numerical = end;
        eps = fabs( psi_1_numerical-psi_1);
        if(verbose)std::cout << "Effective Psi error is "<<eps<<" with "<<N<<" steps\n";
        N*=2;
    }

}

//compute the vector of r and z - values that form one psi surface
//assumes that the initial line is perpendicular
template <class Fpsi, class FieldRZYRYZY>
void compute_rzy(Fpsi fpsi, FieldRZYRYZY fieldRZYRYZY,
        double psi, const thrust::host_vector<double>& y_vec,
        thrust::host_vector<double>& r,
        thrust::host_vector<double>& z,
        thrust::host_vector<double>& yr,
        thrust::host_vector<double>& yz,
        thrust::host_vector<double>& xr,
        thrust::host_vector<double>& xz,
        double& R_0, double& Z_0, double& f, double& fp, bool verbose = false )
{
    thrust::host_vector<double> r_old(y_vec.size(), 0), r_diff( r_old), yr_old(r_old), xr_old(r_old);
    thrust::host_vector<double> z_old(y_vec.size(), 0), z_diff( z_old), yz_old(r_old), xz_old(z_old);
    r.resize( y_vec.size()), z.resize(y_vec.size()), yr.resize(y_vec.size()), yz.resize(y_vec.size()), xr.resize(y_vec.size()), xz.resize(y_vec.size());

    //now compute f and starting values
    std::array<double, 4> begin{ {0,0,0,0} }, end(begin), temp(begin);
    const double f_psi = fpsi.construct_f( psi, begin[0], begin[1]);
    fieldRZYRYZY.set_f(f_psi);
    fp = fpsi.f_prime( psi);
    fieldRZYRYZY.set_fp(fp);
    fieldRZYRYZY.initialize( begin[0], begin[1], begin[2], begin[3]);
    R_0 = begin[0], Z_0 = begin[1];
    if(verbose)std::cout <<f_psi<<" "<<" "<< begin[0] << " "<<begin[1]<<"\t";
    unsigned steps = 1;
    double eps = 1e10, eps_old=2e10;
    using Vec = std::array<double,4>;
    dg::SinglestepTimeloop<Vec> odeint( dg::RungeKutta<Vec>( "Feagin-17-8-10",
                begin), fieldRZYRYZY);
    while( eps < eps_old)
    {
        //begin is left const
        eps_old = eps, r_old = r, z_old = z, yr_old = yr, yz_old = yz, xr_old = xr, xz_old = xz;
        odeint.integrate_steps( 0, begin, y_vec[0], end, steps);
        r[0] = end[0], z[0] = end[1], yr[0] = end[2], yz[0] = end[3];
        fieldRZYRYZY.derive( r[0], z[0], xr[0], xz[0]);
        //std::cout <<end[0]<<" "<< end[1] <<"\n";
        for( unsigned i=1; i<y_vec.size(); i++)
        {
            temp = end;
            odeint.integrate_steps( y_vec[i-1], temp, y_vec[i], end, steps);
            r[i] = end[0], z[i] = end[1], yr[i] = end[2], yz[i] = end[3];
            fieldRZYRYZY.derive( r[i], z[i], xr[i], xz[i]);
        }
        //compute error in R,Z only
        dg::blas1::axpby( 1., r, -1., r_old, r_diff);
        dg::blas1::axpby( 1., z, -1., z_old, z_diff);
        double er = dg::blas1::dot( r_diff, r_diff);
        double ez = dg::blas1::dot( z_diff, z_diff);
        double ar = dg::blas1::dot( r, r);
        double az = dg::blas1::dot( z, z);
        eps =  sqrt( er + ez)/sqrt(ar+az);
        //std::cout << "rel. error is "<<eps<<" with "<<steps<<" steps\n";
        steps*=2;
    }
    r = r_old, z = z_old, yr = yr_old, yz = yz_old, xr = xr_old, xz = xz_old;
    f = f_psi;

}

} //namespace detail
} //namespace geo
} //namespace dg
///@endcond

