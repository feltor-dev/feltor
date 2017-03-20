#pragma once

#include "dg/functors.h"

/*!@file
 *
 * Geometry objects 
 */
namespace dg
{
namespace fields
{
///@addtogroup profiles
///@{


template<class Collective>
struct FuncNeu
{
    FuncNeu( const Collective& c):c_(c){}
    double operator()(double R, double Z, double phi) const {return -c_.psip(R,Z)*cos(phi);
    }
    private:
    Collective c_;
};

template<class Collective>
struct DeriNeu
{
    DeriNeu( const Collective& c, double R0):c_(c), bhat_(c, R0){}
    double operator()(double R, double Z, double phi) const {return c_.psip(R,Z)*bhat_(R,Z,phi)*sin(phi);
    }
    private:
    Collective c_;
    BHatP<Collective> bhat_;
};

//psi * cos(theta)
template<class Collective>
struct FuncDirPer
{
    FuncDirPer( const Collective& c, double R0, double psi_0, double psi_1, double k):
        R_0_(R0), psi0_(psi_0), psi1_(psi_1), k_(k), c_(c) {}
    double operator()(double R, double Z) const {
        double psip = c_.psip(R,Z);
        double result = (psip-psi0_)*(psip-psi1_)*cos(k_*theta(R,Z));
        return 0.1*result;
    }
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);
    }
    double dR( double R, double Z)const
    {
        double psip = c_.psip(R,Z), psipR = c_.psipR(R,Z), theta_ = k_*theta(R,Z);
        double result = (2.*psip*psipR - (psi0_+psi1_)*psipR)*cos(theta_) 
            - (psip-psi0_)*(psip-psi1_)*sin(theta_)*k_*thetaR(R,Z);
        return 0.1*result;
    }
    double dRR( double R, double Z)const
    {
        double psip = c_.psip(R,Z), psipR = c_.psipR(R,Z), theta_=k_*theta(R,Z), thetaR_=k_*thetaR(R,Z);
        double psipRR = c_.psipRR(R,Z);
        double result = (2.*(psipR*psipR + psip*psipRR) - (psi0_+psi1_)*psipRR)*cos(theta_)
            - 2.*(2.*psip*psipR-(psi0_+psi1_)*psipR)*sin(theta_)*thetaR_
            - (psip-psi0_)*(psip-psi1_)*(k_*thetaRR(R,Z)*sin(theta_)+cos(theta_)*thetaR_*thetaR_);
        return 0.1*result;
            
    }
    double dZ( double R, double Z)const
    {
        double psip = c_.psip(R,Z), psipZ = c_.psipZ(R,Z), theta_=k_*theta(R,Z);
        double result = (2*psip*psipZ - (psi0_+psi1_)*psipZ)*cos(theta_) 
            - (psip-psi0_)*(psip-psi1_)*sin(theta_)*k_*thetaZ(R,Z);
        return 0.1*result;
    }
    double dZZ( double R, double Z)const
    {
        double psip = c_.psip(R,Z), psipZ = c_.psipZ(R,Z), theta_=k_*theta(R,Z), thetaZ_=k_*thetaZ(R,Z);
        double psipZZ = c_.psipZZ(R,Z);
        double result = (2.*(psipZ*psipZ + psip*psipZZ) - (psi0_+psi1_)*psipZZ)*cos(theta_)
            - 2.*(2.*psip*psipZ-(psi0_+psi1_)*psipZ)*sin(theta_)*thetaZ_
            - (psip-psi0_)*(psip-psi1_)*(k_*thetaZZ(R,Z)*sin(theta_) + cos(theta_)*thetaZ_*thetaZ_ );
        return 0.1*result;
    }
    private:
    double theta( double R, double Z) const {
        double dR = R-R_0_;
        if( Z >= 0)
            return acos( dR/sqrt( dR*dR + Z*Z));
        else
            return 2.*M_PI-acos( dR/sqrt( dR*dR + Z*Z));
    }
    double thetaR( double R, double Z) const {
        double dR = R-R_0_;
        return -Z/(dR*dR+Z*Z);
    }
    double thetaZ( double R, double Z) const {
        double dR = R-R_0_;
        return dR/(dR*dR+Z*Z);
    }
    double thetaRR( double R, double Z) const {
        double dR = R-R_0_;
        return 2*Z*dR/(dR*dR+Z*Z)/(dR*dR+Z*Z);
    }
    double thetaZZ( double R, double Z) const { return -thetaRR(R,Z);}
    double R_0_;
    double psi0_, psi1_, k_;
    const Collective& c_;
};

//takes the magnetic field as chi
template<class Collective>
struct EllipticDirPerM
{
    EllipticDirPerM( const Collective& c, double R0, double psi_0, double psi_1, double k): func_(c, R0, psi_0, psi_1, k), bmod_(c, R0), br_(c, R0), bz_(c, R0) {}
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);}
    double operator()(double R, double Z) const {
        double bmod = bmod_(R,Z), br = br_(R,Z), bz = bz_(R,Z);
        return -(br*func_.dR(R,Z) + bz*func_.dZ(R,Z) + bmod*(func_.dRR(R,Z) + func_.dZZ(R,Z) ));

    }
    private:
    FuncDirPer<Collective> func_;
    Bmodule<Collective> bmod_;
    BR<Collective> br_;
    BZ<Collective> bz_;
};

//Blob function
template<class Collective>
struct FuncDirNeu
{
    FuncDirNeu( const Collective& c, double psi_0, double psi_1, double R_blob, double Z_blob, double sigma_blob, double amp_blob):
        psi0_(psi_0), psi1_(psi_1), 
        cauchy_(R_blob, Z_blob, sigma_blob, sigma_blob, amp_blob){}

    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);}
    double operator()(double R, double Z) const {
        return cauchy_(R,Z);
        //double psip = psip_(R,Z);
        //return (psip-psi0_)*(psip-psi1_)+cauchy_(R,Z);
        //return (psip-psi0_)*(psip-psi1_);
    }
    double dR( double R, double Z)const
    {
        return cauchy_.dx(R,Z);
        //double psip = psip_(R,Z), psipR = psipR_(R,Z);
        //return (2.*psip-psi0_-psi1_)*psipR + cauchy_.dx(R,Z);
        //return (2.*psip-psi0_-psi1_)*psipR;
    }
    double dRR( double R, double Z)const
    {
        return cauchy_.dxx(R,Z);
        //double psip = psip_(R,Z), psipR = psipR_(R,Z);
        //double psipRR = psipRR_(R,Z);
        //return (2.*(psipR*psipR + psip*psipRR) - (psi0_+psi1_)*psipRR)+cauchy_.dxx(R,Z);
        //return (2.*(psipR*psipR + psip*psipRR) - (psi0_+psi1_)*psipRR);
            
    }
    double dZ( double R, double Z)const
    {
        return cauchy_.dy(R,Z);
        //double psip = psip_(R,Z), psipZ = psipZ_(R,Z);
        //return (2*psip-psi0_-psi1_)*psipZ+cauchy_.dy(R,Z);
        //return (2*psip-psi0_-psi1_)*psipZ;
    }
    double dZZ( double R, double Z)const
    {
        return cauchy_.dyy(R,Z);
        //double psip = psip_(R,Z), psipZ = psipZ_(R,Z);
        //double psipZZ = psipZZ_(R,Z);
        //return (2.*(psipZ*psipZ + psip*psipZZ) - (psi0_+psi1_)*psipZZ)+cauchy_.dyy(R,Z);
        //return (2.*(psipZ*psipZ + psip*psipZZ) - (psi0_+psi1_)*psipZZ);
    }
    private:
    double psi0_, psi1_;
    dg::Cauchy cauchy_;
};


//takes the magnetic field multiplied by (1+0.5sin(theta)) as chi
template<class Collective>
struct BmodTheta
{
    BmodTheta( const Collective& c, double R0): R_0_(R0), bmod_(c, R0){}
    double operator()(double R,double Z, double phi) const{
        return this->operator()(R,Z);}
    double operator()(double R,double Z) const{
        return bmod_(R,Z)*(1.+0.5*sin(theta(R,Z)));
    }
    private:
    double theta( double R, double Z) const {
        double dR = R-R_0_;
        if( Z >= 0)
            return acos( dR/sqrt( dR*dR + Z*Z));
        else
            return 2.*M_PI-acos( dR/sqrt( dR*dR + Z*Z));
    }
    double R_0_;
    Bmodule<Collective> bmod_;

};

//take BmodTheta as chi
template<class Collective>
struct EllipticDirNeuM
{
    EllipticDirNeuM( const Collective& c, double R0, double psi_0, double psi_1, double R_blob, double Z_blob, double sigma_blob, double amp_blob): R_0_(R0), 
    func_(c, psi_0, psi_1, R_blob, Z_blob, sigma_blob,amp_blob), bmod_(c, R0), br_(c, R0), bz_(c, R0) {}
    double operator()(double R, double Z) const {
        double bmod = bmod_(R,Z), br = br_(R,Z), bz = bz_(R,Z), theta_ = theta(R,Z);
        double chi = bmod*(1.+0.5*sin(theta_));
        double chiR = br*(1.+0.5*sin(theta_)) + bmod*0.5*cos(theta_)*thetaR(R,Z);
        double chiZ = bz*(1.+0.5*sin(theta_)) + bmod*0.5*cos(theta_)*thetaZ(R,Z);
        return -(chiR*func_.dR(R,Z) + chiZ*func_.dZ(R,Z) + chi*( func_.dRR(R,Z) + func_.dZZ(R,Z) ));

    }
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);
    }
    private:
    double theta( double R, double Z) const {
        double dR = R-R_0_;
        if( Z >= 0)
            return acos( dR/sqrt( dR*dR + Z*Z));
        else
            return 2.*M_PI-acos( dR/sqrt( dR*dR + Z*Z));
    }
    double thetaR( double R, double Z) const {
        double dR = R-R_0_;
        return -Z/(dR*dR+Z*Z);
    }
    double thetaZ( double R, double Z) const {
        double dR = R-R_0_;
        return dR/(dR*dR+Z*Z);
    }
    double R_0_;
    FuncDirNeu<Collective> func_;
    Bmodule<Collective> bmod_;
    BR<Collective> br_;
    BZ<Collective> bz_;
};

//the psi surfaces
template<class Collective>
struct FuncXDirNeu
{
    FuncXDirNeu( const Collective& c, double psi_0, double psi_1):
        c_(c), psi0_(psi_0), psi1_(psi_1){}

    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);}
    double operator()(double R, double Z) const {
        double psip = c_.psip(R,Z);
        return (psip-psi0_)*(psip-psi1_);
    }
    double dR( double R, double Z)const
    {
        double psip = c_.psip(R,Z), psipR = c_.psipR(R,Z);
        return (2.*psip-psi0_-psi1_)*psipR;
    }
    double dRR( double R, double Z)const
    {
        double psip = c_.psip(R,Z), psipR = c_.psipR(R,Z);
        double psipRR = c_.psipRR(R,Z);
        return (2.*(psipR*psipR + psip*psipRR) - (psi0_+psi1_)*psipRR);
            
    }
    double dZ( double R, double Z)const
    {
        double psip = c_.psip(R,Z), psipZ = c_.psipZ(R,Z);
        return (2*psip-psi0_-psi1_)*psipZ;
    }
    double dZZ( double R, double Z)const
    {
        double psip = c_.psip(R,Z), psipZ = c_.psipZ(R,Z);
        double psipZZ = c_.psipZZ(R,Z);
        return (2.*(psipZ*psipZ + psip*psipZZ) - (psi0_+psi1_)*psipZZ);
    }
    private:
    Collective c_;
    double psi0_, psi1_;
};

//take Bmod as chi
template<class Collective>
struct EllipticXDirNeuM
{
    EllipticXDirNeuM( const Collective& c, double R0, double psi_0, double psi_1): R_0_(R0), 
    func_(c, psi_0, psi_1), bmod_(c, R0), br_(c, R0), bz_(c, R0) {}
    double operator()(double R, double Z) const {
        double bmod = bmod_(R,Z), br = br_(R,Z), bz = bz_(R,Z);
        double chi = 1e4+bmod; //bmod can be zero for a Taylor state(!)
        //double chi = bmod; //bmod can be zero for a Taylor state(!)
        double chiR = br;
        double chiZ = bz;
        return -(chiR*func_.dR(R,Z) + chiZ*func_.dZ(R,Z) + chi*( func_.dRR(R,Z) + func_.dZZ(R,Z) ));
        //return -( func_.dRR(R,Z) + func_.dZZ(R,Z) );

    }
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);
    }
    private:
    double R_0_;
    FuncXDirNeu<Collective> func_;
    Bmodule<Collective> bmod_;
    BR<Collective> br_;
    BZ<Collective> bz_;
};

//take Blob and chi=1
template<class Collective>
struct EllipticBlobDirNeuM
{
    EllipticBlobDirNeuM( const Collective& c, double psi_0, double psi_1, double R_blob, double Z_blob, double sigma_blob, double amp_blob): 
    func_(c, psi_0, psi_1, R_blob, Z_blob, sigma_blob, amp_blob){}
    double operator()(double R, double Z) const {
        return -( func_.dRR(R,Z) + func_.dZZ(R,Z) );
    }
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);
    }
    private:
    double R_0_;
    FuncDirNeu<Collective> func_;
};

template<class Collective>
struct EllipticDirSimpleM
{
    EllipticDirSimpleM( const Collective& c, double psi_0, double psi_1, double R_blob, double Z_blob, double sigma_blob, double amp_blob): func_(c, psi_0, psi_1, R_blob, Z_blob, sigma_blob, amp_blob) {}
    double operator()(double R, double Z, double phi) const {
        return -(( 1./R*func_.dR(R,Z) + func_.dRR(R,Z) + func_.dZZ(R,Z) ));

    }
    private:
    FuncDirNeu<Collective> func_;
};

/**
 * @brief testfunction to test the parallel derivative 
      \f[ f(R,Z,\varphi) = -\frac{\cos(\varphi)}{R\hat b_\varphi} \f]
 */ 
template<class Collective>
struct TestFunction
{
    TestFunction( const Collective& c, double R0) :  
        bhatR_(c, R0),
        bhatZ_(c, R0),
        bhatP_(c, R0) {}
    /**
     * @brief \f[ f(R,Z,\varphi) = -\frac{\cos(\varphi)}{R\hat b_\varphi} \f]
     */ 
    double operator()( double R, double Z, double phi)
    {
//         return psip_(R,Z,phi)*sin(phi);
//         double Rmin = gp_.R_0-(p_.boxscaleRm)*gp_.a;
//         double Rmax = gp_.R_0+(p_.boxscaleRp)*gp_.a;
//         double kR = 1.*M_PI/(Rmax - Rmin);
//         double Zmin = -(p_.boxscaleZm)*gp_.a*gp_.elongation;
//         double Zmax = (p_.boxscaleZp)*gp_.a*gp_.elongation;
//         double kZ = 1.*M_PI/(Zmax - Zmin);
        double kP = 1.;
//         return sin(phi*kP)*sin((R-Rmin)*kR)*sin((Z-Zmin)*kZ); //DIR
//         return cos(phi)*cos((R-Rmin)*kR)*cos((Z-Zmin)*kZ);
//         return sin(phi*kP); //DIR
//         return cos(phi*kP); //NEU
        return -cos(phi*kP)/bhatP_(R,Z,phi)/R; //NEU 2

    }
    private:
    BHatR<Collective> bhatR_;
    BHatZ<Collective> bhatZ_;
    BHatP<Collective> bhatP_;
};

/**
 * @brief analyitcal solution of the parallel derivative of the testfunction
 *  \f[ \nabla_\parallel(R,Z,\varphi) f = \frac{\sin(\varphi)}{R}\f]
 */ 
template<class Collective>
struct DeriTestFunction
{
    DeriTestFunction( const Collective& c, double R0) :
        bhatR_(c, R0),
        bhatZ_(c, R0),
        bhatP_(c, R0) {}
/**
 * @brief \f[ \nabla_\parallel f = \frac{\sin(\varphi)}{R}\f]
 */ 
    double operator()( double R, double Z, double phi)
    {
//         double Rmin = gp_.R_0-(p_.boxscaleRm)*gp_.a;
//         double Rmax = gp_.R_0+(p_.boxscaleRp)*gp_.a;
//         double kR = 1.*M_PI/(Rmax - Rmin);
//         double Zmin = -(p_.boxscaleZm)*gp_.a*gp_.elongation;
//         double Zmax = (p_.boxscaleZp)*gp_.a*gp_.elongation;
//         double kZ = 1.*M_PI/(Zmax - Zmin);
        double kP = 1.;
//          return (bhatR_(R,Z,phi)*sin(phi)*sin((Z-Zmin)*kZ)*cos((R-Rmin)*kR)*kR+
//                 bhatZ_(R,Z,phi)*sin(phi)*sin((R-Rmin)*kR)*cos((Z-Zmin)*kZ)*kZ+
//                 bhatP_(R,Z,phi)*cos(phi)*sin((R-Rmin)*kR)*sin((Z-Zmin)*kZ)*kP); //DIR
//         return -bhatR_(R,Z,phi)*cos(phi)*cos((Z-Zmin)*kZ)*sin((R-Rmin)*kR)*kR-
//                bhatZ_(R,Z,phi)*cos(phi)*cos((R-Rmin)*kR)*sin((Z-Zmin)*kZ)*kZ-
//                bhatP_(R,Z,phi)*sin(phi)*cos((R-Rmin)*kR)*cos((Z-Zmin)*kZ)*kP;
//         return  bhatP_(R,Z,phi)*cos(phi*kP)*kP; //DIR
//         return  -bhatP_(R,Z,phi)*sin(phi*kP)*kP; //NEU
        return sin(phi*kP)*kP/R; //NEU 2

    }
    private:
    BHatR<Collective> bhatR_;
    BHatZ<Collective> bhatZ_;
    BHatP<Collective> bhatP_;
};

///@} 
} //namespace fields
} //namespace dg
