#pragma once

#include "dg/functors.h"
#include "magnetic_field.h"

/*!@file
 *
 * TokamakMagneticField objects
 */
namespace dg
{
namespace geo
{
///@cond
//For testing purposes only
///@addtogroup profiles
///@{
/////////////Test functors for DS////////////////////////

//exp(R-R_0)exp(Z)cos^2 (phi)
struct TestFunctionPsi2
{
    TestFunctionPsi2( const TokamakMagneticField& c, double a):R_0(c.R0()), a(a), c_(c){}
    double operator()(double R, double Z, double phi) const {
        return exp((R-R_0)/a)*exp(Z/a)*cos(phi)*cos(phi);
    }
    double dR( double R, double Z, double phi) const{
        return exp((R-R_0)/a)*exp(Z/a)*cos(phi)*cos(phi)/a;
    }
    double dRR( double R, double Z, double phi) const{
        return exp((R-R_0)/a)*exp(Z/a)*cos(phi)*cos(phi)/a/a;
    }
    double dRZ( double R, double Z, double phi) const{
        return exp((R-R_0)/a)*exp(Z/a)*cos(phi)*cos(phi)/a/a;
    }
    double dZ( double R, double Z, double phi) const{
        return exp((R-R_0)/a)*exp(Z/a)*cos(phi)*cos(phi)/a;
    }
    double dZZ( double R, double Z, double phi) const{
        return exp((R-R_0)/a)*exp(Z/a)*cos(phi)*cos(phi)/a/a;
    }
    double dP( double R, double Z, double phi) const{
        return exp((R-R_0)/a)*exp(Z/a)*(-2.*sin(phi)*cos(phi));
    }
    double dRP( double R, double Z, double phi) const{
        return exp((R-R_0)/a)*exp(Z/a)*(-2.*sin(phi)*cos(phi))/a;
    }
    double dZP( double R, double Z, double phi) const{
        return exp((R-R_0)/a)*exp(Z/a)*(-2.*sin(phi)*cos(phi))/a;
    }
    double dPP( double R, double Z, double phi) const{
        return exp((R-R_0)/a)*exp(Z/a)*(2.*sin(phi)*sin(phi)-2.*cos(phi)*cos(phi));
    }
    private:
    double R_0, a;
    TokamakMagneticField c_;
};

// (cos(M_PI*(R-R_0))+1)*(cos(M_PI*Z/2.)+1)*sin(phi);
struct TestFunctionDirNeu{
    TestFunctionDirNeu( const TokamakMagneticField& c){
        R_0 = c.R0();
    }
    double operator()(double R, double Z, double phi)const{
        return (cos(M_PI*(R-R_0))+1.)*(cos(M_PI*Z)+1.)*sin(phi);
    }
    double dR(double R, double Z, double phi)const{
        return -M_PI*sin(M_PI*(R-R_0))*(cos(M_PI*Z)+1.)*sin(phi);
    }
    double dZ(double R, double Z, double phi)const{
        return -M_PI*(cos(M_PI*(R-R_0))+1.)*sin(M_PI*Z)*sin(phi);
    }
    double dP(double R, double Z, double phi)const{
        return (cos(M_PI*(R-R_0))+1.)*(cos(M_PI*Z)+1.)*cos(phi);
    }
    double dRR(double R, double Z, double phi)const{
        return -M_PI*M_PI*cos(M_PI*(R-R_0))*(cos(M_PI*Z)+1.)*sin(phi);
    }
    double dZZ(double R, double Z, double phi)const{
        return -M_PI*M_PI*(cos(M_PI*(R-R_0))+1.)*cos(M_PI*Z)*sin(phi);
    }
    double dPP(double R, double Z, double phi)const{
        return -(cos(M_PI*(R-R_0))+1.)*(cos(M_PI*Z)+1.)*sin(phi);
    }
    double dRZ(double R, double Z, double phi)const{
        return M_PI*M_PI*sin(M_PI*(R-R_0))*sin(M_PI*Z)*sin(phi);
    }
    double dRP(double R, double Z, double phi)const{
        return -M_PI*sin(M_PI*(R-R_0))*(cos(M_PI*Z)+1.)*cos(phi);
    }
    double dZP(double R, double Z, double phi)const{
        return -M_PI*(cos(M_PI*(R-R_0))+1.)*sin(M_PI*Z)*cos(phi);
    }
    private:
    double R_0;
};

/////With the next functors compute derivatives
// b \nabla f
template<class Function>
struct DsFunction
{
    DsFunction( const TokamakMagneticField& c, Function f): f_(f), c_(c),
        bhatR_(c), bhatZ_(c), bhatP_(c){}
    double operator()(double R, double Z, double phi) const {
        return bhatR_(R,Z)*f_.dR(R,Z,phi) +
               bhatZ_(R,Z)*f_.dZ(R,Z,phi) +
               bhatP_(R,Z)*f_.dP(R,Z,phi);
    }
    private:
    Function f_;
    TokamakMagneticField c_;
    dg::geo::BHatR bhatR_;
    dg::geo::BHatZ bhatZ_;
    dg::geo::BHatP bhatP_;
};
//\nabla( b f)
template<class Function>
struct DsDivFunction
{
    DsDivFunction( const TokamakMagneticField& c, Function f):
        f_(f), dsf_(c,f), divb_(c){}
    double operator()(double R, double Z, double phi) const {
        return f_(R,Z,phi)*divb_(R,Z) + dsf_(R,Z,phi);
    }
    private:
    Function f_;
    DsFunction<Function> dsf_;
    dg::geo::Divb divb_;
};

//2nd derivative \nabla_\parallel^2
template<class Function>
struct DssFunction
{
    DssFunction( TokamakMagneticField c, Function f):f_(f), c_(c),
        bhatR_(c), bhatZ_(c), bhatP_(c),
        bhatRR_(c), bhatZR_(c), bhatPR_(c),
        bhatRZ_(c), bhatZZ_(c), bhatPZ_(c){}
    double operator()(double R, double Z, double phi) const {
        double bhatR = bhatR_(R,Z), bhatZ = bhatZ_(R,Z), bhatP = bhatP_(R,Z);
        double bhatRR = bhatRR_(R,Z), bhatZR = bhatZR_(R,Z), bhatPR = bhatPR_(R,Z);
        double bhatRZ = bhatRZ_(R,Z), bhatZZ = bhatZZ_(R,Z), bhatPZ = bhatPZ_(R,Z);
        double fR = f_.dR(R,Z,phi), fZ = f_.dZ(R,Z,phi), fP = f_.dP(R,Z,phi);
        double fRR = f_.dRR(R,Z,phi), fRZ = f_.dRZ(R,Z,phi), fZZ = f_.dZZ(R,Z,phi);
        double fRP = f_.dRP(R,Z,phi), fZP = f_.dZP(R,Z,phi), fPP = f_.dPP(R,Z,phi);
        double gradbhatR = bhatR*bhatRR+bhatZ*bhatRZ,
               gradbhatZ = bhatR*bhatZR+bhatZ*bhatZZ,
               gradbhatP = bhatR*bhatPR+bhatZ*bhatPZ;
        return bhatR*bhatR*fRR + bhatZ*bhatZ*fZZ + bhatP*bhatP*fPP
            +2.*(bhatR*bhatZ*fRZ + bhatR*bhatP*fRP + bhatZ*bhatP*fZP)
            + gradbhatR*fR + gradbhatZ*fZ + gradbhatP*fP;
    }
    private:
    Function f_;
    TokamakMagneticField c_;
    dg::geo::BHatR bhatR_;
    dg::geo::BHatZ bhatZ_;
    dg::geo::BHatP bhatP_;
    dg::geo::BHatRR bhatRR_;
    dg::geo::BHatZR bhatZR_;
    dg::geo::BHatPR bhatPR_;
    dg::geo::BHatRZ bhatRZ_;
    dg::geo::BHatZZ bhatZZ_;
    dg::geo::BHatPZ bhatPZ_;
};

//positive Laplacian \Delta_\parallel
template<class Function>
struct DsDivDsFunction
{
    DsDivDsFunction( const TokamakMagneticField& c,Function f): dsf_(c,f), dssf_(c,f), divb_(c){}
    double operator()(double R, double Z, double phi) const {
        return divb_(R,Z)*dsf_(R,Z,phi) + dssf_(R,Z,phi);
    }
    private:
    DsFunction<Function> dsf_;
    DssFunction<Function> dssf_;
    dg::geo::Divb divb_;
};

//positive perp Laplacian \Delta_\perp
template<class Function>
struct DPerpFunction
{
    DPerpFunction( const TokamakMagneticField& c, Function f): f_(f), dsf_(c,f){}
    double operator()(double R, double Z, double phi) const {
        return f_.dR(R,Z,phi)/R + f_.dRR(R,Z,phi) + f_.dZZ(R,Z,phi) + f_.dPP(R,Z,phi)/R/R - dsf_(R,Z,phi);
    }
    private:
    Function f_;
    DsDivDsFunction<Function> dsf_;
};

template<class Function>
struct OMDsDivDsFunction
{
    OMDsDivDsFunction( const TokamakMagneticField& c,Function f): f_(f), df_(c,f){}
    double operator()(double R, double Z, double phi) const {
        return f_(R,Z,phi)-df_(R,Z,phi);
    }
    private:
    Function f_;
    DsDivDsFunction<Function> df_;
};

template<class Function>
struct Variation
{
    Variation( Function f): f_(f){}
    double operator()(double R, double Z, double phi) const {
        return sqrt(f_.dR(R,Z,phi)*f_.dR(R,Z,phi) + f_.dZ(R,Z,phi)*f_.dZ(R,Z,phi));
    }
    private:
    Function f_;
};

//////////////function to call DS////////////////////
template<class DS, class container>
void callDS( DS& ds, std::string name, const container& in, container& out,
unsigned max_iter = 1e4, double eps = 1e-6)
{
    if( name == "forward") ds.ds( dg::forward, in, out);
    else if( name == "backward") ds.ds( dg::backward, in, out);
    else if( name == "forward2") ds.forward2( 1., in, 0., out);
    else if( name == "backward2") ds.backward2( 1., in, 0., out);
    else if( name == "centered") ds.ds( dg::centered, in, out);
    else if( name == "dss") ds.dss( in, out);
    else if( name == "centered_bc_along")
        ds.centered_bc_along_field( 1., in, 0., out, ds.fieldaligned().bcx(), {0,0});
    else if( name == "dss_bc_along")
        ds.dss_bc_along_field( 1., in, 0., out, ds.fieldaligned().bcx(), {0,0});
    else if( name == "centered") ds.ds( dg::centered, in, out);
    else if( name == "dss") ds.dss( in, out);
    else if( name == "divForward") ds.div( dg::forward, in, out);
    else if( name == "divBackward") ds.div( dg::backward, in, out);
    else if( name == "divCentered") ds.div( dg::centered, in, out);
    else if( name == "directLap") {
        ds.dssd( 1., in, 0., out);
    }
    else if( name == "directLap_bc_along") {
        ds.dssd_bc_along_field( 1., in, 0., out, ds.fieldaligned().bcx(), {0,0});
    }
    else if( name == "invCenteredLap"){
        //dg::LGMRES<container> invert( in, 30,3,10000);
        dg::BICGSTABl<container> invert( in, max_iter, 3);
        dg::Timer t;
        t.tic();
        double precond = 1.;
        unsigned number = invert.solve( [&](const auto& x, auto& y){
                //  y = ( 1 - D) x
                dg::blas2::symv( ds, x, y);
                dg::blas1::axpby( 1., x, -1., y, y);
            }, out, in, precond, ds.weights(), eps);
        t.toc();
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
    {
#endif //MPI
        std::cout << "#Number of BICGSTABl iterations: "<<number<<"\n";
        std::cout << "#Took                          : "<<t.diff()<<"\n";
#ifdef MPI_VERSION
    }
#endif //MPI
        return;
    }

}
///////////////////////////////Functions for 2d grids//////////////////

//psi * cos(theta)
struct FuncDirPer
{
    FuncDirPer( const TokamakMagneticField& c, double psi_0, double psi_1, double k):
        R_0_(c.R0()), psi0_(psi_0), psi1_(psi_1), k_(k), c_(c) {}
    double operator()(double R, double Z) const {
        double psip = c_.psip()(R,Z);
        double result = (psip-psi0_)*(psip-psi1_)*cos(k_*theta(R,Z));
        return 0.1*result;
    }
    double operator()(double R, double Z, double) const {
        return operator()(R,Z);
    }
    double dR( double R, double Z)const
    {
        double psip = c_.psip()(R,Z), psipR = c_.psipR()(R,Z), theta_ = k_*theta(R,Z);
        double result = (2.*psip*psipR - (psi0_+psi1_)*psipR)*cos(theta_)
            - (psip-psi0_)*(psip-psi1_)*sin(theta_)*k_*thetaR(R,Z);
        return 0.1*result;
    }
    double dRR( double R, double Z)const
    {
        double psip = c_.psip()(R,Z), psipR = c_.psipR()(R,Z), theta_=k_*theta(R,Z), thetaR_=k_*thetaR(R,Z);
        double psipRR = c_.psipRR()(R,Z);
        double result = (2.*(psipR*psipR + psip*psipRR) - (psi0_+psi1_)*psipRR)*cos(theta_)
            - 2.*(2.*psip*psipR-(psi0_+psi1_)*psipR)*sin(theta_)*thetaR_
            - (psip-psi0_)*(psip-psi1_)*(k_*thetaRR(R,Z)*sin(theta_)+cos(theta_)*thetaR_*thetaR_);
        return 0.1*result;

    }
    double dZ( double R, double Z)const
    {
        double psip = c_.psip()(R,Z), psipZ = c_.psipZ()(R,Z), theta_=k_*theta(R,Z);
        double result = (2*psip*psipZ - (psi0_+psi1_)*psipZ)*cos(theta_)
            - (psip-psi0_)*(psip-psi1_)*sin(theta_)*k_*thetaZ(R,Z);
        return 0.1*result;
    }
    double dZZ( double R, double Z)const
    {
        double psip = c_.psip()(R,Z), psipZ = c_.psipZ()(R,Z), theta_=k_*theta(R,Z), thetaZ_=k_*thetaZ(R,Z);
        double psipZZ = c_.psipZZ()(R,Z);
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
    const TokamakMagneticField c_;
};
// Variation of FuncDirPer
struct VariationDirPer
{
    VariationDirPer( dg::geo::TokamakMagneticField mag, double psi_0, double psi_1): m_f(mag, psi_0, psi_1,4. ){}
    double operator()(double R, double Z, double) const {
        return this->operator()(R,Z);}

    double operator()(double R, double Z) const {
        return m_f.dR( R,Z)*m_f.dR(R,Z) + m_f.dZ(R,Z)*m_f.dZ(R,Z);
    }
    private:
    dg::geo::FuncDirPer m_f;
};


//takes the magnetic field as chi
struct EllipticDirPerM
{
    EllipticDirPerM( const TokamakMagneticField& c, double psi_0, double psi_1, double k): func_(c, psi_0, psi_1, k), bmod_(c), br_(c), bz_(c) {}
    double operator()(double R, double Z, double) const {
        return operator()(R,Z);}
    double operator()(double R, double Z) const {
        double bmod = bmod_(R,Z), br = br_(R,Z), bz = bz_(R,Z);
        return -(br*func_.dR(R,Z) + bz*func_.dZ(R,Z) + bmod*(func_.dRR(R,Z) + func_.dZZ(R,Z) ));

    }
    private:
    FuncDirPer func_;
    dg::geo::Bmodule bmod_;
    dg::geo::BR br_;
    dg::geo::BZ bz_;
};

//Blob function
struct FuncDirNeu
{
    FuncDirNeu( const TokamakMagneticField&, double psi_0, double psi_1, double R_blob, double Z_blob, double sigma_blob, double amp_blob):
        psi0_(psi_0), psi1_(psi_1),
        cauchy_(R_blob, Z_blob, sigma_blob, sigma_blob, amp_blob){}

    double operator()(double R, double Z, double) const {
        return operator()(R,Z);}
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
struct BmodTheta
{
    BmodTheta( const TokamakMagneticField& c): R_0_(c.R0()), bmod_(c){}
    double operator()(double R,double Z, double) const{
        return operator()(R,Z);}
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
    dg::geo::Bmodule bmod_;

};

//take BmodTheta as chi
struct EllipticDirNeuM
{
    EllipticDirNeuM( const TokamakMagneticField& c, double psi_0, double psi_1, double R_blob, double Z_blob, double sigma_blob, double amp_blob): R_0_(c.R0()),
    func_(c, psi_0, psi_1, R_blob, Z_blob, sigma_blob,amp_blob), bmod_(c), br_(c), bz_(c) {}
    double operator()(double R, double Z) const {
        double bmod = bmod_(R,Z), br = br_(R,Z), bz = bz_(R,Z), theta_ = theta(R,Z);
        double chi = bmod*(1.+0.5*sin(theta_));
        double chiR = br*(1.+0.5*sin(theta_)) + bmod*0.5*cos(theta_)*thetaR(R,Z);
        double chiZ = bz*(1.+0.5*sin(theta_)) + bmod*0.5*cos(theta_)*thetaZ(R,Z);
        return -(chiR*func_.dR(R,Z) + chiZ*func_.dZ(R,Z) + chi*( func_.dRR(R,Z) + func_.dZZ(R,Z) ));

    }
    double operator()(double R, double Z, double) const {
        return operator()(R,Z);
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
    FuncDirNeu func_;
    dg::geo::Bmodule bmod_;
    dg::geo::BR br_;
    dg::geo::BZ bz_;
};

//the psi surfaces
struct FuncXDirNeu
{
    FuncXDirNeu( const TokamakMagneticField& c, double psi_0, double psi_1):
        c_(c), psi0_(psi_0), psi1_(psi_1){}

    double operator()(double R, double Z, double) const {
        return operator()(R,Z);}
    double operator()(double R, double Z) const {
        double psip = c_.psip()(R,Z);
        return (psip-psi0_)*(psip-psi1_);
    }
    double dR( double R, double Z)const
    {
        double psip = c_.psip()(R,Z), psipR = c_.psipR()(R,Z);
        return (2.*psip-psi0_-psi1_)*psipR;
    }
    double dRR( double R, double Z)const
    {
        double psip = c_.psip()(R,Z), psipR = c_.psipR()(R,Z);
        double psipRR = c_.psipRR()(R,Z);
        return (2.*(psipR*psipR + psip*psipRR) - (psi0_+psi1_)*psipRR);

    }
    double dZ( double R, double Z)const
    {
        double psip = c_.psip()(R,Z), psipZ = c_.psipZ()(R,Z);
        return (2*psip-psi0_-psi1_)*psipZ;
    }
    double dZZ( double R, double Z)const
    {
        double psip = c_.psip()(R,Z), psipZ = c_.psipZ()(R,Z);
        double psipZZ = c_.psipZZ()(R,Z);
        return (2.*(psipZ*psipZ + psip*psipZZ) - (psi0_+psi1_)*psipZZ);
    }
    private:
    TokamakMagneticField c_;
    double psi0_, psi1_;
};

//take Bmod as chi
struct EllipticXDirNeuM
{
    EllipticXDirNeuM( const TokamakMagneticField& c, double psi_0, double psi_1): R_0_(c.R0()),
    func_(c, psi_0, psi_1), bmod_(c), br_(c), bz_(c) {}
    double operator()(double R, double Z) const {
        double bmod = bmod_(R,Z), br = br_(R,Z), bz = bz_(R,Z);
        //double chi = 1e4+bmod; //bmod can be zero for a Taylor state(!)
        double chi = bmod; //bmod for solovev state
        double chiR = br;
        double chiZ = bz;
        return -(chiR*func_.dR(R,Z) + chiZ*func_.dZ(R,Z) + chi*( func_.dRR(R,Z) + func_.dZZ(R,Z) ));
        //return -( func_.dRR(R,Z) + func_.dZZ(R,Z) );

    }
    double operator()(double R, double Z, double) const {
        return operator()(R,Z);
    }
    private:
    double R_0_;
    FuncXDirNeu func_;
    dg::geo::Bmodule bmod_;
    dg::geo::BR br_;
    dg::geo::BZ bz_;
};

//take Blob and chi=1
struct EllipticBlobDirNeuM
{
    EllipticBlobDirNeuM( const TokamakMagneticField& c, double psi_0, double psi_1, double R_blob, double Z_blob, double sigma_blob, double amp_blob):
    func_(c, psi_0, psi_1, R_blob, Z_blob, sigma_blob, amp_blob){}
    double operator()(double R, double Z) const {
        return -( func_.dRR(R,Z) + func_.dZZ(R,Z) );
    }
    double operator()(double R, double Z, double) const {
        return operator()(R,Z);
    }
    private:
    double R_0_;
    FuncDirNeu func_;
};

struct EllipticDirSimpleM
{
    EllipticDirSimpleM( const TokamakMagneticField& c, double psi_0, double psi_1, double R_blob, double Z_blob, double sigma_blob, double amp_blob): func_(c, psi_0, psi_1, R_blob, Z_blob, sigma_blob, amp_blob) {}
    double operator()(double R, double Z, double) const {
        return -(( 1./R*func_.dR(R,Z) + func_.dRR(R,Z) + func_.dZZ(R,Z) ));

    }
    private:
    FuncDirNeu func_;
};

///@}
///@endcond
//
} //namespace functors
} //namespace dg
