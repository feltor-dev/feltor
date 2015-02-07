#include <iostream>

#include <cusp/print.h>

#include "evaluation.cuh"
#include "dzs.cuh"
#include "functions.h"
#include "../blas2.h"
#include "../functors.h"
#include "../cg.h"
#include "interpolation.cuh"

struct InvB
{
    InvB( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi)
    {
        return 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
    }
    private:
    double R_0, I_0;
};
struct B
{
    B( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi)
    {
        return R_0*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/2./sqrt(2.)/R;
    }
    private:
    double R_0, I_0;
};
struct LnB
{
    LnB( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi)
    {
        double invB = 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
        return log(1./invB);
    }
    private:
    double R_0, I_0;
};

struct Field
{
    Field( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
        double B = R_0*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0][i]-R_0))*cos(M_PI*y[1][i]))/2./sqrt(2.)/y[0][i];
        double dldp=y[0][i]*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0][i]-R_0))*cos(M_PI*y[1][i]))/2./sqrt(2)/I_0;
        yp[2][i] = B/dldp;            
        yp[0][i] = -B*M_PI*y[0][i]*cos(M_PI*(y[0][i]-R_0)/2.)*sin(M_PI*y[1][i]/2)/2./I_0/dldp;
        yp[1][i] =  B*M_PI*y[0][i]*sin(M_PI*(y[0][i]-R_0)/2.)*cos(M_PI*y[1][i]/2)/2./I_0/dldp ;
        }
    }
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
        double B = R_0*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0]-R_0))*cos(M_PI*y[1]))/2./sqrt(2.)/y[0];
        double dldp = y[0]*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0]-R_0))*cos(M_PI*y[1]))/2./sqrt(2.)/I_0;
        yp[2] = B/dldp;            
        yp[0] = -B*M_PI*y[0]*cos(M_PI*(y[0]-R_0)/2.)*sin(M_PI*y[1]/2)/2./I_0/dldp;
        yp[1] =  B*M_PI*y[0]*sin(M_PI*(y[0]-R_0)/2.)*cos(M_PI*y[1]/2)/2./I_0/dldp ;
    }
    private:
    double R_0, I_0;
};

struct FieldP
{
    FieldP( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
        double B = R_0*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0][i]-R_0))*cos(M_PI*y[1][i]))/2./sqrt(2.)/y[0][i];
        double dldp=y[0][i]*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0][i]-R_0))*cos(M_PI*y[1][i]))/2./sqrt(2)/I_0;
        yp[2][i] = B*dldp;            
        yp[0][i] = -B*M_PI*y[0][i]*cos(M_PI*(y[0][i]-R_0)/2.)*sin(M_PI*y[1][i]/2)/2./I_0;
        yp[1][i] =  B*M_PI*y[0][i]*sin(M_PI*(y[0][i]-R_0)/2.)*cos(M_PI*y[1][i]/2)/2./I_0 ;
        }
    }
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
        double B = R_0*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0]-R_0))*cos(M_PI*y[1]))/2./sqrt(2.)/y[0];
        double dldp = y[0]*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0]-R_0))*cos(M_PI*y[1]))/2./sqrt(2.)/I_0;
        yp[2] = B*dldp;            
        yp[0] = -B*M_PI*y[0]*cos(M_PI*(y[0]-R_0)/2.)*sin(M_PI*y[1]/2)/2./I_0;
        yp[1] =  B*M_PI*y[0]*sin(M_PI*(y[0]-R_0)/2.)*cos(M_PI*y[1]/2)/2./I_0 ;
    }
    private:
    double R_0, I_0;
};
double R_0 = 10;
double I_0 = 20; //I0=20 and R=10 means q=2

double divb(double R, double Z, double phi)
{
    double fac1 = sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
    double z1 = cos(M_PI*0.5*(R-R_0))*(32.*I_0*I_0+5.*M_PI*M_PI)+
                M_PI*M_PI* cos(M_PI*3.*(R-R_0)/2.)+
                M_PI*R*sin(M_PI*3.*(R-R_0)/2.) ;
    double z2 = cos(M_PI*0.5*(R-R_0)) + 
                cos(M_PI*3*(R-R_0)/2) + 
                M_PI*R*sin(M_PI*0.5*(R-R_0));
    double nenner = fac1*fac1*fac1*2.*sqrt(2.)*R;
    double divb = -M_PI*(z1*sin(M_PI*Z*0.5)-z2*M_PI*M_PI*sin(M_PI*Z*3./2.))/(nenner);
    return divb;
}
double funcadj(double R, double Z, double phi)
{
    double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    return psi*(R*R*sin(Z*Z)+Z*Z*cos(R*Z*phi));
}
double funcNEU(double R, double Z, double phi)
{
    double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    return -psi*cos(phi);

}
double deriNEU(double R, double Z, double phi)
{
    double dldp = R*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/2./sqrt(2.)/I_0;
    double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    double invB = 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
    return psi*sin(phi)/dldp/invB;
    
}
double cut(double R, double Z, double phi)
{
    double psip = 0.5*((R-R_0)*(R-R_0)+Z*Z);
    if (psip >0.5*((0.9)*(0.9))) return 0.;
    if (psip <0.5*((0.1)*(0.1))) return 0.;
    return 1.;
}
int main()
{
    Field field( R_0, I_0);
    InvB invb(R_0, I_0);
    B Bfield(R_0, I_0);
    LnB lnB(R_0, I_0);

    std::cout << "Type n, Nx, Ny, Nz\n";
    //std::cout << "Note, that function is resolved exactly in R,Z for n > 2\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "q = " << I_0/R_0 << std::endl;
    double z0 = 0, z1 = 2.*M_PI;
    //double z0 = M_PI/2., z1 = 3./2.*M_PI;
    double bscale=5.0;
    double Rmin=R_0 - bscale;
    double Rmax=R_0 + bscale;
    double Zmin= - bscale;
    double Zmax= +bscale;
    dg::Grid3d<double> g3d( Rmin,Rmax, Zmin, Zmax, z0, z1,  n, Nx, Ny, Nz,dg::NEU, dg::NEU, dg::PER,dg::cylindrical);
    
    dg::DVec cutongrid = dg::evaluate( cut, g3d);

    const dg::DVec w3d = dg::create::weights( g3d);
    const dg::DVec v3d = dg::create::inv_weights( g3d);


// double hs = g3d.hz();
    double hs = g3d.hz()*(R_0+1)*(sqrt(2)/I_0/2);

    std::cout << "hz = " <<  g3d.hz() << std::endl;
    std::cout << "hs = " << hs << std::endl;
    dg::DZ<dg::DMatrix, dg::DVec> dzs( field, g3d, hs, 1e-4, dg::DefaultLimiter(), dg::NEU);
    dg::DVec func = dg::evaluate(funcNEU, g3d),dzsf(func),dzsfa(func);
    const dg::DVec funca = dg::evaluate(funcadj, g3d);
    const dg::DVec soldzsf = dg::evaluate(deriNEU, g3d);
    const dg::DVec Bfeld = dg::evaluate(Bfield, g3d);
    dg::DVec one = dg::evaluate( dg::one, g3d),dzsone(one),dzsTone(one),dzsTB(one);
    dzs.set_boundaries( dg::PER, 0, 0);

    dzs( one, dzsone); //dz(f)
    dzs( func, dzsf); //dz(f)
    dzs( funca, dzsfa); //dz(f)
    dzs.centeredT( one, dzsTone); //dz(f)
    dzs.centeredT( Bfeld, dzsTB); //dz(f)
    //cut
//     dg::blas1::pointwiseDot(cutongrid,dzsone,dzsone);
//     dg::blas1::pointwiseDot(cutongrid,dzsf,dzsf);
//     dg::blas1::pointwiseDot(cutongrid,dzsTone,dzsTone);
//     dg::blas1::pointwiseDot(cutongrid,func,func);

    double normdzsone  =dg::blas2::dot(dzsone, w3d,dzsone);
    double normdzsTone =dg::blas2::dot(dzsTone, w3d,dzsTone);
    double normdzsTB =dg::blas2::dot(dzsTB, w3d,dzsTB);
    double normonedzsf = dg::blas2::dot(one, w3d,dzsfa);
    double normfdzsone = dg::blas2::dot(funca, w3d,dzsone);
    double normfdzsTone = dg::blas2::dot(funca, w3d,dzsTone);
    
    std::cout << "--------------------testing dzs" << std::endl;
    double normsoldzsf = dg::blas2::dot( w3d, soldzsf);
    std::cout << "|| Solution ||   "<<sqrt( normsoldzsf)<<"\n";
    double errdzsf =dg::blas2::dot( w3d, dzsf);
    std::cout << "|| Derivative || "<<sqrt( errdzsf)<<"\n";
    dg::blas1::axpby( 1.,soldzsf, -1.,dzsf);
    errdzsf=dg::blas2::dot( w3d,dzsf);
    std::cout << "Relative Difference in dzs is "<< sqrt( errdzsf/normsoldzsf )<<"\n"; 
    
    std::cout << "--------------------testing dzs and dzsT " << std::endl;
    std::cout << "|| dzs(1) ||      "<<sqrt( normdzsone)<<"\n";
    std::cout << "|| dzsT(1) ||      "<<sqrt( normdzsTone)<<"\n";
    std::cout << "|| dzsT(B) ||      "<<sqrt( normdzsTB)<<"\n";
    
    std::cout << "--------------------testing adjointness " << std::endl;
    std::cout << "<1,dzs(f)>   = "<< normonedzsf <<"\n";
    std::cout << "<f,dzs(1)>   = "<< normfdzsone <<"\n";
    std::cout << "-<dzsT(1),f> = "<< -normfdzsTone<<"\n";
    std::cout << "Diff         = "<< normonedzsf+normfdzsTone<<"\n";   
    std::cout << "-------------------- " << std::endl;
    
    return 0;

}
