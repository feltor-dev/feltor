#include <iostream>

#include <cusp/print.h>

#include "evaluation.cuh"
#include "dz.cuh"
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
//psi = cos(0.5*pi*(R-R_0))*cos(0.5*pi*Z)
struct Field
{
    Field( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
            yp[2][i] = y[0][i]*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0][i]-R_0))*cos(M_PI*y[1][i]))/2./sqrt(2)/I_0;            
            yp[0][i] = -M_PI*y[0][i]*cos(M_PI*(y[0][i]-R_0)/2.)*sin(M_PI*y[1][i]/2)/2./I_0;
            yp[1][i] =  M_PI*y[0][i]*sin(M_PI*(y[0][i]-R_0)/2.)*cos(M_PI*y[1][i]/2)/2./I_0 ;
        }
    }
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
            yp[2] = y[0]*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0]-R_0))*cos(M_PI*y[1]))/2./sqrt(2.)/I_0;            
            yp[0] = -M_PI*y[0]*cos(M_PI*(y[0]-R_0)/2.)*sin(M_PI*y[1]/2)/2./I_0;
            yp[1] =  M_PI*y[0]*sin(M_PI*(y[0]-R_0)/2.)*cos(M_PI*y[1]/2)/2./I_0 ;
    }
    private:
    double R_0, I_0;
};

double R_0 = 10;
double I_0 = 20; //I0=20 and R=10 means q=2

double funcNEU(double R, double Z, double phi)
{
    double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    return -psi*cos(phi);
}
double deriNEU(double R, double Z, double phi)
{
    double dldp = R*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/2./sqrt(2.)/I_0;
    double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    return psi*sin(phi)/dldp;
}

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

double deriNEUT(double R, double Z, double phi)
{
    double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    double dldp = R*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/2./sqrt(2.)/I_0;
    double fac1 = sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
    double z1 = cos(M_PI*0.5*(R-R_0))*(32.*I_0*I_0+5.*M_PI*M_PI)+
                M_PI*M_PI*( cos(M_PI*3.*(R-R_0)/2.)+
                            M_PI*R*sin(M_PI*3.*(R-R_0)/2.)) ;
    double z2 = cos(M_PI*0.5*(R-R_0)) + 
                cos(M_PI*3*(R-R_0)/2) + 
                M_PI*R*sin(M_PI*0.5*(R-R_0));
    double nenner = fac1*fac1*fac1*2.*sqrt(2.)*R;
    double divb = -M_PI*(z1*sin(M_PI*Z*0.5)-z2*M_PI*M_PI*sin(M_PI*Z*3./2.))/(nenner);
    double func = -psi*cos(phi);
    double deri = psi*sin(phi)/dldp;
    return divb*func + deri;
}
double deriNEU2(double R, double Z, double phi)
{
    double psi  = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    double fac2 = R*(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
    double fac3 = 4*I_0*cos(phi)*fac2/R;
    double fac4 = 16.*I_0*I_0*cos(M_PI*0.5*(R-R_0))+
                  M_PI*M_PI*sin(M_PI*0.5*(R-R_0))*
                  (-M_PI*R*cos(M_PI*(R-R_0))+cos(M_PI*Z))+
                  sin(M_PI*(R-R_0))*(1+cos(M_PI*Z))
                  +4*M_PI*M_PI*cos(M_PI*0.5*(R-R_0))*cos(M_PI*0.5*(R-R_0))*cos(M_PI*0.5*(R-R_0))*sin(M_PI*Z*0.5)*sin(M_PI*Z*0.5);
    double fac5 = M_PI*sin(phi)*sin(0.5*Z*M_PI)*fac4;
    double dz2 = 2.*I_0*psi*(fac3+fac5)/fac2/fac2;
    return dz2 ;
}
double deriNEUT2(double R, double Z, double phi)
{
    double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    double dldp = R*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/2./sqrt(2.)/I_0;
    double fac1 = sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
    double z1 = cos(M_PI*0.5*(R-R_0))*(32.*I_0*I_0+5.*M_PI*M_PI)+
                M_PI*M_PI*( cos(M_PI*3.*(R-R_0)/2.)+
                            M_PI*R*sin(M_PI*3.*(R-R_0)/2.)) ;
    double z2 = cos(M_PI*0.5*(R-R_0)) + 
                cos(M_PI*3*(R-R_0)/2) + 
                M_PI*R*sin(M_PI*0.5*(R-R_0));
    double nenner = fac1*fac1*fac1*2.*sqrt(2.)*R;
    double divb = -M_PI*(z1*sin(M_PI*Z*0.5)-z2*M_PI*M_PI*sin(M_PI*Z*3./2.))/(nenner);
//     double func = -psi*cos(phi);
    double deri = psi*sin(phi)/dldp;
    double fac2 = R*(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
    double fac3 = 4*I_0*cos(phi)*fac2/R;
    double fac4 = 16.*I_0*I_0*cos(M_PI*0.5*(R-R_0))+
                  M_PI*M_PI*sin(M_PI*0.5*(R-R_0))*
                  (-M_PI*R*cos(M_PI*(R-R_0))+cos(M_PI*Z))+
                  sin(M_PI*(R-R_0))*(1+cos(M_PI*Z))
                  +4*M_PI*M_PI*cos(M_PI*0.5*(R-R_0))*cos(M_PI*0.5*(R-R_0))*cos(M_PI*0.5*(R-R_0))*sin(M_PI*Z*0.5)*sin(M_PI*Z*0.5);
    double fac5 = M_PI*sin(phi)*sin(0.5*Z*M_PI)*fac4;
    double dz2 = 2.*I_0*psi*(fac3+fac5)/fac2/fac2;
    return divb*deri + dz2;
}
int main()
{
    Field field( R_0, I_0);
    InvB invb(R_0, I_0);

    std::cout << "Type n, Nx, Ny, Nz\n";
    //std::cout << "Note, that function is resolved exactly in R,Z for n > 2\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "q = " << I_0/R_0 << std::endl;
    double z0 = 0, z1 = 2.*M_PI;
    //double z0 = M_PI/2., z1 = 3./2.*M_PI;
    dg::Grid3d<double> g3d( R_0 - 1, R_0+1, -1, 1, z0, z1,  n, Nx, Ny, Nz,dg::NEU, dg::NEU, dg::PER,dg::cylindrical);
    dg::Grid2d<double> g2d( R_0 - 1, R_0+1, -1, 1,  n, Nx, Ny);
    
    const dg::DVec w3d = dg::create::weights( g3d);
    const dg::DVec w2d = dg::create::weights( g2d);
    dg::DZ<dg::DMatrix, dg::DVec> dz( field, g3d, g3d.hz(), 1e-4, dg::DefaultLimiter(), dg::DIR);
    
    dg::Grid3d<double> g3dp( R_0 - 1, R_0+1, -1, 1, z0, z1,  n, Nx, Ny, 1);
    
    dg::DZ<dg::DMatrix, dg::DVec> dz2d( field, g3dp, g3d.hz(), 1e-4, dg::DefaultLimiter(), dg::DIR);
    dg::DVec boundary=dg::evaluate( dg::zero, g3d);
    


    dg::DVec function = dg::evaluate( funcNEU, g3d) ,
                        temp( function),
                        derivative(function),
                        inverseB( dg::evaluate(invb, g3d)),
                        derivativeT(function),
                        derivativeTdz(function),
                        functionTinv(function),
                        dzTdz(function);

    const dg::DVec solution = dg::evaluate( deriNEU, g3d);
    const dg::DVec solutionT = dg::evaluate( deriNEUT, g3d);
    const dg::DVec solutiondzTdz = dg::evaluate( deriNEUT2, g3d);

    dz( function, derivative); //dz(f)
    dz.centeredT( function, derivativeT); //dz(f)
    dg::blas1::pointwiseDot( inverseB, function, temp);
    dz( temp, derivativeTdz);
    dg::blas1::pointwiseDivide( derivativeTdz, inverseB, derivativeTdz);
    dz.centeredT( derivative, dzTdz); //dz(f)

    
    std::cout << "--------------------testing dz" << std::endl;
    double norm = dg::blas2::dot( w3d, solution);
    std::cout << "Norm Solution    "<<sqrt( norm)<<"\n";
    double err =dg::blas2::dot( w3d, derivative);
    std::cout << "Norm Derivative  "<<sqrt( err)<<"\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    err =dg::blas2::dot( w3d, derivative);
    std::cout << "Relative Difference in DZ is "<< sqrt( err/norm )<<"\n"; 

     std::cout << "--------------------testing dzT" << std::endl;
    double normT = dg::blas2::dot( w3d, solutionT);
    std::cout << "Norm SolutionT    "<<sqrt( normT)<<"\n";
    double errT =dg::blas2::dot( w3d, derivativeT);
    std::cout << "Norm DerivativeT  "<<sqrt( errT)<<"\n";
    dg::blas1::axpby( 1., solutionT, -1., derivativeT);
    err =dg::blas2::dot( w3d, derivativeT);
    std::cout << "Relative Difference in DZT is "<< sqrt( errT/normT )<<"\n"; 
    
    std::cout << "--------------------testing dzT with dz" << std::endl;
    std::cout << "Norm SolutionT    "<<sqrt( normT)<<"\n";
    double errTdz =dg::blas2::dot( w3d, derivativeTdz);
    std::cout << "Norm DerivativeTdz  "<<sqrt( errTdz)<<"\n";
    dg::blas1::axpby( 1., solutionT, -1., derivativeTdz);
    errTdz =dg::blas2::dot( w3d, derivativeTdz);
    std::cout << "Relative Difference in DZT is "<< sqrt( errTdz/normT )<<"\n"; 
    
    std::cout << "--------------------testing dzTdz " << std::endl;
    double normdzTdz = dg::blas2::dot( w3d, solutiondzTdz);
    std::cout << "Norm SolutionT    "<<sqrt( normdzTdz)<<"\n";
    double errdzTdz =dg::blas2::dot( w3d,dzTdz);
    std::cout << "Norm DerivativeTdz  "<<sqrt( errTdz)<<"\n";
    dg::blas1::axpby( 1., solutiondzTdz, -1., dzTdz);
    errdzTdz =dg::blas2::dot( w3d, dzTdz);
    std::cout << "Relative Difference in DZT is "<< sqrt( errdzTdz/normdzTdz )<<"\n"; 
    
    std::cout << "--------------------testing dzT with inversion " << std::endl;

    double eps =1e-6;    
    dg::Invert< dg::DVec> invert(dg::evaluate(dg::zero,g3d), w3d.size(), eps );   
    std::cout << " # of iterations "<< invert( dz, functionTinv, solutionT ) << std::endl; //is dzTdz
    double normf = dg::blas2::dot( w3d, function);
    std::cout << "Norm analytic Solution  "<<sqrt( normf)<<"\n";
    double errinvT =dg::blas2::dot( w3d, functionTinv);
    std::cout << "Norm numerical Solution "<<sqrt( errinvT)<<"\n";
    dg::blas1::axpby( 1., function, -1.,functionTinv);
    errinvT =dg::blas2::dot( w3d, functionTinv);
    std::cout << "Relative Difference is  "<< sqrt( errinvT/normf )<<"\n";
    return 0;

}
