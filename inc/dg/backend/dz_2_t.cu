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
//psi = cos(0.5*pi*(R-R_0))*cos(0.5*pi*Z)
struct Field
{
    Field( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
         
            
            yp[2][i] = y[0][i]*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0][i]-R_0))*cos(M_PI*y[1][i]))/2./sqrt(2)/I_0;   //=dldphi = 1/bphi          
            yp[0][i] = -M_PI*y[0][i]*cos(M_PI*(y[0][i]-R_0)/2.)*sin(M_PI*y[1][i]/2)/2./I_0; //=dR/dphi = b^R/b^phi
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

double BR(double R, double Z, double phi)
{
    double dldphi = R*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/2./sqrt(2)/I_0;
    double brbphi= -M_PI*R*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2)/2./I_0;
    double br =  brbphi/dldphi;
    double invB = 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
    return br/invB;
}
double InvR(double R, double Z, double phi)
{
    return 1/R;
}
double funcNEU(double R, double Z, double phi)
{
    double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    return -psi*cos(phi);
}
double funcNEU2(double R, double Z, double phi)
{
    double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    return -psi*cos(phi)+0.5*(R-R_0)*0.5*(R-R_0) +Z*0.5*0.5*(R-R_0) ;
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
    LnB lnB(R_0, I_0);

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
    const dg::DVec v3d = dg::create::inv_weights( g3d);

    dg::DZ<dg::DMatrix, dg::DVec> dz( field, g3d, g3d.hz(), 1e-4, dg::DefaultLimiter(), dg::NEU);
    
    dg::Grid3d<double> g3dp( R_0 - 1, R_0+1, -1, 1, z0, z1,  n, Nx, Ny, 1);
    
    dg::DZ<dg::DMatrix, dg::DVec> dz2d( field, g3dp, g3d.hz(), 1e-4, dg::DefaultLimiter(), dg::NEU);
    dg::DVec boundary=dg::evaluate( dg::zero, g3d);
    
    dg::DVec function = dg::evaluate( funcNEU, g3d) ,
                        temp( function),
                        temp2( function),
                        temp3( function),
                        derivative(function),
                        derivativeones(function),
                        derivative2(function),
                        inverseB( dg::evaluate(invb, g3d)),
                        derivativeT(function),
                        logB( dg::evaluate(lnB, g3d)),
                        derivativeT2(function),
                        derivativeTones(function),
                        derivativeTdz(function),
                        functionTinv(function),
                        dzTdz(function),
                        divbsol(dg::evaluate(divb, g3d)),
                        divbT(function),
                        divBT(function),
                        dzTdz2(function);


    dg::DVec ones = dg::evaluate( dg::one, g3d);
    const dg::DVec function2 = dg::evaluate( funcNEU2, g3d);
    const dg::DVec solution = dg::evaluate( deriNEU, g3d);
    const dg::DVec solutionT = dg::evaluate( deriNEUT, g3d);
    const dg::DVec solutiondzTdz = dg::evaluate( deriNEUT2, g3d);
    dg::DVec magBR = dg::evaluate( BR, g3d);
    const dg::DVec invR = dg::evaluate( InvR, g3d);
    
    dz.set_boundaries( dg::PER, 0, 0);

    dz( function, derivative); //dz(f)
    dz( ones, derivativeones); //dz(f)
    dz( function2, derivative2); //dz(f)

    dz.centeredT(function, derivativeT); //dz(f)
    //B corrections
    dg::blas1::pointwiseDivide(ones,  inverseB, temp2); //B
    dz.centeredT(temp2, divBT); // dzT B


//     dg::blas1::pointwiseDot(divBT,function,temp2); //f dzTB
//     dg::blas1::pointwiseDot(temp2,inverseB,temp3); //f/B dzTB
//     dg::blas1::axpby(1.0,derivativeT,-1.0,divBT,derivativeT); //... - f/B dzTB
    
    dz.centeredT( function2, derivativeT2); //dz(f)
    dz.centeredT( ones, derivativeTones); //dz(f)
    dg::blas1::pointwiseDot( inverseB, function, temp);
    dz( temp, derivativeTdz);
    dg::blas1::pointwiseDivide( derivativeTdz, inverseB, derivativeTdz);
    dz.centeredT( derivative, dzTdz); //dzT(dz(f))
    dz.centeredT( derivative2, dzTdz2); //dzT(dz(f))

    double normdzdz =dg::blas2::dot(derivative2, w3d,derivative2);
    double normdz1dz =dg::blas2::dot(derivativeones, w3d,derivative2);
    double normdivBT =dg::blas2::dot(divBT, w3d,divBT);
    double normdivbT =dg::blas2::dot(divbT, w3d,divbT);
    double normdivb =dg::blas2::dot(divbsol, w3d,divbsol); 
    double normdzTf = dg::blas2::dot(derivativeT2, w3d, function2);
    double normdzT1 = dg::blas2::dot(derivativeTones, w3d, function2);
    double normfdz = dg::blas2::dot(function2, w3d, derivative2);
    double norm1dz = dg::blas2::dot(ones, w3d, derivative2);
    double normfdzTdz = dg::blas2::dot(function2, w3d, dzTdz2);
    double norm1dzTdz = dg::blas2::dot(ones, w3d, dzTdz2);       
 
    std::cout << "--------------------testing dz" << std::endl;
    double norm = dg::blas2::dot( w3d, solution);
    std::cout << "|| Solution ||   "<<sqrt( norm)<<"\n";
    double err =dg::blas2::dot( w3d, derivative);
    std::cout << "|| Derivative || "<<sqrt( err)<<"\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    err =dg::blas2::dot( w3d, derivative);
    std::cout << "Relative Difference in DZ is "<< sqrt( err/norm )<<"\n"; 

    std::cout << "--------------------testing dzT" << std::endl;
    std::cout << "|| divbsol ||  "<<sqrt( normdivb)<<"\n";
    std::cout << "|| divbT  ||   "<<sqrt( normdivbT)<<"\n";
    dg::blas1::axpby( 1., divbsol, -1., divbT);
    normdivbT =dg::blas2::dot(divbT, w3d,divbT);
    std::cout << "Relative Difference in DZT is   "<<sqrt( normdivbT)<<"\n";
    std::cout << "-------------------- " << std::endl;
    std::cout << "|| divB || "<<sqrt( normdivBT)<<"\n";

    
    std::cout << "-------------------- " << std::endl;
    double normT = dg::blas2::dot( w3d, solutionT);
    std::cout << "|| SolutionT  ||  "<<sqrt( normT)<<"\n";
    double errT =dg::blas2::dot( w3d, derivativeT);
    std::cout << "|| DerivativeT || "<<sqrt( errT)<<"\n";
    dg::blas1::axpby( 1., solutionT, -1., derivativeT);
    errT =dg::blas2::dot( w3d, derivativeT);
    std::cout << "Relative Difference in DZT is "<< sqrt( errT/normT )<<"\n"; 
    
    std::cout << "--------------------testing dzT with dz" << std::endl;
    std::cout << "|| SolutionT ||     "<<sqrt( normT)<<"\n";
    double errTdz =dg::blas2::dot( w3d, derivativeTdz);
    std::cout << "|| DerivativeTdz || "<<sqrt( errTdz)<<"\n";
    dg::blas1::axpby( 1., solutionT, -1., derivativeTdz);
    errTdz =dg::blas2::dot( w3d, derivativeTdz);
    std::cout << "Relative Difference in DZT is "<< sqrt( errTdz/normT )<<"\n"; 
    
    std::cout << "--------------------testing dzTdz " << std::endl;
    double normdzTdz = dg::blas2::dot( w3d, solutiondzTdz);
    std::cout << "|| SolutionT ||      "<<sqrt( normdzTdz)<<"\n";
    double errdzTdz =dg::blas2::dot( w3d,dzTdz);
    std::cout << "|| DerivativeTdz ||  "<<sqrt( errTdz)<<"\n";
    dg::blas1::axpby( 1., solutiondzTdz, -1., dzTdz);
    errdzTdz =dg::blas2::dot( w3d, dzTdz);
    std::cout << "Relative Difference in DZT is "<< sqrt( errdzTdz/normdzTdz )<<"\n";     
    
    std::cout << "--------------------testing adjointness " << std::endl;
    std::cout << "<f,dz(f)>   = "<< normfdz<<"\n";
    std::cout << "-<dzT(f),f> = "<< -normdzTf<<"\n";
    std::cout << "Diff        = "<< normfdz+normdzTf<<"\n";     
    std::cout << "-------------------- " << std::endl;

    std::cout << "<1,dz(f)>   = "<< norm1dz<<"\n";
    std::cout << "-<dzT(1),f> = "<< -normdzT1<<"\n";
    std::cout << "Diff        = "<< norm1dz+normdzT1<<"\n";   
    std::cout << "-------------------- " << std::endl;
  
    std::cout << "<f,dzT(dz(f))> = "<< normfdzTdz<<"\n";
    std::cout << "-<dz(f),dz(f)> = "<< -normdzdz<<"\n";
    std::cout << "Diff           = "<< normfdzTdz+normdzdz<<"\n";     
    std::cout << "-------------------- " << std::endl;
   
    std::cout << "<1,dzT(dz(f))> = "<< norm1dzTdz<<"\n";
    std::cout << "-<dz(1),dz(f)> = "<< -normdz1dz<<"\n";
    std::cout << "Diff           = "<< norm1dzTdz+normdz1dz<<"\n";    
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
