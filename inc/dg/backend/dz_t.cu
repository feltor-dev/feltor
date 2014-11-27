#include <iostream>

#include <cusp/print.h>

#include "evaluation.cuh"
#include "dz.cuh"
#include "functions.h"
#include "../blas2.h"
#include "../functors.h"
#include "interpolation.cuh"

struct InvB
{
    InvB( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double x, double y, double z)
    {
        double gradpsi = ((x-R_0)*(x-R_0) + y*y)/I_0/I_0;
        return  x/sqrt( 1 + gradpsi)/R_0/I_0;
    }
    private:
    double R_0, I_0;
};
//magnetic field with circular cross section and constant I
struct Field
{
    Field( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
            double gradpsi = ((y[0][i]-R_0)*(y[0][i]-R_0) + y[1][i]*y[1][i])/I_0/I_0;
            yp[2][i] = y[0][i]*sqrt(1 + gradpsi);
            yp[0][i] = y[0][i]*y[1][i]/I_0;
            yp[1][i] = y[0][i]/I_0*(R_0-y[0][i]) ;
        }
    }
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
        double gradpsi = ((y[0]-R_0)*(y[0]-R_0) + y[1]*y[1])/I_0/I_0;
        yp[2] = y[0]*sqrt(1 + gradpsi);
        //yp[2] = y[0]*y[0]/I_0/R_0; //now we integrate B\cdot\nabla
        yp[0] = y[0]*y[1]/I_0;
        yp[1] = y[0]/I_0*(R_0-y[0]) ;
    }
    private:
    double R_0, I_0;
};
//psi = cos(0.5*pi*(R-R_0))*cos(0.5*pi*Z)
struct FieldDIR
{
    FieldDIR( double R_0, double I_0):R_0(R_0), I_0(I_0){}
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

double func2d(double R, double Z)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = sqrt(I_0*I_0+r2)/R;
    double bphi = I_0/R/R/B;
    return 1/bphi/R;
}
double funcNEU(double R, double Z, double phi)
{
    double dpsi2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = R_0*sqrt(I_0*I_0+dpsi2)/R;
    double bPh = R_0*I_0/R/R/B;
    return -cos(phi)/bPh/R; //NEU
}

double funcDIR(double R, double Z, double phi)
{
    return sin(M_PI*(Z))*sin(M_PI*(R-R_0))*sin(phi); //DIR 
//         double dpsi2 = (R-R_0)*(R-R_0)+Z*Z;
//     double B = R_0*sqrt(I_0*I_0+dpsi2)/R;
//     double bPh = R_0*I_0/R/R/B;
// 
//     return sin(phi)/bPh/R; //(2)
}
double modulate( double R, double Z, double phi) {return -cos(phi);}
double deri2d(double R, double Z)
{
    return 0;
}

double deriDIR(double R, double Z, double phi)
{
    double dpsi2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = R_0*sqrt(I_0*I_0+dpsi2)/R;
    double bRh = R_0*Z/B/R;
    double bZh = -R_0*(R-R_0)/B/R;
    double bPh = R_0*I_0/R/R/B;
    return M_PI*bRh*sin(M_PI*(Z))*cos(M_PI*(R-R_0))*sin(phi)
           +M_PI*bZh*cos(M_PI*(Z))*sin(M_PI*(R-R_0))*sin(phi)+
           bPh*sin(M_PI*(Z))*sin(M_PI*(R-R_0))*cos(phi);   //(1) 
//     return cos(phi)/R; //(2)
}
double deriDIR2(double R, double Z, double phi)
{
    double dpsi2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = R_0*sqrt(I_0*I_0+dpsi2)/R;
    //(1) too complicated term
    return Z*R_0*sin(phi)/B/R/R/R - R_0*cos(phi)/B/R/R/R ; //(2)
}
double deriNEU(double R, double Z, double phi)
{
    return sin(phi)/R;
}
double deriTNEU(double R, double Z, double phi)
{
//     double dpsi2 = (R-R_0)*(R-R_0)+Z*Z;
//     double B = R_0*sqrt(I_0*I_0+dpsi2)/R;
//     double bPh = R_0*I_0/R/R/B;
//     double divb = Z/sqrt(I_0*I_0+dpsi2)/R;
//     return (sin(phi)/R-divb*(cos(phi)/bPh/R));
    return (I_0*sin(phi)-Z*cos(phi))/I_0/R;
}
double deriNEU2(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = sqrt(I_0*I_0+r2)/R;
    double bphi = I_0/R/R/B;
    double bR = Z/R/B;
    return bphi/R*cos(phi) - bR*sin(phi)/R/R ;
}
double divb(double R, double Z, double phi)
{
    double dpsi2 = (R-R_0)*(R-R_0)+Z*Z;
    return Z/sqrt(I_0*I_0+dpsi2)/R;
}
double cut(double R, double Z, double phi)
{
    double psip = 0.5*((R-R_0)*(R-R_0)+Z*Z);
    if (psip >0.5*((10.9-R_0)*(10.9-R_0)+0.9*0.9)) return 0.;
    return 1.;
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
    dg::DZ<dg::DMatrix, dg::DVec> dz( field, g3d, g3d.hz(), 1e-4, dg::DefaultLimiter(), dg::NEU);
    
    dg::Grid3d<double> g3dp( R_0 - 1, R_0+1, -1, 1, z0, z1,  n, Nx, Ny, 1);
    
    dg::DZ<dg::DMatrix, dg::DVec> dz2d( field, g3dp, g3d.hz(), 1e-4, dg::DefaultLimiter(), dg::NEU);
    dg::DVec boundary=dg::evaluate( dg::zero, g3d);
    
    dz.set_boundaries( dg::PER, 0, 0);
    //dz.set_boundaries( dg::DIR, 0., -0.);
    //dz.set_boundaries( dg::DIR, boundary, 1, 1);

    dg::DVec function = dg::evaluate( funcNEU, g3d), 
             temp( function),
             derivative(function), 
             derivativeT(function), 
             derivativeTone(function), 
             inverseB( dg::evaluate(invb, g3d)),
             dzTdz(function), 
             dzz(dg::evaluate(deriNEU2, g3d));

        
    dg::DVec ones = dg::evaluate( dg::one, g3d);
    dg::DVec divbongrid = dg::evaluate( divb, g3d);
    dg::DVec cutongrid = dg::evaluate( cut, g3d);
    std::cout << "--------------------testing field aligning" << std::endl;
    dg::DVec function2d = dg::evaluate( func2d, g2d), derivative2d( function2d) ;
    dg::DVec follow = dz.evaluate( func2d, 0), sinz(dg::evaluate( modulate, g3d));
    dg::blas1::pointwiseDot( follow, sinz, follow);
    dg::blas1::axpby( 1., function, -1., follow);
    double diff = dg::blas2::dot( w3d, follow);
    std::cout << "Difference between function and followed evaluation: "<<diff<<"\n";
    const dg::DVec solution = dg::evaluate( deriNEU, g3d);
    dg::DVec solutionT = dg::evaluate( deriTNEU, g3d);
    const dg::DVec solution2 = dg::evaluate( deriNEU2, g3d);
    const dg::DVec solution2d = dg::evaluate( deri2d, g2d);
    dz( function, derivative); //dz(f)
    //dz.centeredT( function, derivative); //dz(f)
    //dz.centeredT( ones, derivativeTone); //dzT(1)
    //dz.centeredT( derivative, dzTdz);       //dzT(dz(f))
    dg::blas1::pointwiseDot( inverseB, function, temp);
    dz( temp, derivativeT); //dzT(f)
    dg::blas1::pointwiseDivide( derivativeT, inverseB, derivativeT);
    dz( inverseB, derivativeTone); //dzT(1)
    dg::blas1::pointwiseDivide( derivativeTone, inverseB, derivativeTone);
    dg::blas1::pointwiseDot( inverseB, derivative, temp);
    dz( temp, dzTdz);       //dzT(dz(f))
    dg::blas1::pointwiseDivide( dzTdz, inverseB, dzTdz);
    dz2d( function2d, derivative2d);
    dz.dzz( function, dzz);
    double fdzf = dg::blas2::dot( function, w3d, derivative);
    double dzTff = dg::blas2::dot( function, w3d, derivativeT);
    double dzfdzf = dg::blas2::dot( derivative, w3d, derivative);
    double dzTdzf =  dg::blas2::dot( w3d, dzTdz);
    double dzzf =  dg::blas2::dot( w3d, dzz);
    double fdzTdzf = dg::blas2::dot( function, w3d, dzTdz);
    
    dg::blas1::pointwiseDot(cutongrid,solutionT,solutionT);
    dg::blas1::pointwiseDot(cutongrid,derivativeT,derivativeT);
    //dg::blas1::pointwiseDot(cutongrid,derivativeTone,derivativeTone);
    //dg::blas1::pointwiseDot(cutongrid,divbongrid,divbongrid);
    //-------------------------------------------- dz
    std::cout << "--------------------testing dz" << std::endl;
    double norm = dg::blas2::dot( w3d, solution);
    std::cout << "Norm Solution    "<<sqrt( norm)<<"\n";
    double err =dg::blas2::dot( w3d, derivative);
    std::cout << "Norm Derivative  "<<sqrt( err)<<"\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    err =dg::blas2::dot( w3d, derivative);
    std::cout << "Relative Difference in DZ is "<< sqrt( err/norm )<<"\n";    

    //-------------------------------------------- dzT
    std::cout << "--------------------testing dzT" << std::endl;
    double normT = dg::blas2::dot( w3d, solutionT);
    std::cout << "Norm SolutionT    "<<sqrt( normT)<<"\n";
    double errT =dg::blas2::dot( w3d, derivativeT);
    std::cout << "Norm DerivativeT  "<<sqrt( errT)<<"\n";
    dg::blas1::axpby( 1., solutionT, -1., derivativeT);
    errT =dg::blas2::dot( w3d, derivativeT);
    std::cout << "Relative Difference in DZT is "<< sqrt( errT/normT )<<"\n";    
    double normdivb =  dg::blas2::dot( w3d, divbongrid);  
    std::cout << "Norm divb    "<<sqrt( normdivb)<<"\n";
    double errTdivb =dg::blas2::dot( w3d, derivativeTone);
    std::cout << "Norm DerivativeTone  "<<sqrt( errTdivb)<<"\n";
    dg::blas1::axpby( 1., divbongrid, -1., derivativeTone);
    errTdivb =dg::blas2::dot( w3d, derivativeTone);
    std::cout << "Relative Difference in DZT is "<< sqrt( errTdivb/normdivb )<<"\n";    
    
     //-------------------------------------------- dzz
    std::cout << "--------------------testing dzz" << std::endl;   
    dg::blas1::axpby( 1., solution2, -1., dzz);
    norm = dg::blas2::dot( w3d, solution2);  
    std::cout << "Relative Difference in DZZ is "<< sqrt( dg::blas2::dot( w3d, dzz)/norm )<<"\n";    
    dg::blas1::axpby( 1., solution2d, -1., derivative2d);
    std::cout << "Difference in DZ2d is "<< sqrt( dg::blas2::dot( w2d, derivative2d) )<<"\n";    
    dz.einsPlus( function, derivative);
    //dz.einsMinus( derivative, dzz);
    dz.einsPlusT( derivative, dzz);
    dg::blas1::axpby( 1., function, -1., dzz );
    std::cout << "Difference in EinsPlusMinus is "<< sqrt( dg::blas2::dot( w3d, dzz) )<<" !=0!\n";
    std::cout << "--------------------testing adjoint property of dz" << std::endl;   
    std::cout << "f DZ(f)     = "<< fdzf<< " DZT(f) f = "<<dzTff<<" diff = "<<fdzf-dzTff<<" !=0\n";
    std::cout << "fDZT(DZ(f)) = "<< fdzTdzf<< " -DZ(f)DZ(f) = "<<-dzfdzf<<" diff = "<<fdzTdzf+dzfdzf<<" !=0\n";        
//     std::cout << "dzz(f) = "<< dzzf<< " dzT(dz(f)) = "<< dzTdzf<<" diff = "<< dzTdzf-dzzf<<"\n";        
    //---------------------------------------------------solve Matrix equation
    double eps =1e-6;
    dg::Invert< dg::DVec> invert( initial_guess, w3d.size(), eps );

    std::cout << " # of iterations "<< invert( dz , solution  , rho ) << std::endl;

    
    return 0;
}
