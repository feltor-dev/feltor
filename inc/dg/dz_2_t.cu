#include <iostream>

#include <cusp/print.h>
#include <cusp/csr_matrix.h>
#include "dg/backend/xspacelib.cuh"
#include "file/read_input.h"
// #include "file/nc_utilities.h"

#include "backend/evaluation.cuh"
#include "backend/timer.cuh"
#include "blas.h"
#include "dz.h"
#include "backend/functions.h"
#include "functors.h"
#include "elliptic.h"
#include "cg.h"
// #include "draw/host_window.h"
#include "../../src/heat/geometry_g.h"
#include "../../src/heat/parameters.h"


int main( )
{

    /////////////////initialize params////////////////////////////////
     std::vector<double> v,v2,v3;

        try{
            v = file::read_input("../../src/heat/input.txt");
            v3 = file::read_input( "../../src/heat/geometry_params_g.txt"); 
        }catch( toefl::Message& m){
            m.display();
            return -1;
        }

    const eule::Parameters p( v);
//     p.display( std::cout);
    const solovev::GeomParameters gp(v3);
//     gp.display( std::cout);

    //////////////////////////////////////////////////////////////////////////
    
    double Rmin=gp.R_0-1.0*gp.a;
    double Zmin=-1.0*gp.a*gp.elongation;
    double Rmax=gp.R_0+1.0*gp.a; 
    double Zmax=1.0*gp.a*gp.elongation;
    /////////////////////////////////////////////initialze fields /////////////////////
    
    solovev::Field field(gp);
    solovev::InvB invb(gp);
    solovev::GradLnB gradlnB(gp);
    solovev::LnB lnB(gp);
    solovev::FieldR bR_(gp);
    solovev::FieldZ bZ_(gp);
    solovev::FieldP bPhi_(gp);
    solovev::FuncNeu funcNEU(gp.R_0,gp.I_0);
    solovev::FuncNeu2 funcNEU2(gp.R_0,gp.I_0);
    solovev::DeriNeu deriNEU(gp.R_0,gp.I_0);
    solovev::DeriNeu2 deriNEU2(gp.R_0,gp.I_0);
    solovev::DeriNeuT2 deriNEUT2(gp.R_0,gp.I_0);
    solovev::DeriNeuT deriNEUT(gp.R_0,gp.I_0);
    solovev::Divb divb(gp.R_0,gp.I_0);
    solovev::B Bfield(gp);
    
    std::cout << "Type n, Nx, Ny, Nz\n";
    //std::cout << "Note, that function is resolved exactly in R,Z for n > 2\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    unsigned Nxn = Nx;
    unsigned Nyn = Ny;
    unsigned Nzn = Nz;

    double rk4eps;
    std::cout << "Type RK4 eps (1e-8)\n";
    std::cin >> rk4eps;
    double z0 = 0, z1 = 2.*M_PI;
    for (unsigned i=0;i<6;i++) { 

        Nzn = unsigned(Nz*pow(2,i));
        Nxn = (unsigned)ceil(Nx*pow(2,(double)(i*2./n)));
        Nyn = (unsigned)ceil(Ny*pow(2,(double)(i*2./n)));



        dg::Grid3d<double> g3d( Rmin,Rmax, Zmin,Zmax, z0, z1,  n,Nxn ,Nyn, Nzn,dg::DIR, dg::DIR, dg::PER,dg::cylindrical);
        dg::Grid2d<double> g2d( Rmin,Rmax, Zmin,Zmax,  n, Nxn ,Nyn);

        std::cout << "NR = " << Nxn << std::endl;
        std::cout << "NZ = " << Nyn<< std::endl;
        std::cout << "Nphi = " << Nzn << std::endl;
//            Nxn = (unsigned)ceil(Nxn*pow(2,(double)(2./n)));
//     Nyn = (unsigned)ceil( Nyn*pow(2,(double)(2./n)));

//        dg::Grid3d<double> g3d( Rmin,Rmax, Zmin,Zmax, z0, z1,  n, Nx, Ny, Nz*pow(2,i),dg::DIR, dg::DIR, dg::PER,dg::cylindrical);
//     dg::Grid2d<double> g2d( Rmin,Rmax, Zmin,Zmax,  n, Nx, Ny); 
    const dg::DVec w3d = dg::create::weights( g3d);
    const dg::DVec w2d = dg::create::weights( g2d);
    const dg::DVec v3d = dg::create::inv_weights( g3d);

    std::cout << "computing dzDIR" << std::endl;
    dg::FieldAligned<dg::IDMatrix, dg::DVec>    
        dzFA( field, g3d, rk4eps, dg::DefaultLimiter(), dg::DIR);
    std::cout << "computing dzNEU" << std::endl;
    dg::FieldAligned<dg::IDMatrix, dg::DVec> 
        dzNUFA( field, g3d, rk4eps, dg::DefaultLimiter(), dg::NEU);

    dg::DZ< dg::FieldAligned<dg::IDMatrix, dg::DVec>, dg::DMatrix, dg::DVec> 
        dz ( dzFA, field, g3d, dg::not_normed, dg::centered), 
        dzNU ( dzNUFA, field, g3d, dg::not_normed, dg::centered);

//     dg::DZ<dg::DMatrix, dg::DVec> dzNEU( field, g3d, g3d.hz(), rk4eps, dg::DefaultLimiter(), dg::NEU);
    
//     dg::Grid3d<double> g3dp( Rmin,Rmax, Zmin,Zmax, z0, z1,  n, Nx, Ny, 1);
    
//     dg::DZ<dg::DMatrix, dg::DVec> dz2d( field, g3dp, g3d.hz(), rk4eps, dg::DefaultLimiter(), dg::NEU);
    dg::DVec boundary=dg::evaluate( dg::zero, g3d);
    
    dg::DVec function = dg::evaluate( funcNEU, g3d) ,
                        temp( function),
                        temp2( function),
                        temp3( function),
                        derivative(function),
                        derivativeRZPhi(function),
                        diffRZPhi(function),
                        derivativef(function),
                        derivativeb(function),
                        derivativeones(function),
                        derivative2(function),
                        inverseB( dg::evaluate(invb, g3d)),
                        derivativeT(function),
                        logB( dg::evaluate(lnB, g3d)),
                        derivativeT2(function),
                        derivativeTones(function),
                        derivativeTdz(function),
                        functionTinv(dg::evaluate( dg::zero, g3d)),
                        functionTinv2(dg::evaluate( dg::zero, g3d)),
                        dzTdz(function),
                        dzTdzb(function),
                        dzTdzf(function),
                        dzTdzbd(function),
                        dzTdzfd(function),
                        dzTdzfb(function),
                        dzTdzfbd(function),
                        dzz(function),
                        divbsol(dg::evaluate(divb, g3d)),
                        divbT(function),
                        divBT(function),
                        lambda(function),
                        omega(function),
                        dzTdz2(function);


    dg::DVec ones = dg::evaluate( dg::one, g3d);
    const dg::DVec function2 = dg::evaluate( funcNEU2, g3d);
    const dg::DVec solution = dg::evaluate( deriNEU, g3d);
    const dg::DVec solutionT = dg::evaluate( deriNEUT, g3d);
    const dg::DVec solutiondzz = dg::evaluate( deriNEU2, g3d);
    const dg::DVec solutiondzTdz = dg::evaluate( deriNEUT2, g3d);

    const dg::DVec bhatR = dg::evaluate( bR_, g3d);
    const dg::DVec bhatZ = dg::evaluate( bZ_, g3d);
    const dg::DVec bhatPhi = dg::evaluate(bPhi_, g3d);
//     const dg::DVec Bfield_ = dg::evaluate(Bfield, g3d);
    const dg::DVec gradlnB_ = dg::evaluate(gradlnB, g3d);
//     dg::DMatrix dR(dg::create::dx( g3d, g3d.bcx(),dg::normed,dg::centered));
//     dg::DMatrix dZ(dg::create::dy( g3d, g3d.bcy(),dg::normed,dg::centered));
//     dg::DMatrix dphi(dg::create::dz( g3d, g3d.bcz(), dg::normed,dg::centered));
    
//     dz.set_boundaries( dg::PER, 0, 0);
    //direct gradpar method
//     dg::blas2::gemv( dR, function, temp); //d_R src
//     dg::blas2::gemv( dZ, function, temp2);  //d_Z src
//     dg::blas2::gemv( dphi, function, temp3);  //d_phi src
//     dg::blas1::pointwiseDot( bhatR, temp, temp); // b^R d_R src
//     dg::blas1::pointwiseDot( bhatZ, temp2, temp2); // b^Z d_Z src
//     dg::blas1::pointwiseDot( bhatPhi, temp3, temp3); // b^phi d_phi src
//     dg::blas1::axpby( 1., temp, 1., temp2 ); // b^R d_R src +  b^Z d_Z src
//     dg::blas1::axpby( 1., temp3, 1., temp2,derivativeRZPhi ); // b^R d_R src +  b^Z d_Z src + b^phi d_phi src
// 
//     dg::GeneralEllipticSym<dg::DMatrix, dg::DVec, dg::DVec> ellipticsym( g3d, dg::normed, dg::forward);
//     ellipticsym.set_x(bhatR);
//     ellipticsym.set_y(bhatZ );
//     ellipticsym.set_z(bhatPhi);
//     
//     
  
    dzNU( function, derivative); //dz(f)

//     dzNU.forward( function, derivativef); //dz(f)
//     dzNU.backward( function, derivativeb); //dz(f)

//     dz( ones, derivativeones); //dz(f)
//     dz( function2, derivative2); //dz(f)
//     //compute dzz
//     dz( inverseB, lambda); //gradpar 1/B
//     dg::blas1::pointwiseDivide(lambda,  inverseB, lambda); //-dz lnB
//     dz(function,omega); //dz T
//     dg::blas1::pointwiseDot(omega, lambda, omega);            //- dz lnB dz T
//     dg::blas1::pointwiseDot(omega, gradlnB_, omega);            //- dz lnB dz T
//     dg::blas1::axpby(1.0, omega, 0., dzz,dzz);    
    //     dg::blas1::axpby(-1.0, omega, 0., dzz,dzz);    

    
//     dzNU.forward(derivativeb,temp);
//     dzNU.backward(derivativef,omega);
//     dg::blas1::axpby( -1.0, omega, -0.0, temp,dzz);
//     dz( derivative, dzz); //dz(dz(f))

//     dz.dzz(function,dzz);       
//     dzNU( function, derivative); //dz(f)
//     dg::blas1::pointwiseDot(derivative, gradlnB_, omega);            //- dz lnB dz T
// 
//dz^2 T 
//     dg::blas1::axpby( -1.0, omega, 1.,dzz, dzTdzfbd);

//     
//     
//     dz.centeredT(function, derivativeT); //dz(f)
// 
//     //divB
//     dg::blas1::pointwiseDivide(ones,  inverseB, temp2); //B
//     dz.centeredT(temp2, divBT); // dzT B
// 
//     
//     dz.centeredT( function2, derivativeT2); //dz(f)
//     dz.centeredT( ones, derivativeTones); //dz(f)
    //B dz f/B
//     dg::blas1::pointwiseDot( inverseB, function, temp);
//     dz( temp, derivativeTdz);
//     dg::blas1::pointwiseDot( derivativeTdz, Bfield_, derivativeTdz);
    //oder dz f - f dzlnB
//     dz( function, derivativeTdz);
//     dg::blas1::pointwiseDot(function,gradlnB_,temp);
//     dg::blas1::axpby(- 1.0, temp, 1., derivativeTdz,derivativeTdz);


    
    //     dg::blas1::pointwiseDivide( derivativeTdz, inverseB, derivativeTdz);
//     
//     dz.centeredT( derivative, dzTdz); //dzT(dz(f))
//     
//     //overwrite with sym from adjoint dg
//     ellipticsym.symv(function,dzTdz);
//     dg::blas1::scal(dzTdz,-1.0);
// //     dz.centeredT(ones,divbT);
    dz.forwardT( derivativef, dzTdzf);  //dzT(dz(f))
//     dz.backwardT( derivativeb, dzTdzb); //dzT(dz(f))

//     //centered
//     dz.centeredTD(derivative,dzTdzfbd);
//     dz.forwardTD( derivativef, dzTdzfd); //dzT(dz(f))
//     dz.backwardTD( derivativeb, dzTdzbd); //dzT(dz(f))

//     //arithmetic average
//     dg::blas1::axpby(0.5,dzTdzb,0.5,dzTdzf,dzTdzfb);
//     dg::blas1::axpby(0.5,dzTdzbd,0.5,dzTdzfd,dzTdzfbd); 
    dz.symv(function,dzTdzfb);
    dg::blas1::pointwiseDot(v3d,dzTdzfb,dzTdzfb);
//     dz.centeredT( derivative2, dzTdz2); //dzT(dz(f))
//     dg::blas1::pointwiseDivide(ones,  inverseB, temp2); //B
//     dz.centeredT( ones, divbT);
//     
//     double normdzdz =dg::blas2::dot(derivative2, w3d,derivative2);
//     double normdz1dz =dg::blas2::dot(derivativeones, w3d,derivative2);
//     double normdivBT =dg::blas2::dot(divBT, w3d,divBT);
//     double normdivbT =dg::blas2::dot(divbT, w3d,divbT);
//     double normdivb =dg::blas2::dot(divbsol, w3d,divbsol); 
//     double normdzTf = dg::blas2::dot(derivativeT2, w3d, function2);
//     double normdzT_1 = dg::blas2::dot(derivativeT2, w3d, ones);
//     double normdzT1 = dg::blas2::dot(derivativeTones, w3d, function2);
//     double normfdz = dg::blas2::dot(function2, w3d, derivative2);
//     double norm1dz = dg::blas2::dot(ones, w3d, derivative2);
//     double normfdzTdz = dg::blas2::dot(function2, w3d, dzTdz2);
//     double norm1dzTdz = dg::blas2::dot(ones, w3d, dzTdz2);
//     
//     double norm1dzTB = dg::blas2::dot(ones, w3d, divBT);
//     double normBdz1 = dg::blas2::dot(temp2, w3d, derivativeones);
//     double normfdz1 = dg::blas2::dot(function2, w3d, derivativeones);
// 
    std::cout << "--------------------testing dz" << std::endl;
    double norm = dg::blas2::dot( w3d, solution);
    std::cout << "|| Solution ||   "<<sqrt( norm)<<"\n";
    double err =dg::blas2::dot( w3d, derivative);
    std::cout << "|| Derivative || "<<sqrt( err)<<"\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    err =dg::blas2::dot( w3d, derivative);
    std::cout << "Relative Difference in DZ is "<< sqrt( err/norm )<<"\n"; 
   
//     std::cout << "--------------------testing dz with RZPhi method" << std::endl;
//     std::cout << "|| Solution ||   "<<sqrt( norm)<<"\n";
//     double errRZPhi =dg::blas2::dot( w3d, derivativeRZPhi);
//     std::cout << "|| Derivative || "<<sqrt( errRZPhi)<<"\n";
//     dg::blas1::axpby( 1., solution, -1., derivativeRZPhi);
//     errRZPhi =dg::blas2::dot( w3d, derivativeRZPhi);    
//     std::cout << "Relative Difference in DZ is "<< sqrt( errRZPhi/norm )<<"\n"; 
//     
//     std::cout << "--------------------testing dzT" << std::endl;
//     std::cout << "|| divbsol ||  "<<sqrt( normdivb)<<"\n";
//     std::cout << "|| divbT  ||   "<<sqrt( normdivbT)<<"\n";
//     dg::blas1::axpby( 1., divbsol, -1., divbT);
//     normdivbT =dg::blas2::dot(divbT, w3d,divbT);
//     std::cout << "Relative Difference in DZT is   "<<sqrt( normdivbT)<<"\n";
//     std::cout << "-------------------- " << std::endl;
//     std::cout << "|| divB || "<<sqrt( normdivBT)<<"\n";
// 
//     
//     std::cout << "-------------------- " << std::endl;
//     double normT = dg::blas2::dot( w3d, solutionT);
//     std::cout << "|| SolutionT  ||  "<<sqrt( normT)<<"\n";
//     double errT =dg::blas2::dot( w3d, derivativeT);
//     std::cout << "|| DerivativeT || "<<sqrt( errT)<<"\n";
//     dg::blas1::axpby( 1., solutionT, -1., derivativeT);
//     errT =dg::blas2::dot( w3d, derivativeT);
//     std::cout << "Relative Difference in DZT is "<< sqrt( errT/normT )<<"\n"; 
//     dg::blas1::axpby( 1., derivative, -1., derivativeT,omega);
//     double errTdiffdzdzT =dg::blas2::dot( w3d, omega);
//     std::cout << "Relative Difference in DZT to DZ is "<< sqrt( errTdiffdzdzT/norm )<<"\n";   
//     std::cout << "--------------------testing dzT with dz" << std::endl;
//     std::cout << "|| SolutionT ||     "<<sqrt( normT)<<"\n";
//     double errTdz =dg::blas2::dot( w3d, derivativeTdz);
//     std::cout << "|| DerivativeTdz || "<<sqrt( errTdz)<<"\n";
//     dg::blas1::axpby( 1., solutionT, -1., derivativeTdz);
//     errTdz =dg::blas2::dot( w3d, derivativeTdz);
//     std::cout << "Relative Difference in DZT is "<< sqrt( errTdz/normT )<<"\n"; 
//     std::cout << "--------------------testing dzTdz " << std::endl;
  
    double normdzTdz = dg::blas2::dot( w3d, solutiondzTdz);
//     std::cout << std::setprecision(16);
//     std::cout << "is the norm of the testfunction zero ? =       "<<sqrt( normdzTdz)<<"\n";
//     for (unsigned i=0; i<g3d.size();i++){
//         std::cout << "solvalue " << solutiondzTdz[i]<< std::endl;
//     }
//     double normnormdzTdz = dg::blas2::dot(ones, w3d, solutiondzTdz);
//     std::cout << "is the norm of the testfunction zero ? =       "<< normnormdzTdz<<"\n";


//     std::cout << "|| SolutionT ||      "<<sqrt( normdzTdz)<<"\n";
//     double errdzTdz =dg::blas2::dot( w3d,dzTdz);
//     std::cout << "|| DerivativeTdz ||  "<<sqrt( errdzTdz)<<"\n";
//     dg::blas1::axpby( 1., solutiondzTdz, -1., dzTdz);
//     errdzTdz =dg::blas2::dot( w3d, dzTdz);
//     std::cout << "Relative Difference in DZT is "<< sqrt( errdzTdz/normdzTdz )<<"\n";   
    
    std::cout << "--------------------testing dzTdzfb " << std::endl;
    std::cout << "|| SolutionT ||      "<<sqrt( normdzTdz)<<"\n";
    double errdzTdzfb =dg::blas2::dot( w3d,dzTdzfb);
    std::cout << "|| DerivativeTdz ||  "<<sqrt( errdzTdzfb)<<"\n";
    dg::blas1::axpby( 1., solutiondzTdz, -1., dzTdzfb);
    errdzTdzfb =dg::blas2::dot( w3d, dzTdzfb);
    std::cout << "Relative Difference in DZT is "<< sqrt( errdzTdzfb/normdzTdz )<<"\n";
//   
//     std::cout << "--------------------testing dzTdzfb with direct method" << std::endl;
//     std::cout << "|| SolutionT ||      "<<sqrt( normdzTdz)<<"\n";
//     double errdzTdzfbd =dg::blas2::dot( w3d,dzTdzfbd);
//     std::cout << "|| DerivativeTdz ||  "<<sqrt( errdzTdzfbd)<<"\n";
//     dg::blas1::axpby( 1., solutiondzTdz, -1., dzTdzfbd);
//     errdzTdzfbd =dg::blas2::dot( w3d, dzTdzfbd);
//     std::cout << "Relative Difference in DZT is "<< sqrt( errdzTdzfbd/normdzTdz )<<"\n";
    

//     std::cout << "--------------------testing dzTdz with dzz" << std::endl;
//     double normdzz = dg::blas2::dot( w3d, solutiondzz);
// 
//     std::cout << "|| Solution ||      "<<sqrt( normdzz)<<"\n";
//     double errdzz =dg::blas2::dot( w3d,dzz);
//     std::cout << "|| dzz ||  "<<sqrt( errdzz)<<"\n";
//     dg::blas1::axpby( 1., solutiondzz, -1., dzz);
//     errdzz =dg::blas2::dot( w3d, dzz);
//     std::cout << "Relative Difference in DZT is "<< sqrt( errdzz/normdzz )<<"\n";   
//     
//     std::cout << "--------------------testing adjointness " << std::endl;
//     std::cout << "<f,dz(f)>   = "<< normfdz<<"\n";
//     std::cout << "-<dzT(f),f> = "<< -normdzTf<<"\n";
//     std::cout << "Diff        = "<< normfdz+normdzTf<<"\n";     
//     std::cout << "-------------------- " << std::endl;
// 
//     std::cout << "<B,dz(1)>   = "<< normBdz1<<"\n";
//     std::cout << "-<dzT(B),1> = "<< -norm1dzTB<<"\n";
//     std::cout << "Diff        = "<< normBdz1+norm1dzTB<<"\n";     
//     std::cout << "-------------------- " << std::endl;
//     
//     std::cout << "<f,dz(1)>   = "<< normfdz1<<"\n";
//     std::cout << "-<dzT(f),1> = "<< -normdzT_1<<"\n";
//     std::cout << "Diff        = "<< normfdz1+normdzT_1<<"\n";   
//     std::cout << "-------------------- " << std::endl;
//     
//     std::cout << "<1,dz(f)>   = "<< norm1dz<<"\n";
//     std::cout << "-<dzT(1),f> = "<< -normdzT1<<"\n";
//     std::cout << "Diff        = "<< norm1dz+normdzT1<<"\n";   
//     std::cout << "-------------------- " << std::endl;
//   
//     std::cout << "<f,dzT(dz(f))> = "<< normfdzTdz<<"\n";
//     std::cout << "-<dz(f),dz(f)> = "<< -normdzdz<<"\n";
//     std::cout << "Diff           = "<< normfdzTdz+normdzdz<<"\n";     
//     std::cout << "-------------------- " << std::endl;
//    
//     std::cout << "<1,dzT(dz(f))> = "<< norm1dzTdz<<"\n";
//     std::cout << "-<dz(1),dz(f)> = "<< -normdz1dz<<"\n";
//     std::cout << "Diff           = "<< norm1dzTdz+normdz1dz<<"\n";    
//     
// 
//     std::cout << "--------------------testing GeneralElliptic with inversion " << std::endl; 
//    //set up the parallel diffusion
//     dg::GeneralEllipticSym<dg::DMatrix, dg::DVec, dg::DVec> elliptic( g3d, dg::not_normed, dg::forward);
//     elliptic.set_x(bhatR);
//     elliptic.set_y(bhatZ );
//     elliptic.set_z(bhatPhi);
    
    
    double eps =1e-8;   
    dg::Invert< dg::DVec> invert( dg::evaluate(dg::zero,g3d), w3d.size(), eps );  
    std::cout << "MAX # iterations = " << w3d.size() << std::endl;
// 
//    const dg::DVec rhs = dg::evaluate( solovev::DeriNeuT2( gp.R_0, gp.I_0), g3d);
// // 
//     std::cout << " # of iterations "<< invert( elliptic, functionTinv, rhs ) << std::endl; //is dzTdz 
//   
    double normf = dg::blas2::dot( w3d, function);
// 
//     std::cout << "Norm analytic Solution  "<<sqrt( normf)<<"\n";
    double errinvT =dg::blas2::dot( w3d, functionTinv);
//     std::cout << "Norm numerical Solution "<<sqrt( errinvT)<<"\n";
// 
//     dg::blas1::axpby( 1., function, +1.,functionTinv);
//     errinvT =dg::blas2::dot( w3d, functionTinv);
//     std::cout << "Relative Difference is  "<< sqrt( errinvT/normf )<<"\n";
//     
    std::cout << "--------------------testing dzT" << std::endl; 
    std::cout << " # of iterations "<< invert( dzNU, functionTinv2,solutiondzTdz ) << std::endl; //is dzTdz
    std::cout << "Norm analytic Solution  "<<sqrt( normf)<<"\n";
    double errinvT2 =dg::blas2::dot( w3d, functionTinv2);
    std::cout << "Norm numerical Solution "<<sqrt( errinvT2)<<"\n";
    dg::blas1::axpby( 1., function, -1.,functionTinv2);
    errinvT2 =dg::blas2::dot( w3d, functionTinv2);
    std::cout << "Relative Difference is  "<< sqrt( errinvT2/normf )<<"\n";

//write netcdf
//     file::NC_Error_Handle err;
//     int ncid,tvarID;
//     err = nc_create( "out3.nc",NC_NETCDF4|NC_CLOBBER, &ncid);
//     dg::DVec transferD( dg::evaluate(dg::zero, g3d));
//     dg::HVec transferH( dg::evaluate(dg::zero, g3d));
// 
//     int dim_ids[4];
//     err = file::define_dimensions( ncid, dim_ids, &tvarID, g3d);
//     std::string names[3] = {"TG","TD","TA"}; 
//     int dataIDs[3]; 
//     size_t start[4] = {0, 0, 0, 0};
//     size_t count[4] = {1, g3d.Nz(), g3d.n()*g3d.Ny(), g3d.n()*g3d.Nx()};
//     err = nc_def_var( ncid, names[0].data(), NC_DOUBLE, 4, dim_ids, &dataIDs[0]);  
//     err = nc_def_var( ncid, names[1].data(), NC_DOUBLE, 4, dim_ids, &dataIDs[1]);  
//     err = nc_def_var( ncid, names[2].data(), NC_DOUBLE, 4, dim_ids, &dataIDs[2]);
//     err = nc_enddef( ncid);
//     err = nc_open("out3.nc", NC_WRITE, &ncid);
//     transferD=dzTdz;
//     transferH =transferD;
//     err = nc_put_vara_double( ncid, dataIDs[0], start, count, transferH.data());
//     transferD=dzTdzfbd;
//     transferH =transferD;
//     err = nc_put_vara_double( ncid, dataIDs[1], start, count, transferH.data());    
//     transferD=dzTdzfb;
//     transferH =transferD;
//     err = nc_put_vara_double( ncid, dataIDs[2], start, count, transferH.data());    
//      err = nc_close(ncid);
    }
    
//     std::cout << "make Plot" << std::endl;
//     //make equidistant grid from dggrid
//     dg::HVec hvisual;
//     //allocate mem for visual
//     dg::HVec visual;
//     dg::HMatrix equigrid = dg::create::backscatter(g3d);               
// 
//     //evaluate on valzues from devicevector on equidistant visual hvisual vector
//     visual = dg::evaluate( dg::one, g3d);
//     //Create Window and set window title
//     GLFWwindow* w = draw::glfwInitAndCreateWindow( 100*Nz, 700, "");
//     draw::RenderHostData render(7 , 1*Nz);  
//     //create a colormap
//     draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);
//     dg::DMatrix jump( dg::create::jump2d( g3d, g3d.bcx(), g3d.bcy(), dg::not_normed));
//     dg::blas2::symv( jump, ones, lambda);

//     std::stringstream title;
//     title << std::setprecision(10) << std::scientific;

//     while (!glfwWindowShouldClose( w ))
//     {
//         hvisual = divBT;
//         dg::blas2::gemv( equigrid, hvisual, visual);        
//         colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), -100000000., thrust::maximum<double>()   );
//         colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
//         title <<"divB"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
//         for( unsigned k=0; k<Nz;k++)
//         {
//             unsigned size=g3d.n()*g3d.n()*g3d.Nx()*g3d.Ny();            
//             dg::HVec part( visual.begin() + k*size, visual.begin()+(k+1)*size);
//             render.renderQuad( part, g3d.n()*g3d.Nx(), g3d.n()*g3d.Ny(), colors);
// 
//         }
//         hvisual = derivativeT;         
//         dg::blas2::gemv( equigrid, hvisual, visual);
//         colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), -100000000., thrust::maximum<double>()   );
//         colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
//         title <<"dzT(f)"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
//         for( unsigned k=0; k<Nz;k++)
//         {
//             unsigned size=g3d.n()*g3d.n()*g3d.Nx()*g3d.Ny();            
//             dg::HVec part( visual.begin() + k*size, visual.begin()+(k+1)*size);
//             render.renderQuad( part, g3d.n()*g3d.Nx(), g3d.n()*g3d.Ny(), colors);
//         }
//         hvisual = derivative;
//         dg::blas2::gemv( equigrid, hvisual, visual);
//         colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), -100000000., thrust::maximum<double>()   );
//         colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
//         title <<"dz(f)"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
//         for( unsigned k=0; k<Nz;k++)
//         {            
//             unsigned size=g3d.n()*g3d.n()*g3d.Nx()*g3d.Ny();            
//             dg::HVec part( visual.begin() + k*size, visual.begin()+(k+1)*size);
//             render.renderQuad( part, g3d.n()*g3d.Nx(), g3d.n()*g3d.Ny(), colors);
//         }
//         dg::blas1::axpby(1.0,derivative,-1.0,derivativeT,omega);
//         hvisual = omega;
//         dg::blas2::gemv( equigrid, hvisual, visual);
//         colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), -100000000., thrust::maximum<double>()   );
//         colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
//         title <<"diff"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
//         for( unsigned k=0; k<Nz;k++)
//         {            
//             unsigned size=g3d.n()*g3d.n()*g3d.Nx()*g3d.Ny();            
//             dg::HVec part( visual.begin() + k*size, visual.begin()+(k+1)*size);
//             render.renderQuad( part, g3d.n()*g3d.Nx(), g3d.n()*g3d.Ny(), colors);
//         }
//         hvisual = dzTdz;
//         dg::blas2::gemv( equigrid, hvisual, visual);
//         colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), -100000000., thrust::maximum<double>()   );
//         colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
//         title <<"dzTdzfb"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
//         for( unsigned k=0; k<Nz;k++)
//         {            
//             unsigned size=g3d.n()*g3d.n()*g3d.Nx()*g3d.Ny();            
//             dg::HVec part( visual.begin() + k*size, visual.begin()+(k+1)*size);
//             render.renderQuad( part, g3d.n()*g3d.Nx(), g3d.n()*g3d.Ny(), colors);
//         }
//         hvisual = dzTdz;
//         dg::blas2::gemv( equigrid, hvisual, visual);
//         colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), -100000000., thrust::maximum<double>()   );
//         colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
//         title <<"dzTdz"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
//         for( unsigned k=0; k<Nz;k++)
//         {            
//             unsigned size=g3d.n()*g3d.n()*g3d.Nx()*g3d.Ny();            
//             dg::HVec part( visual.begin() + k*size, visual.begin()+(k+1)*size);
//             render.renderQuad( part, g3d.n()*g3d.Nx(), g3d.n()*g3d.Ny(), colors);
//         }
//         hvisual = functionTinv2;
//         dg::blas2::gemv( equigrid, hvisual, visual);
//         colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), -100000000., thrust::maximum<double>()   );
//         colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
//         title <<"dzz"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
//         for( unsigned k=0; k<Nz;k++)
//         {            
//             unsigned size=g3d.n()*g3d.n()*g3d.Nx()*g3d.Ny();            
//             dg::HVec part( visual.begin() + k*size, visual.begin()+(k+1)*size);
//             render.renderQuad( part, g3d.n()*g3d.Nx(), g3d.n()*g3d.Ny(), colors);
//         }
//         title << std::fixed; 
//         glfwSetWindowTitle(w,title.str().c_str());
//         title.str("");
//         glfwSwapBuffers(w);
//         glfwWaitEvents();
//     }
// 
//     glfwTerminate();
    return 0;
}
