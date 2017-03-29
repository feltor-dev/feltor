#include <iostream>

#include <mpi.h>
#include <cusp/print.h>
#include <cusp/csr_matrix.h>
#include "file/read_input.h"
// #include "file/nc_utilities.h"

#include "dg/backend/xspacelib.cuh"
#include "dg/backend/evaluation.cuh"
#include "dg/backend/timer.cuh"
#include "dg/blas.h"
#include "dg/ds.h"
#include "dg/backend/functions.h"
#include "dg/functors.h"
#include "dg/elliptic.h"
#include "dg/cg.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/typedefs.cuh"
// #include "draw/host_window.h"
#include "guenther.h"
#include "magnetic_field.h"
#include "testfunctors.h"

using namespace dg::geo::guenther;

int main( int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int periods[3] = {false, false, true}; //non-, non-, periodic
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    int np[3];
    if(rank==0)
    {
        std::cout << "Type npx, npy, npz!\n";
        std::cin>> np[0] >> np[1] >>np[2];
        std::cout << "Computing with "<<np[0]<<" x "<<np[1]<<" x "<<np[2] << " = "<<size<<std::endl;
        assert( size == np[0]*np[1]*np[2]);
    }
    MPI_Bcast( np, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 3, np, periods, true, &comm);

    /////////////////initialize params////////////////////////////////
    Json::Reader reader;
    Json::Value js;
    std::ifstream is("guenther_params.js");
    reader.parse(is,js,false);
    GeomParameters gp(js);
//     gp.display( std::cout);

    //////////////////////////////////////////////////////////////////////////
    
    double Rmin=gp.R_0-1.0*gp.a;
    double Zmin=-1.0*gp.a*gp.elongation;
    double Rmax=gp.R_0+1.0*gp.a; 
    double Zmax=1.0*gp.a*gp.elongation;
    /////////////////////////////////////////////initialze fields /////////////////////
    
    Field field(gp.R_0, gp.I_0);
    InvB invb(gp.R_0, gp.I_0);
    GradLnB gradlnB(gp.R_0, gp.I_0);
    LnB lnB(gp.R_0, gp.I_0);
    FieldR bR_(gp.R_0, gp.I_0);
    FieldZ bZ_(gp.R_0, gp.I_0);
    FieldP bPhi_(gp.R_0, gp.I_0);
    FuncNeu funcNEU(gp.R_0,gp.I_0);
    FuncNeu2 funcNEU2(gp.R_0,gp.I_0);
    DeriNeu deriNEU(gp.R_0,gp.I_0);
    DeriNeu2 deriNEU2(gp.R_0,gp.I_0);
    DeriNeuT2 deriNEUT2(gp.R_0,gp.I_0);
    DeriNeuT deriNEUT(gp.R_0,gp.I_0);
    Divb divb(gp.R_0,gp.I_0);
    B Bfield(gp.R_0, gp.I_0);
    
    //std::cout << "Type n, Nx, Ny, Nz\n";
    //std::cout << "Note, that function is resolved exactly in R,Z for n > 2\n";
    unsigned n=3, Nx=5, Ny=5, Nz=5;
    //std::cin >> n>> Nx>>Ny>>Nz;
    unsigned Nxn = Nx;
    unsigned Nyn = Ny;
    unsigned Nzn = Nz;

    double rk4eps = 1e-8;
    //std::cout << "Type RK4 eps (1e-8)\n";
    //std::cin >> rk4eps;
    double z0 = 0, z1 = 2.*M_PI;
    for (unsigned i=1;i<4;i+=2) { 

        Nzn = unsigned(Nz*pow(2,i));
        Nxn = (unsigned)ceil(Nx*pow(2,(double)(i*2./n)));
        Nyn = (unsigned)ceil(Ny*pow(2,(double)(i*2./n)));



        dg::CylindricalMPIGrid3d<dg::MDVec> g3d( Rmin,Rmax, Zmin,Zmax, z0, z1,  n,Nxn ,Nyn, Nzn,dg::DIR, dg::DIR, dg::PER, comm);
        dg::MPIGrid2d g2d( Rmin,Rmax, Zmin,Zmax,  n, Nxn ,Nyn, dg::DIR, dg::DIR, comm);

        if(rank==0)std::cout << "NR = " << Nxn << std::endl;
        if(rank==0)std::cout << "NZ = " << Nyn<< std::endl;
        if(rank==0)std::cout << "Nphi = " << Nzn << std::endl;
//            Nxn = (unsigned)ceil(Nxn*pow(2,(double)(2./n)));
//     Nyn = (unsigned)ceil( Nyn*pow(2,(double)(2./n)));

//        dg::Grid3d g3d( Rmin,Rmax, Zmin,Zmax, z0, z1,  n, Nx, Ny, Nz*pow(2,i),dg::DIR, dg::DIR, dg::PER,dg::cylindrical);
//     dg::Grid2d g2d( Rmin,Rmax, Zmin,Zmax,  n, Nx, Ny); 
    const dg::MDVec w3d = dg::create::volume( g3d);
    const dg::MDVec w2d = dg::create::weights( g2d);
    const dg::MDVec v3d = dg::create::inv_volume( g3d);

    if(rank==0)std::cout << "computing dsDIR" << std::endl;
    dg::MDDS::FieldAligned dsFA( field, g3d, rk4eps, dg::DefaultLimiter(), dg::DIR);
    if(rank==0)std::cout << "computing dsNEU" << std::endl;
    dg::MDDS::FieldAligned dsNUFA( field, g3d, rk4eps, dg::DefaultLimiter(), dg::NEU);


    dg::MDDS ds ( dsFA, field, dg::not_normed, dg::centered), 
         dsNU ( dsNUFA, field, dg::not_normed, dg::centered);

//     dg::DS<dg::DMatrix, dg::MDVec> dsNEU( field, g3d, g3d.hz(), rk4eps, dg::DefaultLimiter(), dg::NEU);
    
//     dg::Grid3d g3dp( Rmin,Rmax, Zmin,Zmax, z0, z1,  n, Nx, Ny, 1);
    
//     dg::DS<dg::DMatrix, dg::MDVec> ds2d( field, g3dp, g3d.hz(), rk4eps, dg::DefaultLimiter(), dg::NEU);
    dg::MDVec boundary=dg::evaluate( dg::zero, g3d);
    
    dg::MDVec function = dg::evaluate( funcNEU, g3d) ,
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
                        derivativeTds(function),
                        functionTinv(dg::evaluate( dg::zero, g3d)),
                        functionTinv2(dg::evaluate( dg::zero, g3d)),
                        dsTds(function),
                        dsTdsb(function),
                        dsTdsf(function),
                        dsTdsbd(function),
                        dsTdsfd(function),
                        dsTdsfb(function),
                        dsTdsfbd(function),
                        dsz(function),
                        divbsol(dg::evaluate(divb, g3d)),
                        divbT(function),
                        divBT(function),
                        lambda(function),
                        omega(function),
                        dsTds2(function);


    dg::MDVec ones = dg::evaluate( dg::one, g3d);
    const dg::MDVec function2 = dg::evaluate( funcNEU2, g3d);
    const dg::MDVec solution = dg::evaluate( deriNEU, g3d);
    const dg::MDVec solutionT = dg::evaluate( deriNEUT, g3d);
    const dg::MDVec solutiondsz = dg::evaluate( deriNEU2, g3d);
    const dg::MDVec solutiondsTds = dg::evaluate( deriNEUT2, g3d);

    const dg::MDVec bhatR = dg::evaluate( bR_, g3d);
    const dg::MDVec bhatZ = dg::evaluate( bZ_, g3d);
    const dg::MDVec bhatPhi = dg::evaluate(bPhi_, g3d);
//     const dg::MDVec Bfield_ = dg::evaluate(Bfield, g3d);
    const dg::MDVec gradlnB_ = dg::evaluate(gradlnB, g3d);
//     dg::DMatrix dR(dg::create::dx( g3d, g3d.bcx(),dg::normed,dg::centered));
//     dg::DMatrix dZ(dg::create::dy( g3d, g3d.bcy(),dg::normed,dg::centered));
//     dg::DMatrix dphi(dg::create::ds( g3d, g3d.bcz(), dg::normed,dg::centered));
    
//     ds.set_boundaries( dg::PER, 0, 0);
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
//     dg::GeneralEllipticSym<dg::DMatrix, dg::MDVec, dg::MDVec> ellipticsym( g3d, dg::normed, dg::forward);
//     ellipticsym.set_x(bhatR);
//     ellipticsym.set_y(bhatZ );
//     ellipticsym.set_z(bhatPhi);
//     
//     
  
    dsNU( function, derivative); //ds(f)

//     dsNU.forward( function, derivativef); //ds(f)
//     dsNU.backward( function, derivativeb); //ds(f)

//     ds( ones, derivativeones); //ds(f)
//     ds( function2, derivative2); //ds(f)
//     //compute dsz
//     ds( inverseB, lambda); //gradpar 1/B
//     dg::blas1::pointwiseDivide(lambda,  inverseB, lambda); //-ds lnB
//     ds(function,omega); //ds T
//     dg::blas1::pointwiseDot(omega, lambda, omega);            //- ds lnB ds T
//     dg::blas1::pointwiseDot(omega, gradlnB_, omega);            //- ds lnB ds T
//     dg::blas1::axpby(1.0, omega, 0., dsz,dsz);    
    //     dg::blas1::axpby(-1.0, omega, 0., dsz,dsz);    

    
//     dsNU.forward(derivativeb,temp);
//     dsNU.backward(derivativef,omega);
//     dg::blas1::axpby( -1.0, omega, -0.0, temp,dsz);
//     ds( derivative, dsz); //ds(ds(f))

//     ds.dsz(function,dsz);       
//     dsNU( function, derivative); //ds(f)
//     dg::blas1::pointwiseDot(derivative, gradlnB_, omega);            //- ds lnB ds T
// 
//ds^2 T 
//     dg::blas1::axpby( -1.0, omega, 1.,dsz, dsTdsfbd);

//     
//     
//     ds.centeredT(function, derivativeT); //ds(f)
// 
//     //divB
//     dg::blas1::pointwiseDivide(ones,  inverseB, temp2); //B
//     ds.centeredT(temp2, divBT); // dsT B
// 
//     
//     ds.centeredT( function2, derivativeT2); //ds(f)
//     ds.centeredT( ones, derivativeTones); //ds(f)
    //B ds f/B
//     dg::blas1::pointwiseDot( inverseB, function, temp);
//     ds( temp, derivativeTds);
//     dg::blas1::pointwiseDot( derivativeTds, Bfield_, derivativeTds);
    //oder ds f - f dslnB
//     ds( function, derivativeTds);
//     dg::blas1::pointwiseDot(function,gradlnB_,temp);
//     dg::blas1::axpby(- 1.0, temp, 1., derivativeTds,derivativeTds);


    
    //     dg::blas1::pointwiseDivide( derivativeTds, inverseB, derivativeTds);
//     
//     ds.centeredT( derivative, dsTds); //dsT(ds(f))
//     
//     //overwrite with sym from adjoint dg
//     ellipticsym.symv(function,dsTds);
//     dg::blas1::scal(dsTds,-1.0);
// //     ds.centeredT(ones,divbT);
    ds.forwardT( derivativef, dsTdsf);  //dsT(ds(f))
//     ds.backwardT( derivativeb, dsTdsb); //dsT(ds(f))

//     //centered
//     ds.centeredTD(derivative,dsTdsfbd);
//     ds.forwardTD( derivativef, dsTdsfd); //dsT(ds(f))
//     ds.backwardTD( derivativeb, dsTdsbd); //dsT(ds(f))

//     //arithmetic average
//     dg::blas1::axpby(0.5,dsTdsb,0.5,dsTdsf,dsTdsfb);
//     dg::blas1::axpby(0.5,dsTdsbd,0.5,dsTdsfd,dsTdsfbd); 
    ds.symv(function,dsTdsfb);
    dg::blas1::pointwiseDot(v3d,dsTdsfb,dsTdsfb);
//     ds.centeredT( derivative2, dsTds2); //dsT(ds(f))
//     dg::blas1::pointwiseDivide(ones,  inverseB, temp2); //B
//     ds.centeredT( ones, divbT);
//     
//     double normdsds =dg::blas2::dot(derivative2, w3d,derivative2);
//     double normds1ds =dg::blas2::dot(derivativeones, w3d,derivative2);
//     double normdivBT =dg::blas2::dot(divBT, w3d,divBT);
//     double normdivbT =dg::blas2::dot(divbT, w3d,divbT);
//     double normdivb =dg::blas2::dot(divbsol, w3d,divbsol); 
//     double normdsTf = dg::blas2::dot(derivativeT2, w3d, function2);
//     double normdsT_1 = dg::blas2::dot(derivativeT2, w3d, ones);
//     double normdsT1 = dg::blas2::dot(derivativeTones, w3d, function2);
//     double normfds = dg::blas2::dot(function2, w3d, derivative2);
//     double norm1ds = dg::blas2::dot(ones, w3d, derivative2);
//     double normfdsTds = dg::blas2::dot(function2, w3d, dsTds2);
//     double norm1dsTds = dg::blas2::dot(ones, w3d, dsTds2);
//     
//     double norm1dsTB = dg::blas2::dot(ones, w3d, divBT);
//     double normBds1 = dg::blas2::dot(temp2, w3d, derivativeones);
//     double normfds1 = dg::blas2::dot(function2, w3d, derivativeones);
// 
    if(rank==0)std::cout << "--------------------testing ds" << std::endl;
    double norm = dg::blas2::dot( w3d, solution);
    if(rank==0)std::cout << "|| Solution ||   "<<sqrt( norm)<<"\n";
    double err =dg::blas2::dot( w3d, derivative);
    if(rank==0)std::cout << "|| Derivative || "<<sqrt( err)<<"\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    err =dg::blas2::dot( w3d, derivative);
    if(rank==0)std::cout << "Relative Difference in DS is "<< sqrt( err/norm )<<"\n"; 
   
//     std::cout << "--------------------testing ds with RZPhi method" << std::endl;
//     std::cout << "|| Solution ||   "<<sqrt( norm)<<"\n";
//     double errRZPhi =dg::blas2::dot( w3d, derivativeRZPhi);
//     std::cout << "|| Derivative || "<<sqrt( errRZPhi)<<"\n";
//     dg::blas1::axpby( 1., solution, -1., derivativeRZPhi);
//     errRZPhi =dg::blas2::dot( w3d, derivativeRZPhi);    
//     std::cout << "Relative Difference in DS is "<< sqrt( errRZPhi/norm )<<"\n"; 
//     
//     std::cout << "--------------------testing dsT" << std::endl;
//     std::cout << "|| divbsol ||  "<<sqrt( normdivb)<<"\n";
//     std::cout << "|| divbT  ||   "<<sqrt( normdivbT)<<"\n";
//     dg::blas1::axpby( 1., divbsol, -1., divbT);
//     normdivbT =dg::blas2::dot(divbT, w3d,divbT);
//     std::cout << "Relative Difference in DST is   "<<sqrt( normdivbT)<<"\n";
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
//     std::cout << "Relative Difference in DST is "<< sqrt( errT/normT )<<"\n"; 
//     dg::blas1::axpby( 1., derivative, -1., derivativeT,omega);
//     double errTdiffdsdsT =dg::blas2::dot( w3d, omega);
//     std::cout << "Relative Difference in DST to DS is "<< sqrt( errTdiffdsdsT/norm )<<"\n";   
//     std::cout << "--------------------testing dsT with ds" << std::endl;
//     std::cout << "|| SolutionT ||     "<<sqrt( normT)<<"\n";
//     double errTds =dg::blas2::dot( w3d, derivativeTds);
//     std::cout << "|| DerivativeTds || "<<sqrt( errTds)<<"\n";
//     dg::blas1::axpby( 1., solutionT, -1., derivativeTds);
//     errTds =dg::blas2::dot( w3d, derivativeTds);
//     std::cout << "Relative Difference in DST is "<< sqrt( errTds/normT )<<"\n"; 
//     std::cout << "--------------------testing dsTds " << std::endl;
  
    double normdsTds = dg::blas2::dot( w3d, solutiondsTds);
//     std::cout << std::setprecision(16);
//     std::cout << "is the norm of the testfunction zero ? =       "<<sqrt( normdsTds)<<"\n";
//     for (unsigned i=0; i<g3d.size();i++){
//         std::cout << "solvalue " << solutiondsTds[i]<< std::endl;
//     }
//     double normnormdsTds = dg::blas2::dot(ones, w3d, solutiondsTds);
//     std::cout << "is the norm of the testfunction zero ? =       "<< normnormdsTds<<"\n";


//     std::cout << "|| SolutionT ||      "<<sqrt( normdsTds)<<"\n";
//     double errdsTds =dg::blas2::dot( w3d,dsTds);
//     std::cout << "|| DerivativeTds ||  "<<sqrt( errdsTds)<<"\n";
//     dg::blas1::axpby( 1., solutiondsTds, -1., dsTds);
//     errdsTds =dg::blas2::dot( w3d, dsTds);
//     std::cout << "Relative Difference in DST is "<< sqrt( errdsTds/normdsTds )<<"\n";   
    
    if(rank==0)std::cout << "--------------------testing dsTdsfb " << std::endl;
    if(rank==0)std::cout << "|| SolutionT ||      "<<sqrt( normdsTds)<<"\n";
    double errdsTdsfb =dg::blas2::dot( w3d,dsTdsfb);
    if(rank==0)std::cout << "|| DerivativeTds ||  "<<sqrt( errdsTdsfb)<<"\n";
    dg::blas1::axpby( 1., solutiondsTds, -1., dsTdsfb);
    errdsTdsfb =dg::blas2::dot( w3d, dsTdsfb);
    if(rank==0)std::cout << "Relative Difference in DST is "<< sqrt( errdsTdsfb/normdsTds )<<"\n";
//   
//     std::cout << "--------------------testing dsTdsfb with direct method" << std::endl;
//     std::cout << "|| SolutionT ||      "<<sqrt( normdsTds)<<"\n";
//     double errdsTdsfbd =dg::blas2::dot( w3d,dsTdsfbd);
//     std::cout << "|| DerivativeTds ||  "<<sqrt( errdsTdsfbd)<<"\n";
//     dg::blas1::axpby( 1., solutiondsTds, -1., dsTdsfbd);
//     errdsTdsfbd =dg::blas2::dot( w3d, dsTdsfbd);
//     std::cout << "Relative Difference in DST is "<< sqrt( errdsTdsfbd/normdsTds )<<"\n";
    

//     std::cout << "--------------------testing dsTds with dsz" << std::endl;
//     double normdsz = dg::blas2::dot( w3d, solutiondsz);
// 
//     std::cout << "|| Solution ||      "<<sqrt( normdsz)<<"\n";
//     double errdsz =dg::blas2::dot( w3d,dsz);
//     std::cout << "|| dsz ||  "<<sqrt( errdsz)<<"\n";
//     dg::blas1::axpby( 1., solutiondsz, -1., dsz);
//     errdsz =dg::blas2::dot( w3d, dsz);
//     std::cout << "Relative Difference in DST is "<< sqrt( errdsz/normdsz )<<"\n";   
//     
//     std::cout << "--------------------testing adjointness " << std::endl;
//     std::cout << "<f,ds(f)>   = "<< normfds<<"\n";
//     std::cout << "-<dsT(f),f> = "<< -normdsTf<<"\n";
//     std::cout << "Diff        = "<< normfds+normdsTf<<"\n";     
//     std::cout << "-------------------- " << std::endl;
// 
//     std::cout << "<B,ds(1)>   = "<< normBds1<<"\n";
//     std::cout << "-<dsT(B),1> = "<< -norm1dsTB<<"\n";
//     std::cout << "Diff        = "<< normBds1+norm1dsTB<<"\n";     
//     std::cout << "-------------------- " << std::endl;
//     
//     std::cout << "<f,ds(1)>   = "<< normfds1<<"\n";
//     std::cout << "-<dsT(f),1> = "<< -normdsT_1<<"\n";
//     std::cout << "Diff        = "<< normfds1+normdsT_1<<"\n";   
//     std::cout << "-------------------- " << std::endl;
//     
//     std::cout << "<1,ds(f)>   = "<< norm1ds<<"\n";
//     std::cout << "-<dsT(1),f> = "<< -normdsT1<<"\n";
//     std::cout << "Diff        = "<< norm1ds+normdsT1<<"\n";   
//     std::cout << "-------------------- " << std::endl;
//   
//     std::cout << "<f,dsT(ds(f))> = "<< normfdsTds<<"\n";
//     std::cout << "-<ds(f),ds(f)> = "<< -normdsds<<"\n";
//     std::cout << "Diff           = "<< normfdsTds+normdsds<<"\n";     
//     std::cout << "-------------------- " << std::endl;
//    
//     std::cout << "<1,dsT(ds(f))> = "<< norm1dsTds<<"\n";
//     std::cout << "-<ds(1),ds(f)> = "<< -normds1ds<<"\n";
//     std::cout << "Diff           = "<< norm1dsTds+normds1ds<<"\n";    
//     
// 
//     std::cout << "--------------------testing GeneralElliptic with inversion " << std::endl; 
//    //set up the parallel diffusion
//     dg::GeneralEllipticSym<dg::DMatrix, dg::MDVec, dg::MDVec> elliptic( g3d, dg::not_normed, dg::forward);
//     elliptic.set_x(bhatR);
//     elliptic.set_y(bhatZ );
//     elliptic.set_z(bhatPhi);
    
    
    double eps =1e-8;   
    dg::Invert< dg::MDVec> invert( dg::evaluate(dg::zero,g3d), w3d.size(), eps );  
    if(rank==0)std::cout << "MAX # iterations = " << w3d.size() << std::endl;
// 
//    const dg::MDVec rhs = dg::evaluate( solovev::DeriNeuT2( gp.R_0, gp.I_0), g3d);
// // 
//     std::cout << " # of iterations "<< invert( elliptic, functionTinv, rhs ) << std::endl; //is dsTds 
//   
    double normf = dg::blas2::dot( w3d, function);
// 
//     std::cout << "Norm analytic Solution  "<<sqrt( normf)<<"\n";
//     double errinvT =dg::blas2::dot( w3d, functionTinv);
//     std::cout << "Norm numerical Solution "<<sqrt( errinvT)<<"\n";
// 
//     dg::blas1::axpby( 1., function, +1.,functionTinv);
//     errinvT =dg::blas2::dot( w3d, functionTinv);
//     std::cout << "Relative Difference is  "<< sqrt( errinvT/normf )<<"\n";
//     
    if(rank==0)std::cout << "--------------------testing dsT" << std::endl; 
    unsigned number = invert(dsNU, functionTinv2, solutiondsTds);
    if(rank==0)std::cout << " # of iterations "<< number << std::endl; //is dsTds
    if(rank==0)std::cout << "Norm analytic Solution  "<<sqrt( normf)<<"\n";
    double errinvT2 =dg::blas2::dot( w3d, functionTinv2);
    if(rank==0)std::cout << "Norm numerical Solution "<<sqrt( errinvT2)<<"\n";
    dg::blas1::axpby( 1., function, -1.,functionTinv2);
    errinvT2 =dg::blas2::dot( w3d, functionTinv2);
    if(rank==0)std::cout << "Relative Difference is  "<< sqrt( errinvT2/normf )<<"\n";

//write netcdf
//     file::NC_Error_Handle err;
//     int ncid,tvarID;
//     err = nc_create( "out3.nc",NC_NETCDF4|NC_CLOBBER, &ncid);
//     dg::MDVec transferD( dg::evaluate(dg::zero, g3d));
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
//     transferD=dsTds;
//     transferH =transferD;
//     err = nc_put_vara_double( ncid, dataIDs[0], start, count, transferH.data());
//     transferD=dsTdsfbd;
//     transferH =transferD;
//     err = nc_put_vara_double( ncid, dataIDs[1], start, count, transferH.data());    
//     transferD=dsTdsfb;
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
//         title <<"dsT(f)"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
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
//         title <<"ds(f)"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
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
//         hvisual = dsTds;
//         dg::blas2::gemv( equigrid, hvisual, visual);
//         colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), -100000000., thrust::maximum<double>()   );
//         colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
//         title <<"dsTdsfb"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
//         for( unsigned k=0; k<Nz;k++)
//         {            
//             unsigned size=g3d.n()*g3d.n()*g3d.Nx()*g3d.Ny();            
//             dg::HVec part( visual.begin() + k*size, visual.begin()+(k+1)*size);
//             render.renderQuad( part, g3d.n()*g3d.Nx(), g3d.n()*g3d.Ny(), colors);
//         }
//         hvisual = dsTds;
//         dg::blas2::gemv( equigrid, hvisual, visual);
//         colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), -100000000., thrust::maximum<double>()   );
//         colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
//         title <<"dsTds"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
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
//         title <<"dsz"<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
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
    MPI_Finalize();
    return 0;
}
