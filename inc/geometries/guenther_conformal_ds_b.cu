#include <iostream>

#include <cusp/print.h>
#include <cusp/csr_matrix.h>
#include "file/read_input.h"
// #include "file/nc_utilities.h"

#include "dg/backend/xspacelib.cuh"
#include "dg/backend/evaluation.cuh"
#include "dg/backend/timer.cuh"
#include "dg/blas.h"
#include "dg/backend/functions.h"
#include "dg/functors.h"
#include "dg/elliptic.h"
#include "dg/cg.h"
#include "dg/geometry.h"
// #include "draw/host_window.h"
#include "guenther.h"
//#include "conformal.h"
#include "orthogonal.h"
#include "dg/ds.h"

//typedef dg::FieldAligned< solovev::ConformalRingGrid3d<dg::DVec> , dg::IDMatrix, dg::DVec> DFA;
typedef dg::FieldAligned< solovev::OrthogonalRingGrid3d<dg::DVec> , dg::IDMatrix, dg::DVec> DFA;

int main( )
{

    /////////////////initialize params////////////////////////////////
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0 >> psi_1;
    std::vector<double> v;

    try{
        v = file::read_input( "guenther_params.txt"); 
    }catch( toefl::Message& m){
        m.display();
        return -1;
    }

    const solovev::GeomParameters gp(v);
    gp.display( std::cout);

    /////////////////////////////////////////////initialze fields /////////////////////
    
    guenther::FuncNeu funcNEU(gp.R_0,gp.I_0);
    guenther::DeriNeu deriNEU(gp.R_0,gp.I_0);
    guenther::DeriNeuT2 deriNEUT2(gp.R_0,gp.I_0);
    guenther::FuncMinusDeriNeuT2 fmderiNEUT2(gp.R_0,gp.I_0);
    solovev::BHatR bhatR( gp);
    solovev::BHatZ bhatZ( gp);
    
    //std::cout << "Type n, Nx, Ny, Nz\n";
    //std::cout << "Note, that function is resolved exactly in R,Z for n > 2\n";
    //unsigned n=3, Nx=5, Ny=5, Nz=5;
    //std::cin >> n>> Nx>>Ny>>Nz;
    unsigned Nxn = Nx;
    unsigned Nyn = Ny;
    unsigned Nzn = Nz;

    const double rk4eps = 1e-6;
    //std::cout << "Type RK4 eps (1e-8)\n";
    //std::cin >> rk4eps;
    for (unsigned i=1;i<2;i+=2) { 

        //Nzn = unsigned(Nz*pow(2,i));
        //Nxn = (unsigned)ceil(Nx*pow(2,(double)(i*2./n)));
        //Nyn = (unsigned)ceil(Ny*pow(2,(double)(i*2./n)));
        //solovev::ConformalRingGrid3d<dg::DVec> g3d(gp, psi_0, psi_1, n, Nxn, Nyn,Nzn, dg::DIR);
        //solovev::ConformalRingGrid2d<dg::DVec> g2d = g3d.perp_grid();
        solovev::OrthogonalRingGrid3d<dg::DVec> g3d(gp, psi_0, psi_1, n, Nxn, Nyn,Nzn, dg::DIR);
        solovev::OrthogonalRingGrid2d<dg::DVec> g2d = g3d.perp_grid();
        g3d.display();
        //g2d.display();
        std::cout << "NR = " << Nxn << std::endl;
        std::cout << "NZ = " << Nyn<< std::endl;
        std::cout << "Nphi = " << Nzn << std::endl;
        const dg::DVec vol = dg::create::volume( g3d);
        const dg::DVec ones = dg::pullback(dg::one, g3d);
        const dg::DVec v3d = dg::create::inv_volume( g3d);

        std::cout << "Computing ds..." << std::endl;
        //DFA dsFA( solovev::ConformalField( gp, g3d.x(), g3d.f_x()), g3d, rk4eps, dg::NoLimiter(), dg::DIR); 
        //dg::DS<DFA, dg::DMatrix, dg::DVec> ds( dsFA, solovev::ConformalField(gp, g3d.x(), g3d.f_x()), dg::not_normed, dg::centered, false);
        DFA dsFA( solovev::OrthogonalField( gp, g2d, g2d.g()), g3d, rk4eps, dg::NoLimiter(), dg::DIR); 
        dg::DS<DFA, dg::DMatrix, dg::DVec> ds( dsFA, solovev::OrthogonalField(gp, g2d, g2d.g()), dg::not_normed, dg::centered, false);

        std::cout << "ds constructed!" << std::endl;
        dg::DVec function = dg::pullback( funcNEU, g3d), derivative(function), temp(function),
                        dsTdsfb(function),
                        functionTinv2(dg::evaluate( dg::zero, g3d));

        double volume = dg::blas1::dot(vol, ones);
        std::cout << "|| volume   ||   "<<volume<<"\n";
        std::cout << "--------------------testing ds" << std::endl;
        const dg::DVec solution = dg::pullback( deriNEU, g3d);
        double norm = fabs(dg::blas2::dot( vol, solution));
        std::cout << "|| num. function ||   "<<sqrt( fabs(dg::blas2::dot( vol, function)) )<<"\n";
        std::cout << "|| ana. function ||   "<<sqrt( volume/2.  )<<"\n";
        std::cout << "|| Solution ||   "<<sqrt( norm)<<"\n";
        ds( function, derivative); //ds(f)
        if(norm == 0) norm =1;
        double err = fabs(dg::blas2::dot( vol, derivative));
        std::cout << "|| Derivative || "<<sqrt( err)<<"\n";
        dg::blas1::axpby( 1., solution, -1., derivative);
        err =dg::blas2::dot( derivative, vol, derivative);
        std::cout << "Relative Difference in DS is "<< sqrt( err/norm )<<"\n"; 
      
        const dg::DVec solutiondsTds = dg::pullback( deriNEUT2, g3d);
        const dg::DVec B = dg::pullback( solovev::Bmodule(gp), g3d); 
        double normdsTds = dg::blas2::dot( vol, solutiondsTds);
        std::cout << "--------------------testing dsTdsfb " << std::endl;
        std::cout << "|| SolutionT ||      "<<sqrt( normdsTds)<<"\n";
        //1st
        //ds( function, derivative); //ds(f)
        //ds( B, temp ); //ds(B)
        //ds( derivative, dsTdsfb); //dss(f)
        //dg::blas1::pointwiseDot(derivative, temp, derivative);
        //dg::blas1::pointwiseDivide(derivative, B, derivative);
        //dg::blas1::axpby( -1., derivative, 1., dsTdsfb, dsTdsfb);
        //2nd
        //ds.symv(function,dsTdsfb);
        //dg::blas1::pointwiseDot(v3d,dsTdsfb,dsTdsfb);
        //3rd
        ds( function, derivative);
        dg::blas1::pointwiseDivide( derivative, B, derivative);
        ds(derivative, dsTdsfb);
        dg::blas1::pointwiseDot( dsTdsfb, B, dsTdsfb);
        double remainder =dg::blas1::dot( vol,dsTdsfb);
        double errdsTdsfb =dg::blas2::dot( vol,dsTdsfb);
        std::cout << "|| DerivativeTds ||  "<<sqrt( errdsTdsfb)<<"\n";
        std::cout << "   Integral          "<<remainder<<"\n";
        dg::blas1::axpby( 1., solutiondsTds, -1., dsTdsfb);
        errdsTdsfb =dg::blas2::dot( vol, dsTdsfb);
        std::cout << "Relative Difference in DST is "<< sqrt( errdsTdsfb/normdsTds )<<"\n";
        
        
        double eps =1e-8;   
        dg::Invert< dg::DVec> invert( dg::evaluate(dg::zero,g3d), vol.size(), eps );  
        std::cout << "MAX # iterations = " << vol.size() << std::endl;
        double normf = dg::blas2::dot( vol, function);
        std::cout << "--------------------testing dsT" << std::endl; 
        std::cout << " # of iterations "<< invert( ds, functionTinv2,solutiondsTds ) << std::endl; //is dsTds
        std::cout << "Norm analytic Solution  "<<sqrt( normf)<<"\n";
        double errinvT2 =dg::blas2::dot( vol, functionTinv2);
        std::cout << "Norm numerical Solution "<<sqrt( errinvT2)<<"\n";
        dg::blas1::axpby( 1., function, -1.,functionTinv2);
        errinvT2 =dg::blas2::dot( vol, functionTinv2);
        std::cout << "Relative Difference is  "<< sqrt( errinvT2/normf )<<"\n";



        const dg::DVec lnB = dg::pullback( solovev::LnB(gp), g3d); 
        dg::DVec gradB(lnB);
        const dg::DVec gradLnB = dg::pullback( solovev::GradLnB(gp), g3d);
        std::cout << "ana. norm GradLnB    "<<sqrt( dg::blas2::dot( gradLnB, vol, gradLnB))<<"\n";
        ds( B, gradB);
        dg::blas1::pointwiseDivide(gradB, B, gradB);
        std::cout << "num. norm GradLnB    "<<sqrt( dg::blas2::dot( gradB, vol, gradB))<<"\n";
        dg::blas1::axpby( 1., gradB, -1., gradLnB, temp);
        std::cout << "ds     Error of lnB is    "<<sqrt( dg::blas2::dot( temp, vol, temp))<<"\n";


        dg::DMatrix dx = dg::create::dx( g3d);
        dg::DMatrix dy = dg::create::dy( g3d);
        dg::DMatrix dz = dg::create::dz( g3d);
        dg::HVec hbx(function), hby(function);
        dg::geo::pushForwardPerp( bhatR, bhatZ, hbx, hby, g3d);
        const dg::DVec bx(hbx), by(hby);
        dg::blas2::symv( dx, lnB, derivative);
        dg::blas2::symv( dy, lnB, temp);
        dg::blas1::pointwiseDot( bx,  derivative, derivative);
        dg::blas1::pointwiseDot( by,  temp, temp);
        dg::blas1::axpby( 1., derivative, +1., temp, temp);
        dg::blas1::axpby( 1., temp, -1., gradB, gradB);
        dg::blas1::axpby( 1., temp, -1., gradLnB, temp);
        std::cout << "direct Error of lnB is    "<<sqrt( dg::blas2::dot( temp, vol, temp))<<"\n";
        //std::cout << "Error of lnB is    "<<sqrt( dg::blas2::dot( gradB, vol, gradB))<<"\n";

        dg::DVec bmod = dg::pullback( solovev::Bmodule( gp), g3d);
        dg::geo::multiplyVolume( bmod, g3d);
        dg::blas1::pointwiseDot( bx,  bmod, derivative);
        dg::blas1::pointwiseDot( by,  bmod, temp);
        dg::blas2::symv( dx, derivative, bmod);
        dg::blas2::symv( dy, temp, derivative);
        dg::blas1::axpby( 1., derivative, +1., bmod, temp);
        dg::geo::divideVolume( temp, g3d);
        std::cout << "direct Error of divB is    "<<sqrt( dg::blas2::dot( temp, vol, temp))<<"\n";

    }
    
    return 0;
}
