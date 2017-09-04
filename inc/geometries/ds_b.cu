#include <iostream>

#include <cusp/print.h>

#include "blas.h"
#include "ds.h"
#include "functors.h"
#include "solovev.h"
#include "flux.h"
#include "geometry.h"

#include "backend/functions.h"
#include "backend/timer.cuh"

double func(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return r2*sin(phi);
}
double deri(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return I_0/R/sqrt(I_0*I_0 + r2)* r2*cos(phi);
}

int main()
{
    std::cout << "First test the cylindrical version\n";
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "You typed "<<n<<" "<<Nx<<" "<<Ny<<" "<<Nz<<std::endl;
    Json::Reader reader;
    Json::Value js;
    if( argc==1) {
        std::ifstream is("geometry_params_Xpoint.js");
        reader.parse(is,js,false);
    }
    else {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    dg::geo::solovev::GeomParameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);
    dg::geo::BHatR bhatR(mag);
    dg::geo::BHatZ bhatZ(mag);
    dg::geo::BHatP bhatP(mag);
    dg::geo::BinaryVectorLvl0 bhat( bhatR, bhatZ, bhatP);
    dg::CylindricalGrid3d g3d( gp.R_0 - 1, gp.R_0+1, -1, 1, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER);
    const dg::DVec vol3d = dg::create::volume( g3d);
    dg::Timer t;
    t.tic();
    dg::geo::BinaryVectorLvl0 bhat( bhatR, bhatZ, bhatP);
    dg::FieldAligned dsFA( bhat, g3d, 1e-10, dg::DefaultLimiter(), dg::NEU);

    dg::DS<dg::aGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds ( dsFA, g3d, field, dg::not_normed, dg::centered);
    t.toc();
    std::cout << "Creation of parallel Derivative took     "<<t.diff()<<"s\n";

    dg::DVec function = dg::evaluate( func, g3d), derivative(function);
    const dg::DVec solution = dg::evaluate( deri, g3d);
    t.tic();
    ds( function, derivative);
    t.toc();
    std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    double norm = dg::blas2::dot( vol3d, solution);
    std::cout << "Norm Solution "<<sqrt( norm)<<"\n";
    std::cout << "Relative Difference Is "<< sqrt( dg::blas2::dot( derivative, vol3d, derivative)/norm )<<"\n";
    std::cout << "Error is from the parallel derivative only if n>2\n"; //since the function is a parabola
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    t.tic();
    function = ds.fieldaligned().evaluate( init0, modulate, Nz/2, 2);
    t.toc();
    std::cout << "Fieldaligned initialization took "<<t.diff()<<"s\n";
    ds( function, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpib)\n";
    ds.forward( function, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Norm Forward  Derivative "<<sqrt( norm)<<" (compare with that of ds_mpib)\n";
    ds.backward( function, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Norm Backward Derivative "<<sqrt( norm)<<" (compare with that of ds_mpib)\n";

    /////////////////////////TEST 3d flux grid//////////////////////////////////////
    std::cout << "Start DS test on flux grid!"<<std::endl;
    t.tic();
    //dg::geo::BinaryVectorLvl0 bhat( dg::geo::BHatR(c), dg::geo::BHatZ(c), dg::geo::BHatP(c));
    unsigned mx, my;
    std::cout << "Type multipleX and multipleY!\n";
    std::cin >> mx >> my;

    dg::geo::FluxGenerator flux( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    dg::CurvilinearProductGrid3d g3d(flux, n, Nx, Ny,Nz, dg::DIR);
    //dg::FieldAligned<dg::aGeometry3d, dg::IHMatrix, dg::HVec> fieldaligned( bhat, g3d, 1, 4, gp.rk4eps, dg::NoLimiter() ); 
    dg::DS<dg::aGeometry3d, dg::IHMatrix, dg::HMatrix, dg::HVec> ds( mag, g3d, dg::normed, dg::centered, false, true, mx, my);

    
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    dg::HVec B = dg::pullback( dg::geo::InvB(mag), g3d), divB(B);
    dg::HVec lnB = dg::pullback( dg::geo::LnB(mag), g3d), gradB(B);
    dg::HVec gradLnB = dg::pullback( dg::geo::GradLnB(mag), g3d);
    dg::blas1::pointwiseDivide( ones3d, B, B);
    dg::HVec function = dg::pullback( dg::geo::FuncNeu(mag), g3d), derivative(function);
    ds( function, derivative);

    ds.centeredAdj( B, divB);
    double norm =  sqrt( dg::blas2::dot(divB, vol3d, divB));
    std::cout << "Divergence of B is "<<norm<<"\n";

    ds.centered( lnB, gradB);
    std::cout << "num. norm of gradLnB is "<<sqrt( dg::blas2::dot( gradB,vol3d, gradB))<<"\n";
    norm = sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB) );
    std::cout << "ana. norm of gradLnB is "<<norm<<"\n";
    dg::blas1::axpby( 1., gradB, -1., gradLnB, gradLnB);
    X = divB;
    err = nc_put_var_double( ncid, varID[4], periodify(X, g2d_periodic).data());
    double norm2 = sqrt(dg::blas2::dot(gradLnB, vol3d, gradLnB));
    std::cout << "rel. error of lnB is    "<<norm2/norm<<"\n";
    
    return 0;
}
