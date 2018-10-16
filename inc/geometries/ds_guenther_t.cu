#include <iostream>

#include <cusp/print.h>
#include <cusp/csr_matrix.h>
#include "json/json.h"

#define DG_BENCHMARK
#include "dg/algorithm.h"
#include "ds.h"
// #include "draw/host_window.h"
#include "guenther.h"
#include "magnetic_field.h"
#include "testfunctors.h"


int main( )
{

    /////////////////initialize params////////////////////////////////
    Json::Value js;
    std::ifstream is("guenther_params.js");
    is>>js;
    dg::geo::guenther::Parameters gp(js);
    //////////////////////////////////////////////////////////////////////////
    double Rmin=gp.R_0-1.0*gp.a;
    double Zmin=-1.0*gp.a*gp.elongation;
    double Rmax=gp.R_0+1.0*gp.a;
    double Zmax=1.0*gp.a*gp.elongation;
    /////////////////////////////////////////////initialze fields /////////////////////
    dg::geo::TokamakMagneticField mag = dg::geo::createGuentherField(gp.R_0, gp.I_0);
    dg::geo::InvB invb(mag);
    dg::geo::GradLnB gradlnB(mag);
    dg::geo::Divb divb(mag);
    dg::geo::guenther::FuncNeu funcNEU(gp.R_0,gp.I_0);
    dg::geo::guenther::DeriNeu deriNEU(gp.R_0,gp.I_0);

    unsigned n=3;
    unsigned Nxn = 20;
    unsigned Nyn = 20;
    unsigned Nzn = 20;

    double rk4eps = 1e-8;
    double z0 = 0, z1 = 2.*M_PI;
    std::cout << "Type n, Nx, Ny, Nz\n";
    std::cin >> n >> Nxn >> Nyn >> Nzn;

    dg::CylindricalGrid3d g3d( Rmin,Rmax, Zmin,Zmax, z0, z1,  n,Nxn ,Nyn, Nzn,dg::DIR, dg::DIR, dg::PER);
    dg::Grid2d g2d( Rmin,Rmax, Zmin,Zmax,  n, Nxn ,Nyn);

    const dg::DVec w3d = dg::create::volume( g3d);

    std::cout << "Type multipleX (10) and multipleY (10)!\n";
    unsigned mx, my;
    std::cin >> mx >> my;

    std::cout << "computing dsDIR" << std::endl;
    dg::geo::Fieldaligned<dg::aProductGeometry3d, dg::IDMatrix, dg::DVec>  dsFA( mag, g3d, dg::DIR, dg::DIR, dg::geo::NoLimiter(), rk4eps, mx, my);

    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds ( dsFA, dg::not_normed, dg::centered);

    dg::DVec function = dg::evaluate( funcNEU, g3d) ,
                        temp( function),
                        derivative(function),
                        inverseB( dg::evaluate(invb, g3d)),
                        divbsol(dg::evaluate(divb, g3d)),
                        divbT(function),
                        divBT(function);


    dg::DVec ones = dg::evaluate( dg::one, g3d);
    const dg::DVec solution = dg::evaluate( deriNEU, g3d);

    const dg::DVec gradlnB_ = dg::evaluate(gradlnB, g3d);

    ds( function, derivative); //ds(f)
    dg::blas1::pointwiseDivide(ones,  inverseB, temp); //B
    ds.centeredDiv( temp, divBT); // dsT B
    ds.centeredDiv( ones, divbT);

    double norm = dg::blas2::dot( w3d, solution);
    double err =dg::blas2::dot( w3d, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    err =dg::blas2::dot( w3d, derivative);
    std::cout << "Relative Difference in ds f   = "<< sqrt( err/norm )<<"\n";

    norm = dg::blas2::dot( w3d, divbsol);
    err = dg::blas2::dot( w3d, divbT);
    dg::blas1::axpby( 1., divbsol, -1., divbT);
    err = dg::blas2::dot(divbT, w3d,divbT);
    std::cout << "Relative Difference in div(b) = "<<sqrt(err/norm)<<"\n";
    double normdivBT =dg::blas2::dot(divBT, w3d,divBT);
    std::cout << "Error in div(B)  = "<<sqrt( normdivBT)<<"\n";

    return 0;
}
