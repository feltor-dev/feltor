#include <iostream>

#include <cusp/print.h>
#include "dg/algorithm.h"
#include "magnetic_field.h"
#include "testfunctors.h"
#define DG_BENCHMARK
#include "ds.h"
#include "toroidal.h"


const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main(int argc, char * argv[])
{
    std::cout << "# This program tests the parallel derivative DS in cylindrical coordinates for circular flux surfaces with DIR and NEU boundary conditions.\n";
    std::cout << "# Type n (3), Nx(20), Ny(20), Nz(20)\n";
    unsigned n, Nx, Ny, Nz, mx, my, max_iter;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout <<"# You typed\n"
              <<"n: "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<std::endl;
    std::cout << "# Type mx (10) and my (10)\n";
    std::cin >> mx>> my;
    std::cout << "# You typed\n"
              <<"mx: "<<mx<<"\n"
              <<"my: "<<my<<std::endl;
    std::cout << "# Type max iterations (1000) \n";
    std::cin >> max_iter;
    std::cout << "# You typed\n"
              <<"max_iter: "<<max_iter<<"\n";
    std::cout << "# Create parallel Derivative!\n";

    //![doxygen]
    const dg::CylindricalGrid3d g3d( R_0 - a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER);
    //create magnetic field
    const dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    const dg::geo::BinaryVectorLvl0 bhat( (dg::geo::BHatR)(mag), (dg::geo::BHatZ)(mag), (dg::geo::BHatP)(mag));
    //create Fieldaligned object and construct DS from it
    dg::geo::Fieldaligned<dg::aProductGeometry3d,dg::IDMatrix,dg::DVec>  dsFA( bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx, my);
    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds( dsFA, dg::normed, dg::centered);
    //![doxygen]
    ///##########################################################///
    //apply to function
    const dg::DVec function = dg::evaluate( dg::geo::FunctionSinNEU(mag), g3d);
    dg::DVec derivative(function);
    dg::DVec sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::FunctionSinNEU>(mag), g3d);
    dg::DVec sol1 = dg::evaluate( dg::geo::DssFunction<dg::geo::FunctionSinNEU>(mag), g3d);
    dg::DVec sol2 = dg::evaluate( dg::geo::DsDivFunction<dg::geo::FunctionSinNEU>(mag), g3d);
    dg::DVec sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::FunctionSinNEU>(mag), g3d);
    std::vector<std::pair<std::string, const dg::DVec&>> names{
         {"forward",sol0}, {"backward",sol0},
         {"centered",sol0}, {"dss",sol1},
         {"forwardDiv",sol2}, {"backwardDiv",sol2}, {"centeredDiv",sol2},
         {"forwardLap",sol3}, {"backwardLap",sol3}, {"centeredLap",sol3}
    };
    std::cout << "# TEST NEU Boundary conditions!\n";
    std::cout << "# TEST ADJOINT derivatives do unfortunately not fulfill Neumann BC!\n";
    ///##########################################################///
    std::cout <<"Neumann:\n";
    const dg::DVec vol3d = dg::create::volume( g3d);
    for( const auto& name :  names)
    {
        callDS( ds, name.first, function, derivative);
        double sol = dg::blas2::dot( vol3d, name.second);
        dg::blas1::axpby( 1., name.second, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::cout <<"    "<<name.first<<":"
                  <<std::setw(16-name.first.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    dg::DVec solution = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::FunctionSinNEU>(mag), g3d);
    ds.set_direction( dg::forward);
    ds.set_norm( dg::not_normed);
    dg::geo::DSS< dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec>, dg::DVec> dss( ds);
    dg::Invert<dg::DVec> invert( solution, max_iter, 1e-5);
    invert( dss, derivative, solution);

    double sol = dg::blas2::dot( vol3d, function);
    dg::blas1::axpby( 1., function, 1., derivative);
    double norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "    invForwardLap:   "<< sqrt( norm/sol )<<"\n";

    ///##########################################################///
    std::cout << "# Reconstruct parallel derivative!\n";
    dsFA.construct( bhat, g3d, dg::DIR, dg::DIR, dg::geo::NoLimiter(), 1e-8, mx, my);
    ds.construct( dsFA, dg::normed, dg::centered);
    const dg::DVec functionDIR = dg::evaluate( dg::geo::FunctionSinDIR(mag), g3d);
    sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::FunctionSinDIR>(mag), g3d);
    sol1 = dg::evaluate( dg::geo::DssFunction<dg::geo::FunctionSinDIR>(mag), g3d);
    sol2 = dg::evaluate( dg::geo::DsDivFunction<dg::geo::FunctionSinDIR>(mag), g3d);
    sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::FunctionSinDIR>(mag), g3d);
    std::cout << "# TEST DIR Boundary conditions!\n";
    ///##########################################################///
    std::cout << "Dirichlet: \n";
    for( const auto& name :  names)
    {
        callDS( ds, name.first, functionDIR, derivative);
        double sol = dg::blas2::dot( vol3d, name.second);
        dg::blas1::axpby( 1., name.second, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::cout <<"    "<<name.first<<":"
                  <<std::setw(16-name.first.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    solution = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::FunctionSinDIR>(mag), g3d);
    ds.set_direction( dg::forward);
    ds.set_norm( dg::not_normed);
    dg::Invert<dg::DVec> invertDIR( solution, max_iter, 1e-5);
    invertDIR( dss, derivative, solution);
    sol = dg::blas2::dot( vol3d, functionDIR);

    ////ds.symv( functionDIR, derivative);
    dg::blas1::axpby( 1., functionDIR, 1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "    invForwardLap:   "<< sqrt( norm/sol )<<"\n";

    ///##########################################################///
    std::cout << "# TEST FIELDALIGNED EVALUATION of a Gaussian\n";
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    dg::DVec aligned = dsFA.evaluate( init0, modulate, Nz/2, 2);
    ds( aligned, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "# Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpit)\n";

    return 0;
}
