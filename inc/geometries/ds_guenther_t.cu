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

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main( )
{
    std::cout << "# This program tests the parallel derivative DS in cylindrical coordinates for the guenther flux surfaces. Fieldlines do not cross boundaries.\n";
    std::cout << "# Type n (3), Nx(20), Ny(20), Nz(20)\n";
    unsigned n, Nx, Ny, Nz, mx, my, max_iter = 1e4;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<std::endl;
    std::cout << "# Type mx (10) and my (10)\n";
    std::cin >> mx>> my;
    std::cout << "# You typed\n"
              <<"mx: "<<mx<<"\n"
              <<"my: "<<my<<std::endl;
    std::cout << "# Create parallel Derivative!\n";
    ////////////////////////////////initialze fields /////////////////////
    const dg::CylindricalGrid3d g3d( R_0 - a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER);
    const dg::geo::TokamakMagneticField mag = dg::geo::createGuentherField(R_0, I_0);
    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds(
        mag, g3d, dg::NEU, dg::NEU, dg::geo::FullLimiter(),
        dg::normed, dg::centered, 1e-8, mx, my);

    ///##########################################################///
    //apply to function
    const dg::DVec function = dg::evaluate( dg::geo::TestFunctionPsi(mag), g3d);
    dg::DVec derivative(function);
    dg::DVec sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::TestFunctionPsi>(mag), g3d);
    dg::DVec sol1 = dg::evaluate( dg::geo::DssFunction<dg::geo::TestFunctionPsi>(mag), g3d);
    dg::DVec sol2 = dg::evaluate( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi>(mag), g3d);
    dg::DVec sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi>(mag), g3d);
    std::vector<std::pair<std::string, const dg::DVec&>> names{
         {"forward",sol0}, {"backward",sol0},
         {"centered",sol0}, {"dss",sol1},
         {"forwardDiv",sol2}, {"backwardDiv",sol2}, {"centeredDiv",sol2},
         {"forwardLap",sol3}, {"backwardLap",sol3}, {"centeredLap",sol3}
    };
    ///##########################################################///
    std::cout << "# TEST Guenther (No Boundary conditions)!\n";
    std::cout <<"Guenther:\n";
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
    std::vector<std::pair<std::string, dg::direction>> namesLap{
         {"invForwardLap",dg::forward}, {"invBackwardLap",dg::backward}, {"invCenteredLap",dg::centered}
    };
    dg::DVec solution = dg::evaluate( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi>(mag), g3d);
    dg::Invert<dg::DVec> invert( solution, max_iter, 1e-10);
    dg::geo::InvertDS< dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec>, dg::DVec>
        rhs(ds);
    ds.set_norm( dg::normed);
    for( auto name : namesLap)
    {
        ds.set_direction( name.second);
        invert( rhs, derivative, solution);
        dg::blas1::axpby( 1., function, -1., derivative);
        double sol = dg::blas2::dot( vol3d, function);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::cout <<"    "<<name.first<<":"
                  <<std::setw(16-name.first.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    return 0;
}
