#include <iostream>
#include <iomanip>

#define DG_BENCHMARK
#include "dg/algorithm.h"
#include "ds.h"
#include "guenther.h"
#include "magnetic_field.h"
#include "testfunctors.h"

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main( )
{
    std::cout << "# Test the parallel derivative DS in cylindrical coordinates for the guenther flux surfaces. Fieldlines do not cross boundaries.\n";
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
        dg::centered, 1e-8, mx, my);

    ///##########################################################///
    const dg::DVec fun = dg::evaluate( dg::geo::TestFunctionPsi2(mag), g3d);
    const dg::DVec divb = dg::evaluate( dg::geo::Divb(mag), g3d);
    dg::DVec derivative(fun);
    dg::DVec sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::DVec sol1 = dg::evaluate( dg::geo::DssFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::DVec sol2 = dg::evaluate( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::DVec sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::DVec sol4 = dg::evaluate( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    std::vector<std::tuple<std::string, const dg::DVec&, const dg::DVec&>> names{
         {"forward",fun,sol0},          {"backward",fun,sol0},
         {"centered",fun,sol0},         {"dss",fun,sol1},
         {"divForward",fun,sol2},       {"divBackward",fun,sol2},
         {"divCentered",fun,sol2},      {"divDirectForward",fun,sol2},
         {"divDirectBackward",fun,sol2},{"divDirectCentered",fun,sol2},
         {"forwardLap",fun,sol3},       {"backwardLap",fun,sol3},
         {"centeredLap",fun,sol3},      {"directLap",fun,sol3},
         {"invForwardLap",sol4,fun},    {"invBackwardLap",sol4,fun},
         {"invCenteredLap",sol4,fun}
    };

    ///##########################################################///
    std::cout << "# TEST Guenther (No Boundary conditions)!\n";
    std::cout <<"Guenther:\n";
    const dg::DVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::DVec& function = std::get<1>(tuple);
        const dg::DVec& solution = std::get<2>(tuple);
        callDS( ds, name, function, derivative, divb, max_iter,1e-8);
        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    return 0;
}
