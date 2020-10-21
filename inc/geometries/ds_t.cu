#include <iostream>
#include <iomanip>

#define DG_BENCHMARK
#include "dg/algorithm.h"
#include "magnetic_field.h"
#include "testfunctors.h"
#include "ds.h"
#include "toroidal.h"

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main(int argc, char * argv[])
{
    std::cout << "# Test the parallel derivative DS in cylindrical coordinates for circular flux surfaces with DIR and NEU boundary conditions.\n";
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

    //![doxygen]
    const dg::CylindricalGrid3d g3d( R_0-a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER);
    //create magnetic field
    const dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    const dg::geo::CylindricalVectorLvl0 bhat = dg::geo::createBHat(mag);
    //create Fieldaligned object and construct DS from it
    dg::geo::Fieldaligned<dg::aProductGeometry3d,dg::IDMatrix,dg::DVec>  dsFA( bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx, my);
    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds( dsFA, dg::centered, dg::geo::boundary::along_field );
    //![doxygen]
    ///##########################################################///
    const dg::DVec fun = dg::pullback( dg::geo::TestFunctionDirNeu(mag), g3d);
    dg::DVec derivative(fun);
    const dg::DVec divb = dg::pullback( dg::geo::Divb(mag), g3d);
    const dg::DVec sol0 = dg::pullback( dg::geo::DsFunction<dg::geo::TestFunctionDirNeu>(mag), g3d);
    const dg::DVec sol1 = dg::pullback( dg::geo::DssFunction<dg::geo::TestFunctionDirNeu>(mag), g3d);
    const dg::DVec sol2 = dg::pullback( dg::geo::DsDivFunction<dg::geo::TestFunctionDirNeu>(mag), g3d);
    const dg::DVec sol3 = dg::pullback( dg::geo::DsDivDsFunction<dg::geo::TestFunctionDirNeu>(mag), g3d);
    const dg::DVec sol4 = dg::pullback( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionDirNeu>(mag), g3d);
    std::vector<std::pair<std::string, std::array<const dg::DVec*,2>>> names{
         {"forward",{&fun,&sol0}},          {"backward",{&fun,&sol0}},
         {"centered",{&fun,&sol0}},         {"dss",{&fun,&sol1}},
         {"divForward",{&fun,&sol2}},       {"divBackward",{&fun,&sol2}},
         {"divCentered",{&fun,&sol2}},      {"divDirectForward",{&fun,&sol2}},
         {"divDirectBackward",{&fun,&sol2}},{"divDirectCentered",{&fun,&sol2}},
         {"forwardLap",{&fun,&sol3}},       {"backwardLap",{&fun,&sol3}},
         {"centeredLap",{&fun,&sol3}},      {"directLap",{&fun,&sol3}},
         {"invForwardLap",{&sol4,&fun}},    {"invBackwardLap",{&sol4,&fun}},
         {"invCenteredLap",{&sol4,&fun}}
    };
    std::cout << "# TEST NEU Boundary conditions!\n";
    std::cout << "# TEST ADJOINT derivatives do unfortunately not fulfill Neumann BC!\n";
    ///##########################################################///
    std::cout <<"Neumann:\n";
    dg::DVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::DVec& function = *std::get<1>(tuple)[0];
        const dg::DVec& solution = *std::get<1>(tuple)[1];
        callDS( ds, name, function, derivative, divb, max_iter,1e-8);
        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    std::cout << "# Reconstruct parallel derivative!\n";
    dsFA.construct( bhat, g3d, dg::DIR, dg::DIR, dg::geo::NoLimiter(), 1e-8, mx, my);
    ds.construct( dsFA, dg::centered, dg::geo::boundary::along_field);
    std::cout << "# TEST DIR Boundary conditions!\n";
    ///##########################################################///
    std::cout << "Dirichlet: \n";
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::DVec& function = *std::get<1>(tuple)[0];
        const dg::DVec& solution = *std::get<1>(tuple)[1];
        callDS( ds, name, function, derivative, divb, max_iter,1e-8);
        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }

    ///##########################################################///
    std::cout << "# TEST FIELDALIGNED EVALUATION of a Gaussian\n";
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    dg::DVec aligned = dsFA.evaluate( init0, modulate, Nz/2, 2);
    ds( aligned, derivative);
    double norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "# Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpit)\n";

    return 0;
}
