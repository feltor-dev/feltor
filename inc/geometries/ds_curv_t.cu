#include <iostream>

#include <cusp/print.h>
#include "json/json.h"

#include "dg/geometry/functions.h"
#include "dg/backend/timer.h"
#include "dg/blas.h"
#include "dg/functors.h"
#include "dg/geometry/geometry.h"
#include "testfunctors.h"
#include "ds.h"
#include "solovev.h"
#include "flux.h"
#include "toroidal.h"


int main(int argc, char * argv[])
{
    std::cout << "# Start DS test on flux grid!"<<std::endl;
    Json::Value js;
    if( argc==1) {
        std::ifstream is("geometry_params_Xpoint.js");
        is >> js;
    }
    else {
        std::ifstream is(argv[1]);
        is >> js;
    }
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);
    std::cout << "# Type n(3), Nx(8), Ny(80), Nz(20)\n";
    unsigned n,Nx,Ny,Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "# Type multipleX (1) and multipleY (100)!\n";
    unsigned mx, my;
    std::cin >> mx >> my;
    std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<"\n"
              <<"mx: "<<mx<<"\n"
              <<"my: "<<my<<std::endl;

    double psi_0 = -20, psi_1 = -4;
    dg::Timer t;
    t.tic();
    dg::geo::FluxGenerator flux( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    std::cout << "# Constructing Grid..."<<std::endl;
    dg::geo::CurvilinearProductGrid3d g3d(flux, n, Nx, Ny,Nz, dg::NEU);
    //dg::geo::Fieldaligned<dg::aGeometry3d, dg::IDMatrix, dg::DVec> fieldaligned( bhat, g3d, 1, 4, gp.rk4eps, dg::NoLimiter() );
    std::cout << "# Constructing Fieldlines..."<<std::endl;
    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds( mag, g3d, dg::NEU, dg::PER, dg::geo::FullLimiter(), dg::centered, 1e-8, mx, my);

    t.toc();
    std::cout << "# Construction took "<<t.diff()<<"s\n";
    ///##########################################################///
    //apply to function (MIND THE PULLBACK!)
    const dg::DVec function = dg::pullback( dg::geo::TestFunctionPsi(mag), g3d);
    dg::DVec derivative(function);
    dg::DVec sol0 = dg::pullback( dg::geo::DsFunction<dg::geo::TestFunctionPsi>(mag), g3d);
    dg::DVec sol1 = dg::pullback( dg::geo::DssFunction<dg::geo::TestFunctionPsi>(mag), g3d);
    dg::DVec sol2 = dg::pullback( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi>(mag), g3d);
    dg::DVec sol3 = dg::pullback( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi>(mag), g3d);
    std::vector<std::pair<std::string, const dg::DVec&>> names{
         {"forward",sol0}, {"backward",sol0},
         {"centered",sol0}, {"dss",sol1},
         {"forwardDiv",sol2}, {"backwardDiv",sol2}, {"centeredDiv",sol2},
         {"forwardLap",sol3}, {"backwardLap",sol3}, {"centeredLap",sol3}
    };
    ///##########################################################///
    std::cout << "# TEST Flux (No Boundary conditions)!\n";
    std::cout <<"Flux:\n";
    const dg::DVec vol3d = dg::create::volume( g3d);
    for( const auto& name :  names)
    {
        callDS( ds, name.first, function, derivative);
        dg::blas1::axpby( 1., name.second, -1., derivative);
        double sol = dg::blas2::dot( vol3d, name.second);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::cout <<"    "<<name.first<<":"
                  <<std::setw(16-name.first.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    std::vector<std::pair<std::string, dg::direction>> namesLap{
         {"invForwardLap",dg::forward}, {"invBackwardLap",dg::backward}, {"invCenteredLap",dg::centered}
    };
    dg::DVec solution = dg::pullback( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi>(mag), g3d);
    dg::Invert<dg::DVec> invert( solution, g3d.size(), 1e-10);
    dg::geo::TestInvertDS< dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec>, dg::DVec>
        rhs(ds);
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
