#include <iostream>

#include <cusp/print.h>

#include "dg/algorithm.h"
#include "ds.h"
#include "flux.h"
#include "guenter.h"
#include "toroidal.h"
#include "testfunctors.h"


int main(int argc, char * argv[])
{
    std::cout << "# Test DS on flux grid (No Boundary conditions)!\n";
    const double R_0 = 3;
    const double I_0 = 10; //q factor at r=1 is I_0/R_0
    const double a  = 1; //small radius
    const dg::geo::TokamakMagneticField mag = dg::geo::createGuenterField(R_0, I_0);
    double psi_0 = 0.8, psi_1 = 0.2;
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
    std::string method = "cubic";
    std::cout << "# Type method (dg, nearest, linear, cubic) \n";
    std::cin >> method;
    method.erase( std::remove( method.begin(), method.end(), '"'), method.end());
    std::cout << "# You typed\n"
              <<"method: "<< method<<std::endl;

    dg::Timer t;
    t.tic();
    dg::geo::FluxGenerator flux( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, R_0, 0., 1);
    std::cout << "# Constructing Grid..."<<std::endl;
    dg::geo::CurvilinearProductGrid3d g3d(flux, n, Nx, Ny,Nz, dg::NEU);
    std::cout << "# Constructing Fieldlines..."<<std::endl;
    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DVec> ds( mag, g3d, dg::NEU, dg::PER, dg::geo::FullLimiter(), 1e-8, mx, my, -1, method);

    t.toc();
    std::cout << "# Construction took "<<t.diff()<<"s\n";
    ///##########################################################///
    //(MIND THE PULLBACK!)
    auto ff = dg::geo::TestFunctionPsi2(mag,a);
    const dg::DVec fun = dg::pullback( ff, g3d);
    dg::DVec derivative(fun);
    dg::DVec sol0 = dg::pullback( dg::geo::DsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::DVec sol1 = dg::pullback( dg::geo::DssFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::DVec sol2 = dg::pullback( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::DVec sol3 = dg::pullback( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::DVec sol4 = dg::pullback( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    std::vector<std::pair<std::string, std::array<const dg::DVec*,2>>> names{
         {"forward",{&fun,&sol0}},          {"backward",{&fun,&sol0}},
         {"centered",{&fun,&sol0}},         {"dss",{&fun,&sol1}},
         {"divForward",{&fun,&sol2}},       {"divBackward",{&fun,&sol2}},
         {"divCentered",{&fun,&sol2}},      {"directLap",{&fun,&sol3}}//,
         //{"invCenteredLap",{&sol4,&fun}}
    };
    ///##########################################################///
    std::cout <<"Flux:\n";
    const dg::DVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::DVec& function = *std::get<1>(tuple)[0];
        const dg::DVec& solution = *std::get<1>(tuple)[1];
        callDS( ds, name, function, derivative, g3d.size(),1e-8);
        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        double vol = dg::blas1::dot( vol3d, derivative)/sqrt( dg::blas2::dot( vol3d, function)); // using function in denominator makes entries comparable
        std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<std::endl
                  <<"    "<<name+"_vol:"<<std::setw(30-name.size())
                  <<" "<<vol<<"\n";
    }
    ///##########################################################///
    std::cout << "# TEST VOLUME FORMS\n";
    double volume = dg::blas1::dot( 1., ds.fieldaligned().sqrtG());
    double volumeM = dg::blas1::dot( 1., ds.fieldaligned().sqrtGm());
    double volumeP = dg::blas1::dot( 1., ds.fieldaligned().sqrtGp());
    // does error in volume form indicate a bug somewhere?
    std::cout << "volume_error:\n";
    std::cout <<"    minus:"<<std::setw(13)<<" "<<fabs(volumeM-volume)/volume<<"\n";
    std::cout <<"    plus:" <<std::setw(14)<<" "<<fabs(volumeP-volume)/volume<<"\n";
    return 0;
}
