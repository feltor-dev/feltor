#include <iostream>

#include <cusp/print.h>
#include "json/json.h"

#include "dg/algorithm.h"
#include "ds.h"
#include "solovev.h"
#include "flux.h"
#include "guenter.h"
#include "toroidal.h"
#include "testfunctors.h"


int main(int argc, char * argv[])
{
    std::cout << "# Test DS on flux grid (No Boundary conditions)!\n";
    Json::Value js;
    //std::stringstream ss;
    //ss << "{"
    //   << "    \"A\" : 0.0,"
    //   << "    \"PP\": 1,"
    //   << "    \"PI\": 1,"
    //   << "    \"c\" :[  0.07350114445500399706283007092406934834526,"
    //   << "           -0.08662417436317227513877947632069712210813,"
    //   << "           -0.1463931543401102620740934776490506239925,"
    //   << "           -0.07631237100536276213126232216649739043965,"
    //   << "            0.09031790113794227394476271394334515457567,"
    //   << "           -0.09157541239018724584036670247895160625891,"
    //   << "           -0.003892282979837564486424586266476650443202,"
    //   << "            0.04271891225076417603805495295590637082745,"
    //   << "            0.2275545646002791311716859154040182853650,"
    //   << "           -0.1304724136017769544849838714185080700328,"
    //   << "           -0.03006974108476955225335835678782676287818,"
    //   << "            0.004212671892103931178531621087067962015783 ],"
    //   << "    \"R_0\"                : 547.891714877869,"
    //   << "    \"inverseaspectratio\" : 0.41071428571428575,"
    //   << "    \"elongation\"         : 1.75,"
    //   << "    \"triangularity\"      : 0.47,"
    //   << "    \"equilibrium\"  : \"solovev\","
    //   << "    \"description\" : \"standardX\""
    //   << "}";
    //ss >> js;
    //dg::geo::solovev::Parameters gp(js);
    //dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);
    //double psi_0 = -20, psi_1 = -4;
    //const double R_0 = gp.R_0;
    //const double a = gp.a;
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
    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds( mag, g3d, dg::NEU, dg::PER, dg::geo::FullLimiter(), 1e-8, mx, my, -1, method);

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
