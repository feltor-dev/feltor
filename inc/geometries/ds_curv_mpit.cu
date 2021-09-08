#include <iostream>
#include <iomanip>
#include "json/json.h"

#include "mpi.h"
#undef DG_DEBUG
#include "dg/algorithm.h"
#include "testfunctors.h"
#include "ds.h"
#include "solovev.h"
#include "flux.h"
#include "toroidal.h"
#include "mpi_curvilinear.h"


int main(int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz, mx[2];
    MPI_Comm comm;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)std::cout << "# Test DS on flux grid (No Boundary conditions)!\n";
    dg::mpi_init3d( dg::DIR, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    if( rank == 0)
    {
        std::cout <<"# You typed\n"
                  <<"n:  "<<n<<"\n"
                  <<"Nx: "<<Nx<<"\n"
                  <<"Ny: "<<Ny<<"\n"
                  <<"Nz: "<<Nz<<std::endl;
        std::cout <<"# Type mx (1) and my (100)\n";
        std::cin >> mx[0] >> mx[1];
        std::cout << "# You typed\n"
                  <<"mx: "<<mx[0]<<"\n"
                  <<"my: "<<mx[1]<<std::endl;
        std::cout << "# Create parallel Derivative!\n";
    }
    MPI_Bcast( mx, 2, MPI_INT, 0, MPI_COMM_WORLD);
    Json::Value js;
    std::stringstream ss;
    ss << "{"
       << "    \"A\" : 0.0,"
       << "    \"PP\": 1,"
       << "    \"PI\": 1,"
       << "    \"c\" :[  0.07350114445500399706283007092406934834526,"
       << "           -0.08662417436317227513877947632069712210813,"
       << "           -0.1463931543401102620740934776490506239925,"
       << "           -0.07631237100536276213126232216649739043965,"
       << "            0.09031790113794227394476271394334515457567,"
       << "           -0.09157541239018724584036670247895160625891,"
       << "           -0.003892282979837564486424586266476650443202,"
       << "            0.04271891225076417603805495295590637082745,"
       << "            0.2275545646002791311716859154040182853650,"
       << "           -0.1304724136017769544849838714185080700328,"
       << "           -0.03006974108476955225335835678782676287818,"
       << "            0.004212671892103931178531621087067962015783 ],"
       << "    \"R_0\"                : 547.891714877869,"
       << "    \"inverseaspectratio\" : 0.41071428571428575,"
       << "    \"elongation\"         : 1.75,"
       << "    \"triangularity\"      : 0.47,"
       << "    \"equilibrium\"  : \"solovev\","
       << "    \"description\" : \"standardX\""
       << "}";
    ss >> js;
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);
    double psi_0 = -20, psi_1 = -4;
    dg::Timer t;
    t.tic();
    dg::geo::FluxGenerator flux( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    if(rank==0)std::cout << "# Constructing Grid..."<<std::endl;
    dg::geo::CurvilinearProductMPIGrid3d g3d(flux, n, Nx, Ny,Nz, dg::NEU, dg::PER, dg::PER, comm);
    if(rank==0)std::cout << "# Constructing Fieldlines..."<<std::endl;
    dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIDMatrix, dg::MDMatrix,
        dg::MDVec> ds( mag, g3d, dg::NEU, dg::PER, dg::geo::FullLimiter(),
                1e-8, mx[0], mx[1]);

    t.toc();
    if(rank==0)std::cout << "# Construction took "<<t.diff()<<"s\n";
    ///##########################################################///
    //(MIND THE PULLBACK!)
    const dg::MDVec fun = dg::pullback( dg::geo::TestFunctionPsi2(mag), g3d);
    dg::MDVec derivative(fun);
    dg::MDVec sol0 = dg::pullback( dg::geo::DsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol1 = dg::pullback( dg::geo::DssFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol2 = dg::pullback( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol3 = dg::pullback( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol4 = dg::pullback( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    std::vector<std::pair<std::string, std::array<const dg::MDVec*,2>>> names{
         {"forward",{&fun,&sol0}},          {"backward",{&fun,&sol0}},
         {"centered",{&fun,&sol0}},         {"dss",{&fun,&sol1}},
         {"divForward",{&fun,&sol2}},       {"divBackward",{&fun,&sol2}},
         {"divCentered",{&fun,&sol2}},      {"directLap",{&fun,&sol3}},
         {"invCenteredLap",{&sol4,&fun}}
    };
    ///##########################################################///
    if(rank==0)std::cout <<"Flux:\n";
    const dg::MDVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::MDVec& function = *std::get<1>(tuple)[0];
        const dg::MDVec& solution = *std::get<1>(tuple)[1];
        callDS( ds, name, function, derivative, g3d.size(),1e-8);
        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        double vol = dg::blas1::dot( vol3d, derivative)/sqrt( dg::blas2::dot( vol3d, function)); // using function in denominator makes entries comparable
        if(rank==0)std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n"
                  <<"    "<<name+"_vol:"<<std::setw(30-name.size())
                  <<" "<<vol<<"\n";
    }
    ///##########################################################///
    MPI_Finalize();
    return 0;
}
