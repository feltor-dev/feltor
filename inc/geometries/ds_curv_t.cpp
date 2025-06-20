#include <iostream>
#include <iomanip>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "dg/algorithm.h"
#include "ds.h"
#include "flux.h"
#include "guenter.h"
#include "toroidal.h"
#include "testfunctors.h"
#ifdef WITH_MPI
#include "mpi_curvilinear.h"
#endif


int main(
#ifdef WITH_MPI
    int argc, char * argv[]
#endif
)
{
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    DG_RANK0 std::cout << "# Test DS on flux grid (No Boundary conditions)!\n";
    const double R_0 = 3;
    const double I_0 = 10; //q factor at r=1 is I_0/R_0
    const double a  = 1; //small radius
    const dg::geo::TokamakMagneticField mag = dg::geo::createGuenterField(R_0, I_0);
    double psi_0 = 0.8, psi_1 = 0.2;
    unsigned n,Nx,Ny,Nz;
    unsigned mx, my;
    std::string method = "cubic";
#ifdef WITH_MPI
    MPI_Comm comm;
    dg::mpi_init3d( dg::DIR, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    unsigned letters = 0;
    if( rank == 0)
    {
        std::cout << "# Type multipleX (1) and multipleY (100)!\n";
        std::cin >> mx >> my;
        std::cout << "# Type method (dg, nearest, linear, cubic) \n";
        std::cin >> method;
        method.erase( std::remove( method.begin(), method.end(), '"'), method.end());
        letters = method.size();
    }
    MPI_Bcast( &mx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &my, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &letters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    method.resize( letters);
    MPI_Bcast( &method[0], letters, MPI_CHAR, 0, MPI_COMM_WORLD);
#else
    std::cout << "# Type n(3), Nx(8), Ny(80), Nz(20)\n";
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "# Type multipleX (1) and multipleY (100)!\n";
    std::cin >> mx >> my;
    std::cout << "# Type method (dg, nearest, linear, cubic) \n";
    std::cin >> method;
    method.erase( std::remove( method.begin(), method.end(), '"'), method.end());
#endif
    DG_RANK0 std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<"\n"
              <<"mx: "<<mx<<"\n"
              <<"my: "<<my<<"\n"
              <<"method: "<< method<<std::endl;

    dg::Timer t;
    t.tic();
    dg::geo::FluxGenerator flux( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, R_0, 0., 1);
    DG_RANK0 std::cout << "# Constructing Grid..."<<std::endl;
    dg::geo::x::CurvilinearProductGrid3d g3d(flux, n, Nx, Ny,Nz, dg::NEU, dg::PER, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    DG_RANK0 std::cout << "# Constructing Fieldlines..."<<std::endl;
    dg::geo::DS<dg::x::aProductGeometry3d, dg::x::IDMatrix, dg::x::DVec> ds(
        mag, g3d, dg::NEU, dg::PER, dg::geo::FullLimiter(), 1e-8, mx, my, -1,
        method);

    t.toc();
    DG_RANK0 std::cout << "# Construction took "<<t.diff()<<"s\n";
    ///##########################################################///
    //(MIND THE PULLBACK!)
    auto ff = dg::geo::TestFunctionPsi2(mag,a);
    const dg::x::DVec fun = dg::pullback( ff, g3d);
    dg::x::DVec derivative(fun);
    dg::x::DVec sol0 = dg::pullback( dg::geo::DsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol1 = dg::pullback( dg::geo::DssFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol2 = dg::pullback( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol3 = dg::pullback( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol4 = dg::pullback( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    std::vector<std::pair<std::string, std::array<const dg::x::DVec*,2>>> names{
         {"forward",{&fun,&sol0}},          {"backward",{&fun,&sol0}},
         {"centered",{&fun,&sol0}},         {"dss",{&fun,&sol1}},
         {"divForward",{&fun,&sol2}},       {"divBackward",{&fun,&sol2}},
         {"divCentered",{&fun,&sol2}},      {"directLap",{&fun,&sol3}}//,
         //{"invCenteredLap",{&sol4,&fun}}
    };
    ///##########################################################///
    DG_RANK0 std::cout <<"Flux:\n";
    const dg::x::DVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::x::DVec& function = *std::get<1>(tuple)[0];
        const dg::x::DVec& solution = *std::get<1>(tuple)[1];
        callDS( ds, name, function, derivative, g3d.size(),1e-8);
        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        double vol = dg::blas1::dot( vol3d, derivative)/sqrt( dg::blas2::dot( vol3d, function)); // using function in denominator makes entries comparable
        DG_RANK0 std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<std::endl
                  <<"    "<<name+"_vol:"<<std::setw(30-name.size())
                  <<" "<<vol<<"\n";
    }
    ///##########################################################///
    DG_RANK0 std::cout << "# TEST VOLUME FORMS\n";
    double volume = dg::blas1::dot( 1., ds.fieldaligned().sqrtG());
    double volumeM = dg::blas1::dot( 1., ds.fieldaligned().sqrtGm());
    double volumeP = dg::blas1::dot( 1., ds.fieldaligned().sqrtGp());
    // does error in volume form indicate a bug somewhere?
    DG_RANK0 std::cout << "volume_error:\n";
    DG_RANK0 std::cout <<"    minus:"<<std::setw(13)<<" "<<fabs(volumeM-volume)/volume<<"\n";
    DG_RANK0 std::cout <<"    plus:" <<std::setw(14)<<" "<<fabs(volumeP-volume)/volume<<"\n";
#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
