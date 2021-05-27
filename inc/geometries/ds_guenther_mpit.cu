#include <iostream>
#include <iomanip>

#include <mpi.h>
#define DG_BENCHMARK
#undef DG_DEBUG
#include "dg/algorithm.h"
#include "ds.h"
#include "guenther.h"
#include "magnetic_field.h"
#include "testfunctors.h"

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main(int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz, mx[2], max_iter = 1e4;
    MPI_Comm comm;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)std::cout << "# Test the parallel derivative DS in cylindrical coordinates for the guenther flux surfaces. Fieldlines do not cross boundaries.\n";
    dg::mpi_init3d( dg::NEU, dg::NEU, dg::PER, n, Nx, Ny, Nz, comm);
    if( rank == 0)
    {
        std::cout <<"# You typed\n"
                  <<"n:  "<<n<<"\n"
                  <<"Nx: "<<Nx<<"\n"
                  <<"Ny: "<<Ny<<"\n"
                  <<"Nz: "<<Nz<<std::endl;
        std::cout <<"# Type mx (10) and my (10)\n";
        std::cin >> mx[0] >> mx[1];
        std::cout << "# You typed\n"
                  <<"mx: "<<mx[0]<<"\n"
                  <<"my: "<<mx[1]<<std::endl;
        std::cout << "# Create parallel Derivative!\n";
    }
    MPI_Bcast( mx, 2, MPI_INT, 0, MPI_COMM_WORLD);
    ////////////////////////////////initialze fields /////////////////////
    const dg::CylindricalMPIGrid3d g3d( R_0 - a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER, comm);
    const dg::geo::TokamakMagneticField mag = dg::geo::createGuentherField(R_0, I_0);
    dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIDMatrix, dg::MDMatrix, dg::MDVec> ds(
        mag, g3d, dg::NEU, dg::NEU, dg::geo::FullLimiter(),
        dg::centered, 1e-8, mx[0], mx[1]);

    ///##########################################################///
    const dg::MDVec fun = dg::evaluate( dg::geo::TestFunctionPsi2(mag), g3d);
    const dg::MDVec divb = dg::evaluate( dg::geo::Divb(mag), g3d);
    dg::MDVec derivative(fun);
    dg::MDVec sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol1 = dg::evaluate( dg::geo::DssFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol2 = dg::evaluate( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol4 = dg::evaluate( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    std::vector<std::pair<std::string, std::array<const dg::MDVec*,2>>> names{
         {"forward",{&fun,&sol0}},          {"backward",{&fun,&sol0}},
         {"centered",{&fun,&sol0}},         {"dss",{&fun,&sol1}},
         {"centered_bc_along",{&fun,&sol0}},{"dss_bc_along",{&fun,&sol1}},
         {"divForward",{&fun,&sol2}},       {"divBackward",{&fun,&sol2}},
         {"divCentered",{&fun,&sol2}},      {"divDirectForward",{&fun,&sol2}},
         {"divDirectBackward",{&fun,&sol2}},{"divDirectCentered",{&fun,&sol2}},
         {"forwardLap",{&fun,&sol3}},       {"backwardLap",{&fun,&sol3}},
         {"centeredLap",{&fun,&sol3}},      {"directLap",{&fun,&sol3}},
         {"directLap_bc_along",{&fun,&sol3}},
         {"invForwardLap",{&sol4,&fun}},    {"invBackwardLap",{&sol4,&fun}},
         {"invCenteredLap",{&sol4,&fun}}
    };
    ///##########################################################///
    if(rank==0)std::cout << "# TEST Guenther (No Boundary conditions)!\n";
    if(rank==0)std::cout <<"Guenther:\n";
    const dg::MDVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::MDVec& function = *std::get<1>(tuple)[0];
        const dg::MDVec& solution = *std::get<1>(tuple)[1];
        callDS( ds, name, function, derivative, divb, max_iter,1e-8);
        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        if(rank==0)std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    MPI_Finalize();
    return 0;
}
