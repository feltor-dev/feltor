#include <iostream>
#include <iomanip>

#include <mpi.h>
#include "dg/algorithm.h"
#include "ds.h"
#include "guenter.h"
#include "magnetic_field.h"
#include "testfunctors.h"

const double R_0 = 3;
const double I_0 = 10; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main(int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz, mx[2], max_iter = 1e4;
    std::string method = "cubic";
    unsigned letters = 0;
    MPI_Comm comm;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)std::cout << "# Test the parallel derivative DS in cylindrical coordinates for the guenter flux surfaces. Fieldlines do not cross boundaries.\n";
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
        std::cout << "# Type method (dg, nearest, linear, cubic) \n";
        std::cin >> method;
        method.erase( std::remove( method.begin(), method.end(), '"'), method.end());
        letters = method.size();
        std::cout << "# You typed\n"
                  <<"method: "<< method<<std::endl;
        std::cout << "# Create parallel Derivative!\n";
    }
    MPI_Bcast( mx, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &letters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    method.resize( letters);
    MPI_Bcast( &method[0], letters, MPI_CHAR, 0, MPI_COMM_WORLD);
    ////////////////////////////////initialze fields /////////////////////
    const dg::CylindricalMPIGrid3d g3d( R_0 - a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER, comm);
    const dg::geo::TokamakMagneticField mag = dg::geo::createGuenterField(R_0, I_0);
    dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIDMatrix,
        dg::MDVec> ds( mag, g3d, dg::NEU, dg::NEU, dg::geo::FullLimiter(),
                1e-8, mx[0], mx[1], -1., method);

    ///##########################################################///
    auto ff = dg::geo::TestFunctionPsi2(mag,a);
    const dg::MDVec fun = dg::evaluate( ff, g3d);
    dg::MDVec derivative(fun);
    dg::MDVec sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::MDVec sol1 = dg::evaluate( dg::geo::DssFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::MDVec sol2 = dg::evaluate( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::MDVec sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::MDVec sol4 = dg::evaluate( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    std::vector<std::pair<std::string, std::array<const dg::MDVec*,2>>> names{
         {"forward",{&fun,&sol0}},          {"backward",{&fun,&sol0}},
         {"forward2",{&fun,&sol0}},         {"backward2",{&fun,&sol0}},
         {"centered",{&fun,&sol0}},         {"dss",{&fun,&sol1}},
         {"centered_bc_along",{&fun,&sol0}},{"dss_bc_along",{&fun,&sol1}},
         {"divForward",{&fun,&sol2}},       {"divBackward",{&fun,&sol2}},
         {"divCentered",{&fun,&sol2}},      {"directLap",{&fun,&sol3}},
         {"invCenteredLap",{&sol4,&fun}}
    };
    ///##########################################################///
    if(rank==0)std::cout << "# TEST Guenter (No Boundary conditions)!\n";
    if(rank==0)std::cout <<"Guenter:\n";
    const dg::MDVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::MDVec& function = *std::get<1>(tuple)[0];
        const dg::MDVec& solution = *std::get<1>(tuple)[1];
        callDS( ds, name, function, derivative, max_iter,1e-8);
        double sol = dg::blas2::dot( vol3d, solution);
        double vol = dg::blas1::dot( vol3d, derivative)/sqrt( dg::blas2::dot( vol3d, function)); // using function in denominator makes entries comparable
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        if(rank==0)std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n"
                  <<"    "<<name+"_vol:"<<std::setw(30-name.size())
                  <<" "<<vol<<"\n";
    }
    ///##########################################################///
    MPI_Finalize();
    return 0;
}
