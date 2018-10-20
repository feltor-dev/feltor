#include <iostream>
#include <iomanip>

#include <mpi.h>
#define DG_BENCHMARK
#undef DG_DEBUG
#include "dg/algorithm.h"
#include "magnetic_field.h"
#include "testfunctors.h"
#include "ds.h"
#include "toroidal.h"

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz, mx[2], max_iter = 1e4;
    MPI_Comm comm;
    dg::mpi_init3d( dg::NEU, dg::NEU, dg::PER, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
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

    const dg::CylindricalMPIGrid3d g3d( R_0-a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER, comm);
    //create magnetic field
    const dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    const dg::geo::BinaryVectorLvl0 bhat( (dg::geo::BHatR)(mag), (dg::geo::BHatZ)(mag), (dg::geo::BHatP)(mag));
    //create Fieldaligned object and construct DS from it
    dg::geo::Fieldaligned<dg::aProductMPIGeometry3d,dg::MIDMatrix,dg::MDVec>  dsFA( bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx[0], mx[1]);
    dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIDMatrix, dg::MDMatrix, dg::MDVec> ds( dsFA, dg::centered);
    ///##########################################################///
    //apply to function
    const dg::MDVec functionNEU = dg::evaluate( dg::geo::TestFunctionSin(mag), g3d);
    const dg::MDVec functionDIR = dg::evaluate( dg::geo::TestFunctionCos(mag), g3d);
    dg::MDVec derivative(functionNEU);
    dg::MDVec sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::TestFunctionSin>(mag), g3d);
    dg::MDVec sol1 = dg::evaluate( dg::geo::DssFunction<dg::geo::TestFunctionSin>(mag), g3d);
    dg::MDVec sol2 = dg::evaluate( dg::geo::DsDivFunction<dg::geo::TestFunctionSin>(mag), g3d);
    dg::MDVec sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::TestFunctionSin>(mag), g3d);
    std::vector<std::pair<std::string, const dg::MDVec&>> names{
         {"forward",sol0}, {"backward",sol0},
         {"centered",sol0}, {"dss",sol1},
         {"forwardDiv",sol2}, {"backwardDiv",sol2}, {"centeredDiv",sol2},
         {"forwardLap",sol3}, {"backwardLap",sol3}, {"centeredLap",sol3}
    };
    std::vector<std::pair<std::string, dg::direction>> namesLap{
         {"invForwardLap",dg::forward}, {"invBackwardLap",dg::backward}, {"invCenteredLap",dg::centered}
    };
    if(rank==0)std::cout << "# TEST NEU Boundary conditions!\n";
    if(rank==0)std::cout << "# TEST ADJOINT derivatives do unfortunately not fulfill Neumann BC!\n";
    ///##########################################################///
    if(rank==0)std::cout <<"Neumann:\n";
    dg::MDVec vol3d = dg::create::volume( g3d);
    for( const auto& name :  names)
    {
        callDS( ds, name.first, functionNEU, derivative);
        double sol = dg::blas2::dot( vol3d, name.second);
        dg::blas1::axpby( 1., name.second, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        if(rank==0)std::cout <<"    "<<name.first<<":"
                  <<std::setw(16-name.first.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    dg::MDVec solution =dg::evaluate( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionSin>(mag), g3d);
    dg::Invert<dg::MDVec> invert( solution, max_iter, 1e-6, 1);
    dg::geo::TestInvertDS< dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIDMatrix, dg::MDMatrix, dg::MDVec>, dg::MDVec>
        rhs(ds);
    for( auto name : namesLap)
    {
        ds.set_direction( name.second);
        dg::blas1::scal( derivative, 0);
        invert( rhs, derivative, solution);
        dg::blas1::axpby( 1., functionNEU, -1., derivative);
        double sol = dg::blas2::dot( vol3d, functionNEU);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        if(rank==0)std::cout <<"    "<<name.first<<":"
                  <<std::setw(16-name.first.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    if(rank==0)std::cout << "# Reconstruct parallel derivative!\n";
    dsFA.construct( bhat, g3d, dg::DIR, dg::DIR, dg::geo::NoLimiter(), 1e-8, mx[0], mx[1]);
    ds.construct( dsFA, dg::centered);
    sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::TestFunctionCos>(mag), g3d);
    sol1 = dg::evaluate( dg::geo::DssFunction<dg::geo::TestFunctionCos>(mag), g3d);
    sol2 = dg::evaluate( dg::geo::DsDivFunction<dg::geo::TestFunctionCos>(mag), g3d);
    sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::TestFunctionCos>(mag), g3d);
    if(rank==0)std::cout << "# TEST DIR Boundary conditions!\n";
    ///##########################################################///
    if(rank==0)std::cout << "Dirichlet: \n";
    for( const auto& name :  names)
    {
        callDS( ds, name.first, functionDIR, derivative);
        double sol = dg::blas2::dot( vol3d, name.second);
        dg::blas1::axpby( 1., name.second, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        if(rank==0)std::cout <<"    "<<name.first<<":"
                  <<std::setw(16-name.first.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    solution =dg::evaluate( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionCos>(mag), g3d);
    for( auto name : namesLap)
    {
        ds.set_direction( name.second);
        invert(rhs, derivative, solution);
        dg::blas1::axpby( 1., functionDIR, -1., derivative);
        double sol = dg::blas2::dot( vol3d, functionDIR);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        if(rank==0)std::cout <<"    "<<name.first<<":"
                  <<std::setw(16-name.first.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }

    ///##########################################################///

    if(rank==0)std::cout << "TEST FIELDALIGNED EVALUATION of a Gaussian\n";
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    dg::MDVec aligned = ds.fieldaligned().evaluate( init0, modulate, Nz/2, 2);
    ds( aligned, derivative);
    double norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_t)\n";
    MPI_Finalize();
    return 0;
}
