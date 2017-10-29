#include <iostream>

#include <mpi.h>
#define DG_BENCHMARK
#include "dg/backend/timer.cuh"
#include "dg/backend/mpi_evaluation.h"
#include "dg/backend/mpi_init.h"
#include "dg/backend/functions.h"
#include "dg/geometry/geometry.h"
#include "ds.h"
#include "toroidal.h"

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius  
double func(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return r2*sin(phi);
}
double deri(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z; //(grad psi)^2
    return I_0/R/sqrt(I_0*I_0 + r2)* r2*cos(phi);
}


int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
    dg::mpi_init3d( dg::NEU, dg::NEU, dg::PER, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::CylindricalMPIGrid3d g3d( R_0 - a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER, comm);
    const dg::MDVec vol3d = dg::create::volume( g3d);
    if(rank==0)std::cout << "Note that it's faster to compute with OMP_NUM_THREADS=1\n";
    if(rank==0)std::cout << "Create parallel Derivative!\n";
    dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    dg::geo::BinaryVectorLvl0 bhat( (dg::geo::BHatR)(mag), (dg::geo::BHatZ)(mag), (dg::geo::BHatP)(mag));
    dg::geo::Fieldaligned<dg::aProductMPIGeometry3d,dg::MIDMatrix,dg::MDVec>  dsFA( bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-6, 5,5,true,true);
    dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIDMatrix, dg::MDMatrix, dg::MDVec> ds( dsFA, dg::not_normed, dg::centered);
    if(rank==0)std::cout << "Ready!\n";

    dg::MDVec function = dg::evaluate( func, g3d), derivative(function);
    const dg::MDVec solution = dg::evaluate( deri, g3d);
    ds( function, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    double norm = dg::blas2::dot( derivative, vol3d, derivative);
    const double sol = dg::blas2::dot( vol3d, solution);
    if(rank==0)std::cout << "Error centered derivative "<< sqrt( norm/sol )<<"\n";
    ds.forward( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Error Forward  Derivative "<<sqrt( norm/sol)<<"\n";
    ds.backward( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Error Backward Derivative "<<sqrt( norm/sol)<<"\n";
    if(rank==0)std::cout << "(Since the function is a parabola, the error is from the parallel derivative only if n>2/ no interpolation error)\n"; 
    if(rank==0)std::cout << "TEST FIELDALIGNED EVALUATION of a Gaussian\n";
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    function = ds.fieldaligned().evaluate( init0, modulate, Nz/2, 2);
    ds( function, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_t)\n";
    MPI_Finalize();
    return 0;
}
