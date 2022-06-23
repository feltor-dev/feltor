#include <iostream>
#include <sstream>
#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "dg/backend/transpose.h"
#include "dg/blas.h"
#include "dg/blas1.h"
#include "mpi_weights.h"
#include "mpi_projection.h"
#include "mpi_evaluation.h"
#include "fast_interpolation.h"

double sine( double x){ return sin(x);}
double sine( double x, double y){return sin(x)*sin(y);}
double sine( double x, double y, double z){return sin(x)*sin(y);}
//Actually this file is a test for fast_interpolation

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    if(size!=4)
    {
        std::cerr << "Please run with 4 processes!\n";
        MPI_Finalize();
        return 0;
    }
    unsigned n, Nx, Ny, Nz;
    MPI_Comm comm;
    std::stringstream ss;
    ss<< "2 2 1 3 8 8 8";
    mpi_init3d( dg::PER, dg::PER, dg::PER, n, Nx, Ny, Nz, comm, ss);
    MPI_Comm_rank( comm, &rank);

    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0)std::cout << "TEST 2D and 3D\n";
    unsigned n_old = 9, n_new = 3, N_old = 40, N_new = 20;
    if(rank==0)std::cout << "Fine   Grid "<< n_old << " x "<<N_old <<"\n";
    if(rank==0)std::cout << "Coarse Grid "<< n_new << " x "<<N_new <<"\n";
    dg::MPIGrid3d g2o (0, M_PI, 0, M_PI, 0,1, n_old, N_old, N_old, 4, comm);
    dg::MPIGrid3d g2n (0, M_PI, 0, M_PI, 0,1, n_new, N_new, N_new, 4, comm);
    dg::MIHMatrix inte2d = dg::create::interpolation( g2n, g2o);
    auto proj2d = dg::create::fast_projection( g2o, n_old/n_new, N_old/N_new, N_old/N_new);
    auto fast_inte2d = dg::create::fast_interpolation( g2n, n_old/n_new, N_old/N_new, N_old/N_new);
    auto forward = dg::create::fast_transform( g2o.dltx().forward(), g2o.dlty().forward(), g2o);
    auto backward = dg::create::fast_transform( g2o.dltx().backward(), g2o.dlty().backward(), g2o);
    const dg::MHVec sinO( dg::evaluate( sine, g2o)),
                                sinN( dg::evaluate( sine, g2n));
    dg::MHVec w2do = dg::create::weights( g2o);
    dg::MHVec w2dn = dg::create::weights( g2n);
    dg::MHVec sinP( sinN), sinI(sinO), sinF(sinO);
    dg::blas2::gemv( proj2d, sinO, sinP); //FAST PROJECTION
    double value0 = sqrt(dg::blas2::dot( sinO, w2do, sinO));
    if(rank==0)std::cout << "Original vector     "<<value0 << "\n";
    double value1 = sqrt(dg::blas2::dot( sinP, w2dn, sinP));
    if(rank==0)std::cout << "Projected vector    "<<value1 << "\n";
    if(rank==0)std::cout << "Difference in Norms "<<value0-value1 << std::endl;
    dg::blas1::axpby( 1., sinN, -1., sinP);
    double value2 = sqrt(dg::blas2::dot( sinP, w2dn, sinP)/dg::blas2::dot(sinN, w2dn, sinN));
    if(rank==0)std::cout << "Difference between projection and evaluation      "<<value2<<"\n";
    dg::blas2::gemv( inte2d, sinO, sinP);
    value1 = sqrt(dg::blas2::dot( sinP, w2dn, sinP));
    if(rank==0)std::cout << "Interpolated vec    "<<value1 << "\n";
    value0 = sqrt(dg::blas2::dot( sinO, w2do, sinO));
    if(rank==0)std::cout << "Difference in Norms "<<value0 - value1 << "\n" << std::endl;
    dg::blas2::gemv( fast_inte2d, sinN, sinI);
    value2 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
    if(rank==0)std::cout << "Fast Interpolated vec "<< value2 << "\n";
    double value3 = sqrt(dg::blas2::dot( sinN, w2dn, sinN));
    if(rank==0)std::cout << "Difference in Norms   "<<value2 - value3  << "\n" << std::endl;
    dg::blas2::gemv( forward, sinO, sinF);
    dg::blas2::gemv( backward, sinF, sinI);
    dg::blas1::axpby( 1., sinO, -1., sinI);
    value2 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
    if(rank==0)std::cout << "Forward-Backward Error   "<<value2 << " (Must be zero)\n" << std::endl;

    MPI_Finalize();
    return 0;
}
