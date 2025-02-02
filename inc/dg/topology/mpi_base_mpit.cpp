#include <iostream>
#include <sstream>
#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "mpi_base.h"

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    int dims[3] = {0,0,0};
    MPI_Dims_create( size, 3, dims);
    MPI_Comm comm3d;
    std::stringstream ss;
    ss<< dims[0]<<" "<<dims[1]<<" "<<dims[2];
    dg::mpi_init3d( dg::DIR, dg::DIR, dg::PER, comm3d, ss);
    int dims2d[2] = {0,0};
    MPI_Dims_create( size, 2, dims2d);
    MPI_Comm comm2d;
    std::stringstream ss2d;
    ss2d<< dims2d[0]<<" "<<dims2d[1];
    dg::mpi_init2d( dg::DIR, dg::DIR, comm2d, ss2d);
    MPI_Comm_rank( comm2d, &rank);
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0) std::cout << "Test mpi grid methods!\n";
    dg::RealCartesianMPIGrid2d g2d(
            1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 3, 10, 10, dg::DIR, dg::DIR, comm2d);
    dg::RealCartesianMPIGrid2d g2d_test{ g2d.gx(),g2d.gy()};
    assert( g2d.shape(0) == 30);
    assert( g2d.shape(1) == 30);
    assert( g2d_test.shape(0) == 30);
    assert( g2d_test.shape(1) == 30);
    dg::RealCartesianMPIGrid3d g3d(
        1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0., 2.*M_PI, 3, 10, 10, 10,
        dg::DIR, dg::DIR, dg::PER, comm3d);
    assert( g3d.shape(0) == 30);
    assert( g3d.shape(1) == 30);
    assert( g3d.shape(2) == 10);
    dg::RealCylindricalMPIGrid3d c3d(
        1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0., 2.*M_PI, 3, 10, 10, 10,
        dg::DIR, dg::DIR, dg::PER, comm3d);
    assert( c3d.shape(0) == 30);
    assert( c3d.shape(1) == 30);
    assert( c3d.shape(2) == 10);
    std::cout << "Rank "<<rank<<" PASSED\n";


    MPI_Finalize();
    return 0;
}
