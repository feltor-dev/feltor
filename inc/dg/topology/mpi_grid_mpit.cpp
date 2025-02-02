#include <iostream>
#include <sstream>
#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "mpi_grid.h"
#include "mpi_evaluation.h"

int function( double x, double y)
{
    return int(10000*x*y);
}
int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    unsigned nx = 2, ny = 3,  Nx = 8, Ny = 10;
    int dims[2] = {0,0};
    MPI_Dims_create( size, 2, dims);
    MPI_Comm comm;
    std::stringstream ss;
    ss<< dims[0]<<" "<<dims[1];
    dg::mpi_init2d( dg::PER, dg::PER, comm, ss);
    MPI_Comm_rank( comm, &rank);
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0) std::cout << "Test mpi grid methods!\n";
    if(rank==0) std::cout << "Test local2globalIdx\n";

    // TODO Check that all processes agree on the index
    // Assume that evaluate works
    dg::MPIGrid2d g2d( {0.,0.}, {1.,1.}, {nx,ny}, {Nx,Ny}, {dg::PER, dg::PER},
            dg::mpi_cart_split_as<2>(comm));
    auto local_vec = dg::evaluate( function, g2d);
    auto global_vec = dg::evaluate( function, g2d.global());
    auto local_vec2 = dg::global2local( global_vec, g2d);


    int local_size  = g2d.local_size();
    bool success = true;
    for( int i=0; i<local_size; i++)
    {
        if( local_vec.data()[i] != local_vec2.data()[i])
            success = false;
    }
    if( !success)
        std::cout << "FAILED from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS from rank "<<rank<<"!\n";
    if(rank==0)std::cout << std::endl;
    MPI_Barrier(comm);

    if(rank==0) std::cout << "Test global2localIdx"<<std::endl;

    success = true;
    int global_size  = g2d.size();
    for ( int i=0; i<global_size; i++)
    {
        int lIdx = 0, pid = 0;
        g2d.global2localIdx( i, lIdx, pid);
        if( pid == rank)
            if( global_vec[i] != local_vec.data()[lIdx] or lIdx >= (int)local_vec.data().size())
                success = false;
    }
    if( !success)
        std::cout << "FAILED from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS from rank "<<rank<<"!\n";
    MPI_Barrier(comm);

    if( rank==0) std::cout << "Now test that all processes agree on indices\n";
    bool success_g2l = true, success_l2g = true;
    for ( int i=0; i<global_size; i++)
    {
        int lIdx = 0, pid = 0;
        g2d.global2localIdx( i, lIdx, pid);
        int locals[size], localp[size];
        // Send everything to root
        MPI_Gather( &lIdx, 1, MPI_INT, locals, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather( &pid, 1, MPI_INT, localp, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(rank==0)
        {
            for( int k=0; k<size; k++)
                if( locals[k] != lIdx)
                    success_g2l = false;
            for( int k=0; k<size; k++)
                if( localp[k] != pid)
                    success_g2l = false;
        }
        int gIdx = 0;
        g2d.local2globalIdx( lIdx, pid, gIdx);
        if( gIdx != i)
        {
            if(rank==0)std::cerr << pid <<" lIdx "<<lIdx<< " gIdx "<<gIdx<<" "<<i<<"\n";
            success_l2g = false;
        }
    }
    std::cout << (success_g2l ? "SUCCESS" : "FAILED")
              << " g2l from rank "<<rank<<"!\n";
    std::cout << (success_l2g ? "SUCCESS" : "FAILED")
              << " l2g from rank "<<rank<<"!\n";

    // Test start and count
    int gIdx;
    g2d.local2globalIdx( 0, rank, gIdx);
    auto start = g2d.start();
    int sIdx = start[1]*g2d.shape(0) + start[0];
    std::cout << (sIdx == gIdx ? "SUCCESS" : "FAILED")
              << " start from rank "<<rank<<"!\n";

    MPI_Finalize();
    return 0;
}
