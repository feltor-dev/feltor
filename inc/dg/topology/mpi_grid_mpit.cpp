#include <iostream>
#include <sstream>
#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "mpi_grid.h"
#include "mpi_evaluation.h"

#include "catch2/catch_all.hpp"

static int function( double x, double y)
{
    return int(10000*x*y);
}
// TODO Test what happens if # processes is larger than N?
TEST_CASE( "MPI Grid")
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    unsigned nx = 2, ny = 3,  Nx = 8, Ny = 10;
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1,1});
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    // Assume that evaluate works
    dg::MPIGrid2d g2d( {0.,0.}, {1.,1.}, {nx,ny}, {Nx,Ny}, {dg::PER, dg::PER},
            dg::mpi_cart_split_as<2>(comm));
    auto local_vec = dg::evaluate( function, g2d);
    auto global_vec = dg::evaluate( function, g2d.global());
    SECTION( "global2local")
    {
        auto local_vec2 = dg::global2local( global_vec, g2d);
        int local_size  = g2d.local_size();
        for( int i=0; i<local_size; i++)
            CHECK( local_vec.data()[i] == local_vec2.data()[i]);
    }

    SECTION("global2localIdx")
    {
        int global_size = g2d.size();
        for ( int i=0; i<global_size; i++)
        {
            int lIdx = 0, pid = 0;
            g2d.global2localIdx( i, lIdx, pid);
            if( pid == rank)
            {
                CHECK( global_vec[i] == local_vec.data()[lIdx]);
                CHECK( lIdx < (int)local_vec.data().size());
            }
        }
    }

    SECTION( "Processes agree on indices")
    {
        int global_size = g2d.size();
        for ( int i=0; i<global_size; i++)
        {
            int lIdx = 0, pid = 0;
            g2d.global2localIdx( i, lIdx, pid);
            int locals[size], localp[size];
            // Send everything to root
            MPI_Gather( &lIdx, 1, MPI_INT, locals, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(  &pid, 1, MPI_INT, localp, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(rank==0)
            {
                for( int k=0; k<size; k++)
                    CHECK( locals[k] == lIdx);
                for( int k=0; k<size; k++)
                    CHECK( localp[k] == pid);
            }
            int gIdx = 0;
            g2d.local2globalIdx( lIdx, pid, gIdx);
            INFO("Rank "<<rank<<" pid "<<pid <<" lIdx "<<lIdx<< " gIdx "<<gIdx<<" "<<i);
            CHECK( gIdx == i);
        }
    }
    SECTION( "Start and count")
    {
        int gIdx;
        g2d.local2globalIdx( 0, rank, gIdx);
        auto start = g2d.start();
        int sIdx = start[0]*g2d.shape(0) + start[1];
        INFO( "Rank "<<rank<<" sIdx "<<sIdx<<" gIdx "<<gIdx);
        CHECK( sIdx == gIdx);
    }
}
