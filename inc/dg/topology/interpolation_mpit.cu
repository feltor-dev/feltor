#include <iostream>
#include <sstream>
#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "dg/backend/transpose.h"
#include "dg/blas.h"
#include "mpi_projection.h"
#include "mpi_evaluation.h"


double shift = 0.2;
double function( double x, double y){ return sin(M_PI*x/2.)*sin(M_PI*y/2.);}

int main(int argc, char* argv[])
{

    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    //if(size!=4)
    //{
    //    std::cerr << "Please run with 4 processes!\n";
    //    MPI_Finalize();
    //    return 0;
    //}
    unsigned n = 3, Nx = 8, Ny = 8;
    int dims[2] = {0,0};
    MPI_Dims_create( size, 2, dims);
    MPI_Comm comm;
    std::stringstream ss;
    ss<< dims[0]<<" "<<dims[1];
    dg::mpi_init2d( dg::PER, dg::PER, comm, ss);
    MPI_Comm_rank( comm, &rank);
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0) std::cout << "Test NON-COMMUNICATING MPI matrix-creation!\n";
    dg::MPIGrid2d g2d( 0,1,0,1, n,Nx,Ny, comm);
    dg::MPIGrid2d g2d_half = g2d;
    g2d_half.multiplyCellNumbers(0.5, 0.5);

    dg::MIHMatrix converted_p = dg::create::interpolation( g2d, g2d_half);
    dg::MHVec sine = dg::evaluate( function, g2d_half);
    dg::MHVec temp = dg::evaluate( function, g2d);
    converted_p.symv( sine, temp);

    dg::IHMatrix direct_p= dg::create::interpolation( g2d.global(), g2d_half.global());
    dg::HVec global_sine = dg::evaluate( function, g2d_half.global());
    dg::HVec g_temp( g2d.global().size());
    dg::blas2::symv( direct_p, global_sine, g_temp);
    //now compare
    bool success = true;
    for( unsigned i=0; i<temp.size(); i++) //local size
    {
        int gIdx = 0;
        g2d.local2globalIdx( i, rank, gIdx);
        if( fabs(temp.data()[i] - g_temp[gIdx]) > 1e-14)
        {
            std::cerr << rank<<" "<<i<<" "<<gIdx<<" "<<temp.data()[i]<<" "<<g_temp[gIdx]<<"\n";
            success = false;
        }
    }
    if( !success)
        std::cout << "FAILED creation from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS creation from rank "<<rank<<"!\n";
    MPI_Barrier(comm);

    /////%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0) std::cout << "Now test COMMUNICATING MPI matrix-creation!\n";
    auto x = dg::evaluate( dg::cooX2d, g2d.local());
    auto y = dg::evaluate( dg::cooY2d, g2d.local());
    for( unsigned i=0; i<x.size(); i++)
    {
        x[i] +=shift;
        y[i] +=shift;
        bool negative = false;
        dg::create::detail::shift( negative, x[i], g2d.bcx(), g2d.global().x0(), g2d.global().x1());
        dg::create::detail::shift( negative, y[i], g2d.bcy(), g2d.global().y0(), g2d.global().y1());
    }
    dg::MIHMatrix converted_i = dg::create::interpolation( x,y,g2d);
    dg::IHMatrix  direct_i = dg::create::interpolation( x,y,g2d.global());
    sine = dg::evaluate( function, g2d);
    temp = sine;
    converted_i.symv( sine, temp);
    global_sine = dg::evaluate( function, g2d.global());
    g_temp.resize( g2d.local().size());
    dg::blas2::symv( direct_i, global_sine, g_temp);
    //now compare
    success = true;
    for( unsigned i=0; i<temp.size(); i++) //local size
    {
        if( fabs(temp.data()[i] - g_temp[i]) > 1e-12)
        {
            std::cerr << rank<<" "<<i<<" "<<temp.data()[i]<<" "<<g_temp[i]<<"\n";
            success = false;
        }
    }
    if( !success)
        std::cout << "FAILED comm from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS comm from rank "<<rank<<"!\n";
    MPI_Barrier(comm);
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0) std::cout << "Now test COMMUNICATING MPI local2global interpolation!\n";
    dg::MIHMatrix local2global;
    if(rank==0)
    {
        auto xx = dg::evaluate( dg::cooX2d, g2d.global());
        auto yy = dg::evaluate( dg::cooY2d, g2d.global());
        local2global = dg::create::interpolation( xx,yy,g2d);
    }
    else
    {
        auto xx = dg::HVec(); // empty for all other pids
        auto yy = dg::HVec();
        local2global = dg::create::interpolation( xx,yy,g2d);
    }
    dg::MHVec mpi_sine = dg::evaluate( function, g2d);
    dg::MHVec mpi_temp( dg::HVec(rank==0?g2d.size() : 0), g2d.communicator());
    local2global.symv( mpi_sine, mpi_temp);
    global_sine = dg::evaluate( function, g2d.global());
    //now compare
    success = true;
    if(rank==0)
    {
        // Test local matrix
        auto identity = global_sine;
        dg::blas2::symv( local2global.matrix(), global_sine, identity);
        for( unsigned u=0; u<identity.size(); u++)
            if( fabs(global_sine[u] - identity[u]) > 1e-14)
                success = false;
        for( unsigned i=0; i<mpi_temp.size(); i++)
            if( fabs(mpi_temp.data()[i] - global_sine[i]) > 1e-14)
            {
                std::cerr << i<<" "<<mpi_temp.data()[i]<<" "<<global_sine[i]<<"\n";
                success = false;
            }
    }
    else
        if( mpi_temp.data().size() != 0)
            success = false;

    if( !success)
        std::cout << "FAILED  local2global from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS local2global from rank "<<rank<<"!\n";
    MPI_Barrier(comm);
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0) std::cout << "Now test TRANSPOSITION!\n";
    converted_i = dg::transpose( converted_i);
    converted_i.symv( sine, temp);
    //Compute global transposition and distribute among processes
    x = dg::evaluate( dg::cooX2d, g2d.global());
    y = dg::evaluate( dg::cooY2d, g2d.global());
    for( unsigned i=0; i<x.size(); i++)
    {
        x[i] +=shift;
        y[i] +=shift;
        bool negative = false;
        dg::create::detail::shift( negative, x[i], g2d.bcx(), g2d.global().x0(), g2d.global().x1());
        dg::create::detail::shift( negative, y[i], g2d.bcy(), g2d.global().y0(), g2d.global().y1());
    }
    direct_i = dg::transpose(dg::create::interpolation( x,y,g2d.global()));
    g_temp.resize( g2d.global().size());
    dg::blas2::symv( direct_i, global_sine, g_temp);
    dg::MHVec mpi_g_temp = dg::global2local( g_temp, g2d);
    //now compare
    success = true;
    for( unsigned i=0; i<temp.size(); i++)
        if( fabs(temp.data()[i] - mpi_g_temp.data()[i]) > 1e-14)
        {
            std::cout << rank << " "<<i<<" "<<temp.data()[i]<<" "<<mpi_g_temp.data()[i]<<"\n";
            success = false;
        }
    if( !success)
        std::cout << "FAILED Transposition from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS Transposition from rank "<<rank<<"!\n";

    MPI_Finalize();
    return 0;
}
