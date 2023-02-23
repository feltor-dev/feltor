#include <iostream>
#include <sstream>
#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "dg/backend/transpose.h"
#include "dg/blas.h"
#include "mpi_projection.h"
#include "mpi_evaluation.h"


double shift = 0.2;
double function( double x, double y){ return sin(2*M_PI*x)*sin(2*M_PI*y);}

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
    unsigned n, Nx, Ny;
    MPI_Comm comm;
    std::stringstream ss;
    ss<< "2 2 3 8 8";
    mpi_init2d( dg::PER, dg::PER, n, Nx, Ny, comm, ss);
    MPI_Comm_rank( comm, &rank);
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0) std::cout << "Test NON-COMMUNICATING MPI matrix-creation!\n";
    dg::MPIGrid2d g2d( 0,1,0,1, n,Nx,Ny, comm);
    dg::MPIGrid2d g2d_half = g2d;
    g2d_half.multiplyCellNumbers(0.5, 0.5);
    dg::MIHMatrix direct_p= dg::create::interpolation( g2d, g2d_half);
    dg::HVec x = dg::evaluate( dg::cooX2d, g2d.local());
    dg::HVec y = dg::evaluate( dg::cooY2d, g2d.local());
    dg::IHMatrix global_projection = dg::create::interpolation( x,y, g2d_half.global());
    dg::MIHMatrix converted_p = dg::convert(global_projection, g2d_half);

    //now compare
    bool equal_cols=true, equal_rows=true, equal_values=true;
    for( unsigned i=0; i<direct_p.matrix().values.size(); i++)
    {
        if( direct_p.matrix().column_indices[i] - converted_p.matrix().column_indices[i] > 1e-15) equal_cols = false;
        if( direct_p.matrix().values[i] - converted_p.matrix().values[i] > 1e-15) equal_values = false;
    }
    for( unsigned i=0; i<direct_p.matrix().num_rows+1; i++)
        if( direct_p.matrix().row_offsets[i] - converted_p.matrix().row_offsets[i] > 1e-15) equal_rows = false;

    if( !equal_cols || !equal_rows || !equal_values || direct_p.collective().buffer_size() != 0 || converted_p.collective().buffer_size() != 0 )
        std::cout << "FAILED from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS from rank "<<rank<<"!\n";

    MPI_Barrier(comm);
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0) std::cout << "Now test COMMUNICATING MPI matrix-creation!\n";
    x = dg::evaluate( dg::cooX2d, g2d.local());
    y = dg::evaluate( dg::cooY2d, g2d.local());
    for( unsigned i=0; i<x.size(); i++)
    {
        x[i] +=shift;
        y[i] +=shift;
        bool negative = false;
        g2d.global().shift( negative, x[i], y[i]);
    }
    dg::MIHMatrix converted_i = dg::create::interpolation( x,y,g2d);
    dg::IHMatrix  direct_i = dg::create::interpolation( x,y,g2d.global());
    dg::MHVec sine = dg::evaluate( function, g2d);
    dg::MHVec temp(sine);
    converted_i.symv( sine, temp);
    dg::HVec global_sine = dg::evaluate( function, g2d.global());
    dg::HVec g_temp( g2d.local().size());
    dg::blas2::symv( direct_i, global_sine, g_temp);
    //now compare
    bool success = true;
    for( unsigned i=0; i<temp.size(); i++)
        if( fabs(temp.data()[i] - g_temp[i]) > 1e-14)
            success = false;
    if( !success)
        std::cout << "FAILED from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS from rank "<<rank<<"!\n";
    MPI_Barrier(comm);
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0) std::cout << "Now test COMMUNICATING MPI local2global interpolation!\n";
    if(rank==0)
    {
        x = dg::evaluate( dg::cooX2d, g2d.global());
        y = dg::evaluate( dg::cooY2d, g2d.global());
    }
    else
    {
        x = dg::HVec(); // empty for all other pids
        y = dg::HVec();
    }
    dg::MIHMatrix local2global = dg::create::interpolation( x,y,g2d);
    dg::MHVec mpi_sine = dg::evaluate( function, g2d);
    dg::MHVec mpi_temp( x, g2d.communicator());
    local2global.symv( mpi_sine, mpi_temp);
    global_sine = dg::evaluate( function, g2d.global());
    //now compare
    success = true;
    if(rank==0)
    {
        for( unsigned i=0; i<mpi_temp.size(); i++)
            if( fabs(mpi_temp.data()[i] - global_sine[i]) > 1e-14)
                success = false;
    }
    else
        if( mpi_temp.data().size() != 0)
            success = false;

    if( !success)
        std::cout << "FAILED from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS from rank "<<rank<<"!\n";
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
        g2d.global().shift( negative, x[i], y[i]);
    }
    direct_i = dg::transpose(dg::create::interpolation( x,y,g2d.global()));
    g_temp.resize( g2d.global().size());
    dg::blas2::symv( direct_i, global_sine, g_temp);
    dg::MHVec mpi_g_temp = dg::global2local( g_temp, g2d);
    //now compare
    success = true;
    for( unsigned i=0; i<temp.size(); i++)
        if( fabs(temp.data()[i] - mpi_g_temp.data()[i]) > 1e-14)
            success = false;
    if( !success)
        std::cout << "FAILED from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS from rank "<<rank<<"!\n";

    MPI_Finalize();
    return 0;
}
