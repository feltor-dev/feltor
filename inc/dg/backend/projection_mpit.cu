#include <iostream>
#include <sstream>
#include <mpi.h>
#include "mpi_projection.h"
#include "mpi_init.h"
#include "mpi_evaluation.h"


double shift = 0.2;
double function( double x, double y){ return sin(2*M_PI*x)*sin(2*M_PI*y);}
double shifted_function(double x, double y){return function( x-shift, y-shift);}

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
    if(rank==0) std::cout << "Test non-communicating MPI matrix-creation!\n";
    dg::MPIGrid2d g2d( 0,1,0,1, n,Nx,Ny, comm);
    dg::MPIGrid2d g2d_half = g2d;
    g2d_half.multiplyCellNumbers(0.5, 0.5);
    dg::MIHMatrix direct_p= dg::create::interpolation( g2d, g2d_half);
    dg::HVec x = dg::evaluate( dg::cooX2d, g2d.local());
    dg::HVec y = dg::evaluate( dg::cooY2d, g2d.local());
    dg::IHMatrix global_projection = dg::create::interpolation( x,y, g2d_half.global());
    dg::MIHMatrix converted_p = dg::create::convert_row_dist(global_projection, g2d_half);

    //now compare
    bool equal_cols=true, equal_rows=true, equal_values=true;
    for( unsigned i=0; i<direct_p.matrix().values.size(); i++)
    {
        if( direct_p.matrix().column_indices[i] - converted_p.matrix().column_indices[i] > 1e-15) equal_cols = false;
        if( direct_p.matrix().values[i] - converted_p.matrix().values[i] > 1e-15) equal_values = false;
    }
    for( unsigned i=0; i<direct_p.matrix().num_rows+1; i++)
        if( direct_p.matrix().row_offsets[i] - converted_p.matrix().row_offsets[i] > 1e-15) equal_rows = false;

    if( !equal_cols || !equal_rows || !equal_values || direct_p.collective().size() != 0 || converted_p.collective().size() != 0 )
        std::cout << "FAILED from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS from rank "<<rank<<"!\n";

    MPI_Barrier(comm);
    if(rank==0) std::cout << "Now test communicating MPI matrix-creation!\n";
    x = dg::evaluate( dg::cooX2d, g2d.local());
    y = dg::evaluate( dg::cooY2d, g2d.local());
    for( unsigned i=0; i<x.size(); i++)
    {
        x[i] -=shift;
        y[i] -=shift;
        g2d.global().shift_topologic( x[i], y[i], x[i], y[i]);
    }
    dg::MIHMatrix converted_i = dg::create::interpolation( x,y,g2d);
    dg::MHVec sine = dg::evaluate( function, g2d); 
    dg::MHVec temp(sine);
    dg::MHVec shifted_sine = dg::evaluate( shifted_function, g2d);
    converted_i.symv( sine, temp);
    //now compare
    bool success = true;
    for( unsigned i=0; i<temp.size(); i++)
        if( temp.data()[i] - shifted_sine.data()[i] > 1e-14) 
            success = false; 
    if( !success) 
        std::cout << "FAILED from rank "<<rank<<"!\n";
    else
        std::cout << "SUCCESS from rank "<<rank<<"!\n";

    MPI_Finalize();
    return 0;
}
