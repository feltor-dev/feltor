#include <iostream>
#include <iomanip>

#include <mpi.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "backend/timer.cuh"
#include "blas.h"
#include "backend/mpi_evaluation.h"
#include "backend/mpi_derivatives.h"
#include "backend/mpi_init.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
double function(double x, double y){ return sin(y)*sin(x);}

typedef float value_type;
//typedef dg::MPI_Vector<dg::DVec> MPIVector;
typedef dg::MPI_Vector<cusp::array1d<float, cusp::device_memory> > MPIVector;
//typedef double value_type;
//typedef dg::MPI_Vector<cusp::array1d<double, cusp::device_memory> > MPIVector;

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny; 
    MPI_Comm comm;
    dg::mpi_init2d( dg::PER, dg::PER, n, Nx, Ny, comm);

    dg::MPIGrid2d grid( 0., lx, 0, ly, n, Nx, Ny, comm);
    MPIVector w2d, v2d;
    dg::blas1::transfer( dg::create::weights(grid), w2d);
    dg::blas1::transfer( dg::create::inv_weights(grid), v2d);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    dg::Timer t;
    if(rank==0)std::cout<<"Evaluate a function on the grid\n";
    t.tic();
    MPIVector x;
    dg::blas1::transfer( dg::evaluate( function, grid), x);
    value_type gbytes=(value_type)grid.global().size()*sizeof(value_type)/1e9;
    MPIVector y(x);
    t.tic();
    for( unsigned i=0; i<20;i++)
        value_type norm = dg::blas1::dot( w2d, x);
    t.toc();
    if(rank==0)std::cout<<"DOT took                         " <<t.diff()/20.<<"s\t"<<gbytes*20./t.diff()<<"GB/s\n";

    t.tic();
    for( unsigned i=0; i<20;i++)
        dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    if(rank==0)std::cout<<"AXPBY took                       "<<t.diff()/20.<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    t.tic();
    for( unsigned i=0; i<20;i++)
        dg::blas1::pointwiseDot( y, x, x);
    t.toc();
    if(rank==0)std::cout<<"pointwiseDot took                "<<t.diff()/20<<"s\t" <<gbytes*20/t.diff()<<"GB/s\n";

    MPI_Finalize();
    return 0;
}
