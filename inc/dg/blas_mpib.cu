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

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny; 
    MPI_Comm comm;
    mpi_init2d( dg::PER, dg::PER, n, Nx, Ny, comm);

    dg::MPI_Grid2d grid( 0., lx, 0, ly, n, Nx, Ny, comm);
    const dg::MDVec w2d = dg::create::weights( grid);
    const dg::MDVec v2d = dg::create::inv_weights( grid);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    double gbytes=(double)grid.global().size()*sizeof(double)/1e9;
    dg::Timer t;
    if(rank==0)std::cout<<"Evaluate a function on the grid\n";
    t.tic();
    dg::MDVec x = dg::evaluate( function, grid);
    t.toc();
    if(rank==0)std::cout<<"Evaluation of a function took    "<<t.diff()<<"s\n";
    t.tic();
    for( int i=0; i<20; i++)
        double norm = dg::blas2::dot( w2d, x);
    t.toc();
    if(rank==0)std::cout<<"DOT took                         " <<t.diff()/20.<<"s\t"<<gbytes*20./t.diff()<<"GB/s\n";
    dg::MDVec y(x);
    dg::MDMatrix M = dg::create::dx( grid, dg::centered);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"centered x derivative took       "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";
    M = dg::create::dx( grid, dg::forward);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"forward x derivative took        "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";
    M = dg::create::dy( grid, dg::forward);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"forward y derivative took        "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";
    M = dg::create::dy( grid, dg::centered);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"centered y derivative took       "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";
    M = dg::create::jumpX( grid);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"jump X took                      "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";
    t.tic();
    dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    if(rank==0)std::cout<<"AXPBY took                       "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";
    t.tic();
    dg::blas1::pointwiseDot( y, x, x);
    t.toc();
    if(rank==0)std::cout<<"pointwiseDot took                "<<t.diff()<<"s\t" <<gbytes/t.diff()<<"GB/s\n";
    t.tic();
    double norm = dg::blas2::dot( w2d, y);
    t.toc();
    if(rank==0)std::cout<<"DOT(w,y) took                    " <<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";
    t.tic();
    norm = dg::blas2::dot( x, w2d, y);
    t.toc();
    if(rank==0)std::cout<<"DOT(x,w,y) took                  " <<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";
    norm++;

    MPI_Finalize();
    return 0;
}
