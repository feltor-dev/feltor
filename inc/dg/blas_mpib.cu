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

typedef dg::MDVec Vector;
typedef dg::MDMatrix Matrix;

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny; 
    MPI_Comm comm;
    dg::mpi_init2d( dg::PER, dg::PER, n, Nx, Ny, comm);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::MPIGrid2d grid( 0., lx, 0, ly, n, Nx, Ny, comm);
    Vector w2d;
    dg::blas1::transfer( dg::create::weights(grid), w2d);
    if(rank==0)std::cout<<"Evaluate a function on the grid\n";
    if(rank==0)std::cout<<"Evaluate a function on the grid\n";
    dg::Timer t;
    t.tic();
    Vector x;
    dg::blas1::transfer( dg::evaluate( function, grid), x);
    t.toc();
    if(rank==0)std::cout<<"Evaluation of a function took    "<<t.diff()<<"s\n";
    if(rank==0)std::cout << "Sizeof value type is "<<sizeof(double)<<"\n";
    double gbytes=(double)grid.global().size()*sizeof(double)/1e9;
    if(rank==0)std::cout << "Sizeof vectors is "<<gbytes<<" GB\n";
    t.tic();
    double norm=0;
    for( unsigned i=0; i<20; i++)
        norm += dg::blas1::dot( w2d, x);
    t.toc();
    if(rank==0)std::cout<<"DOT took                         " <<t.diff()/20.<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";
    Vector y(x);
    Matrix M;
    dg::blas2::transfer(dg::create::dx( grid, dg::centered), M);
    t.tic();
    for( int i=0; i<20; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"centered x derivative took       "<<t.diff()/20.<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dx( grid, dg::forward), M);
    t.tic();
    for( int i=0; i<20; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"forward x derivative took        "<<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::forward), M);
    t.tic();
    for( int i=0; i<20; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"forward y derivative took        "<<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::centered), M);
    t.tic();
    for( int i=0; i<20; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"centered y derivative took       "<<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::jumpX( grid), M);
    t.tic();
    for( int i=0; i<20; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"jump X took                      "<<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    t.tic();
    for( int i=0; i<20; i++)
        dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    if(rank==0)std::cout<<"AXPBY took                       "<<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    t.tic();
    for( int i=0; i<20; i++)
        dg::blas1::pointwiseDot( y, x, x);
    t.toc();
    if(rank==0)std::cout<<"pointwiseDot took                "<<t.diff()/20<<"s\t" <<gbytes*20/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<20; i++)
        norm += dg::blas2::dot( w2d, y);
    t.toc();
    if(rank==0)std::cout<<"DOT(w,y) took                    " <<t.diff()/20.<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    t.tic();
    for( int i=0; i<20; i++)
    {
        norm += dg::blas2::dot( x, w2d, y);
    }
    t.toc();
    if(rank==0)std::cout<<"DOT(x,w,y) took                  " <<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";
    if(rank==0)std::cout<<norm<<std::endl;

    MPI_Finalize();
    return 0;
}
