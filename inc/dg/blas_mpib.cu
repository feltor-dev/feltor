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
#include "backend/fast_interpolation.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
double function(double x, double y, double z){ return sin(y)*sin(x);}

typedef dg::MDVec Vector;
typedef dg::MDMatrix Matrix;

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
    dg::mpi_init3d( dg::PER, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::MPIGrid3d grid( 0., lx, 0, ly,0., ly, n, Nx, Ny, Nz, comm);
    dg::MPIGrid3d grid_half = grid; grid_half.multiplyCellNumbers(0.5, 0.5);
    Vector w2d;
    dg::blas1::transfer( dg::create::weights(grid), w2d);
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
    dg::MultiMatrix<Matrix, Vector> inter, project; 
    dg::blas2::transfer(dg::create::fast_interpolation( grid_half, 2,2), inter);
    dg::blas2::transfer(dg::create::fast_projection( grid, 2,2), project);

    int multi=100;
    t.tic();
    double norm=0;
    for( int i=0; i<multi; i++)
        norm += dg::blas1::dot( w2d, x);
    t.toc();
    if(rank==0)std::cout<<"DOT took                         " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    Vector y(x), z(x), u(x), v(x);
    Matrix M;
    dg::blas2::transfer(dg::create::dx( grid, dg::centered), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"centered x derivative took       "<<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dx( grid, dg::forward), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"forward x derivative took        "<<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::forward), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"forward y derivative took        "<<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::centered), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"centered y derivative took       "<<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::jumpX( grid), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"jump X took                      "<<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    Vector x_half = dg::evaluate( dg::zero, grid_half);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( inter, x_half, x);
    t.toc();
    if(rank==0)std::cout<<"Interpolation half to full grid  "<<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( project, x, x_half);
    t.toc();
    if(rank==0)std::cout<<"Projection full to half grid     "<<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    if(rank==0)std::cout<<"AXPBY took                       "<<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpbypgz( 1., x, -1., y, 2., z);
    t.toc();
    if(rank==0)std::cout<<"AXPBYPGZ (1*x-1*y+2*z=z)         "<<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpbypgz( 1., x, -1., y, 3., x);
    t.toc();
    if(rank==0)std::cout<<"AXPBYPGZ (1*x-1.*y+3*x=x)        "<<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";

    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot(  y, x, x);
    t.toc();
    if(rank==0)std::cout<<"pointwiseDot (yx=x)              "<<t.diff()/multi<<"s\t" <<gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 1., y, x, 2.,u,v,0.,  z);
    t.toc();
    if(rank==0)std::cout<<"pointwiseDot (1*yx+2*uv=z)       "<<t.diff()/multi<<"s\t" <<gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 1., y, x, 2.,u,v,0.,  v);
    t.toc();
    if(rank==0)std::cout<<"pointwiseDot (1*yx+2*uv=v)       "<<t.diff()/multi<<"s\t" <<gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas2::dot( w2d, y);
    t.toc();
    if(rank==0)std::cout<<"DOT(w,y) took                    " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
    {
        norm += dg::blas2::dot( x, w2d, y);
    }
    t.toc();
    if(rank==0)std::cout<<"DOT(x,w,y) took                  " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    if(rank==0)std::cout<<norm<<std::endl;

    MPI_Finalize();
    return 0;
}
