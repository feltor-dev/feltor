#include <iostream>
#include <iomanip>

#include <mpi.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "backend/timer.cuh"
#include "backend/mpi_init.h"
#include "blas.h"
#include "geometry/mpi_evaluation.h"
#include "geometry/mpi_derivatives.h"
#include "geometry/mpi_weights.h"
#include "geometry/fast_interpolation.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
double function(double x, double y, double z){ return sin(y)*sin(x);}

using Vector = dg::MDVec;
using Matrix = dg::MDMatrix;
using ArrayVec = std::array<Vector, 3>;

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
   
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)std::cout << "This program is the MPI equivalent of blas_b. See blas_b for more information.\n";
    if(rank==0)std::cout << "Additional input parameters: \n";
    if(rank==0)std::cout << "    npx: # of processes in x (must divide Nx and total # of processes!\n";
    if(rank==0)std::cout << "    npy: # of processes in y (must divide Ny and total # of processes!\n";
    if(rank==0)std::cout << "    npz: # of processes in z (must divide Nz and total # of processes!\n";
    dg::mpi_init3d( dg::PER, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);

    dg::MPIGrid3d grid( 0., lx, 0, ly,0., ly, n, Nx, Ny, Nz, comm);
    dg::MPIGrid3d grid_half = grid; grid_half.multiplyCellNumbers(0.5, 0.5);
    Vector w2d;
    dg::blas1::transfer( dg::create::weights(grid), w2d);
    dg::Timer t;
    t.tic();
    ArrayVec x;
    dg::blas1::transfer( dg::evaluate( function, grid), x);
    t.toc();
    double gbytes=(double)x.size()*grid.size()*sizeof(double)/1e9;
    if(rank==0)std::cout << "Sizeof vectors is "<<gbytes<<" GB\n";
    dg::MultiMatrix<Matrix, ArrayVec> inter, project; 
    dg::blas2::transfer(dg::create::fast_interpolation( grid_half, 2,2), inter);
    dg::blas2::transfer(dg::create::fast_projection( grid, 2,2), project);

    int multi=100;
    ArrayVec y(x), z(x), u(x), v(x);
    Matrix M;
    dg::blas2::transfer(dg::create::dx( grid, dg::centered), M);
    dg::blas2::symv( M, x, y);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"centered x derivative took       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dx( grid, dg::forward), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"forward x derivative took        "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::forward), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"forward y derivative took        "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::centered), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"centered y derivative took       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::jumpX( grid), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    if(rank==0)std::cout<<"jump X took                      "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    ArrayVec x_half = dg::transfer<ArrayVec>(dg::evaluate( dg::zero, grid_half));
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( inter, x_half, x);
    t.toc();
    if(rank==0)std::cout<<"Interpolation quarter to full    "<<t.diff()/multi<<"s\t"<<3.75*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( project, x, x_half);
    t.toc();
    if(rank==0)std::cout<<"Projection full to quarter       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    if(rank==0)std::cout<<"AXPBY (1*y-1*x=x)                "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpbypgz( 1., x, -1., y, 2., z);
    t.toc();
    if(rank==0)std::cout<<"AXPBYPGZ (1*x-1*y+2*z=z)         "<<t.diff()/multi<<"s\t"<<4*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpbypgz( 1., x, -1., y, 3., x);
    t.toc();
    if(rank==0)std::cout<<"AXPBYPGZ (1*x-1.*y+3*x=x) (A)    "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot(  y, x, x);
    t.toc();
    if(rank==0)std::cout<<"pointwiseDot (yx=x) (A)          "<<t.diff()/multi<<"s\t" <<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 1., y, x, 2.,u,v,0.,  z);
    t.toc();
    if(rank==0)std::cout<<"pointwiseDot (1*yx+2*uv=z)       "<<t.diff()/multi<<"s\t" <<6*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 1., y, x, 2.,u,v,0.,  v);
    t.toc();
    if(rank==0)std::cout<<"pointwiseDot (1*yx+2*uv=v) (A)   "<<t.diff()/multi<<"s\t" <<5*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    double norm=0;
    for( int i=0; i<multi; i++)
        norm += dg::blas1::dot( x,y);
    t.toc();
    if(rank==0)std::cout<<"DOT1(x,y) took                   " <<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas2::dot( w2d, y);
    t.toc();
    if(rank==0)std::cout<<"DOT2(y,w,y) (A) took             " <<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
    {
        norm += dg::blas2::dot( x, w2d, y);
    }
    t.toc();
    if(rank==0)std::cout<<"DOT2(x,w,y) took                 " <<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n"; //DOT should be faster than axpby since it is only loading vectors and not writing them

    MPI_Finalize();
    return 0;
}
