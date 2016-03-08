#include <iostream>
#include <iomanip>
#include <mpi.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "timer.cuh"
#include "sparseblockmat.h"
#include "vector_traits.h"
#include "selfmade_blas.cuh"
#include "thrust_vector_blas.cuh"
#include "mpi_evaluation.h"
#include "mpi_derivatives.h"
#include "mpi_matrix.h"
#include "mpi_precon.h"
#include "mpi_init.h"

#include "sparseblockmat.cuh"
#include "typedefs.cuh"
#include "blas.h"

const double lx = 2*M_PI;
double sinx(   double x, double y, double z) { return sin(x);}
double cosx(   double x, double y, double z) { return cos(x);}
double siny(   double x, double y, double z) { return sin(y);}
double cosy(   double x, double y, double z) { return cos(y);}
double sinz(   double x, double y, double z) { return sin(z);}
double cosz(   double x, double y, double z) { return cos(z);}

dg::bc bcx = dg::DIR; 
dg::bc bcy = dg::DIR;
dg::bc bcz = dg::DIR;

typedef dg::RowColDistMat<dg::EllSparseBlockMatDevice<double>, dg::CooSparseBlockMatDevice<double>, dg::NNCD> Matrix;
typedef dg::MPI_Vector<thrust::device_vector<double> > Vector;

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny,Nz; 
    MPI_Comm comm;
    mpi_init3d( bcx, bcy, bcz, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::MPI_Grid3d g( 0, lx, 0, lx,0,lx, n, Nx, Ny,Nz, bcx, bcy,bcz, comm);
    const Vector w3d = dg::create::weights(g);
    dg::Timer t;
    if(rank==0)std::cout << "TEST DX \n";
    {
    Matrix dx = dg::create::dx( g, bcx, dg::forward);
    Vector v = dg::evaluate( sinx, g);
    Vector w = v;
    const Vector u = dg::evaluate( cosx, g);

    t.tic();
    dg::blas2::symv( dx, v, w);
    t.toc();
    if(rank==0)std::cout << "Dx took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., u, -1., w);
    double tmp = dg::blas2::dot(w, w3d, w);
    if(rank==0)std::cout << "DX: Distance to true solution: "<<sqrt(tmp)<<"\n";
    }
    if(rank==0)std::cout << "TEST DY \n";
    {
    const Vector func = dg::evaluate( siny, g);
    const Vector deri = dg::evaluate( cosy, g);

    Matrix dy = dg::create::dy( g); 
    Vector temp( func);
    t.tic();
    dg::blas2::gemv( dy, func, temp);
    t.toc();
    if(rank==0)std::cout << "Dy took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., deri, -1.,temp);
    double tmp = dg::blas2::dot(temp, w3d, temp);
    if(rank==0)std::cout << "DY(1):           Distance to true solution: "<<sqrt(tmp)<<"\n";
    }
    if(rank==0)std::cout << "TEST DZ \n";
    {
    const Vector func = dg::evaluate( sinz, g);
    const Vector deri = dg::evaluate( cosz, g);

    Matrix dz = dg::create::dz( g); 
    Vector temp( func);
    t.tic();
    dg::blas2::gemv( dz, func, temp);
    t.toc();
    if(rank==0)std::cout << "Dz took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., deri, -1., temp);
    double tmp = dg::blas2::dot(temp, w3d, temp);
    if(rank==0)std::cout << "DZ(1):           Distance to true solution: "<<sqrt(tmp)<<"\n";
    }
    if(rank==0)std::cout << "JumpX and JumpY \n";
    {
    const Vector func = dg::evaluate( sinx, g);

    Matrix jumpX = dg::create::jumpX( g); 
    Matrix jumpY = dg::create::jumpY( g); 
    Matrix jumpZ = dg::create::jumpZ( g); 
    Vector temp( func);
    t.tic();
    dg::blas2::gemv( jumpX, func, temp);
    t.toc();
    if(rank==0)std::cout << "JumpX took "<<t.diff()<<"s\n";
    t.tic();
    dg::blas2::gemv( jumpY, func, temp);
    t.toc();
    if(rank==0)std::cout << "JumpY took "<<t.diff()<<"s\n";
    t.tic();
    dg::blas2::gemv( jumpZ, func, temp);
    t.toc();
    if(rank==0)std::cout << "JumpZ took "<<t.diff()<<"s\n";
    }

    MPI_Finalize();
    return 0;
}
