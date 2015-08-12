#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "sparseblockmat.h"
#include "vector_traits.h"
#include "selfmade_blas.cuh"
#include "thrust_vector_blas.cuh"
#include "mpi_evaluation.h"
#include "mpi_derivatives.h"
#include "mpi_matrix.h"
#include "mpi_precon.h"
#include "mpi_init.h"

#include "blas.h"

const double lx = 2*M_PI;
/*
double function( double x, double y, double z) { return sin(3./4.*z);}
double derivative( double x, double y, double z) { return 3./4.*cos(3./4.*z);}
dg::bc bcz = dg::DIR_NEU;
*/
double function( double x, double y, double z)   { return sin(x);}
double derivative( double x, double y, double z) { return cos(x);}
double siny(   double x, double y, double z) { return sin(y);}
double cosy(   double x, double y, double z) { return cos(y);}
double sinz(   double x, double y, double z) { return sin(z);}
double cosz(   double x, double y, double z) { return cos(z);}

dg::bc bcx = dg::PER; 
dg::bc bcy = dg::PER;
dg::bc bcz = dg::PER;

typedef dg::RowDistMat<dg::SparseBlockMat, dg::NNCH> Matrix;
typedef dg::MPI_Vector<thrust::host_vector<double> > Vector;

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny,Nz; 
    MPI_Comm comm;
    mpi_init3d( bcx, bcy, bcz, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::MPI_Grid3d g( 0, lx, 0, lx,0,lx, n, Nx, Ny,Nz, bcx, bcy,bcz,dg::cartesian, comm);
    const Vector func = dg::evaluate( function, g);
    const Vector deriv = dg::evaluate( derivative, g);
    const Vector w2d = dg::create::weights(g);

    {
    Matrix dx = dg::create::dx( g, bcx, dg::centered);
    Vector result( func);
    dg::blas2::symv( dx, func, result);

    dg::blas1::axpby( 1., deriv, -1., result);
    double error = sqrt(dg::blas2::dot(result, w2d, result));
    if(rank==0) std::cout << "DX(symm): Distance to true solution: "<<error<<"\n";
    }
    {
    Matrix dx = dg::create::dx( g, bcx, dg::forward);
    Vector result( func);
    dg::blas2::symv( dx, func, result);

    dg::blas1::axpby( 1., deriv, -1., result);
    double error = sqrt(dg::blas2::dot(result, w2d, result));
    if(rank==0) std::cout << "DX(forw): Distance to true solution: "<<error<<"\n";
    }
    {
    Matrix dx = dg::create::dx( g, bcx, dg::backward);
    Vector result( func);
    dg::blas2::symv( dx, func, result);

    dg::blas1::axpby( 1., deriv, -1., result);
    double error = sqrt(dg::blas2::dot(result, w2d, result));
    if(rank==0) std::cout << "DX(back): Distance to true solution: "<<error<<"\n";
    }
    {
    Matrix dy = dg::create::dy( g, bcy, dg::centered);
    const Vector func = dg::evaluate( siny, g);
    const Vector deri = dg::evaluate( cosy, g);
    Vector result( func);
    dg::blas2::symv( dy, func, result);

    dg::blas1::axpby( 1., deri, -1., result);
    double error = sqrt(dg::blas2::dot(result, w2d, result));
    if(rank==0) std::cout << "DY(symm): Distance to true solution: "<<error<<"\n";
    }
    {
    Matrix dz = dg::create::dz( g, bcz, dg::centered);
    const Vector func = dg::evaluate( sinz, g);
    const Vector deri = dg::evaluate( cosz, g);
    Vector result( func);
    dg::blas2::symv( dz, func, result);

    dg::blas1::axpby( 1., deri, -1., result);
    double error = sqrt(dg::blas2::dot(result, w2d, result));
    if(rank==0) std::cout << "DZ(symm): Distance to true solution: "<<error<<"\n";
    }

    MPI_Finalize();
    return 0;
}
