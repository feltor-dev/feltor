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
double function( double x, double y)   { return sin(x);}
double derivative( double x, double y) { return cos(x);}

dg::bc bcx = dg::DIR; 
dg::bc bcy = dg::PER;

typedef dg::RowDistMat<dg::SparseBlockMat, dg::NNCH> Matrix;
typedef dg::MPI_Vector<thrust::host_vector<double> > Vector;

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny; 
    MPI_Comm comm;
    mpi_init2d( bcx, bcy, n, Nx, Ny, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::MPI_Grid2d g( 0, lx, 0, lx, n, Nx, Ny, bcx, bcy, comm);

    Matrix dx = dg::create::dx( g, bcx, dg::forward);

    Vector func = dg::evaluate( function, g);
    Vector result( func);
    Vector deriv = dg::evaluate( derivative, g);

    Vector w2d = dg::create::weights(g);

    dg::blas2::symv( dx, func, result);

    dg::blas1::axpby( 1., deriv, -1., result);
    double error = sqrt(dg::blas2::dot(result, w2d, result));
    if(rank==0) std::cout << "DX: Distance to true solution: "<<error<<"\n";
;

    MPI_Finalize();
    return 0;
}
