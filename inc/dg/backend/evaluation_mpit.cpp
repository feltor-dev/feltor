#include <iostream>
#include <cmath>

#include "mpi_evaluation.h"
#include "mpi_precon.h"

#include "blas.h"
#include "mpi_init.h"

double function( double x)
{
    return exp(x);
}

double function( double x, double y)
{
        return exp(x)*exp(y);
}
double function( double x, double y, double z)
{
        return exp(x)*exp(y)*exp(z);
}

const double lx = 2;
const double ly = 2;
const double lz = 2;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;

using namespace std;
int main(int argc, char** argv)
{
    MPI_Init( &argc, &argv);
    int np[2], rank;
    unsigned n, Nx, Ny; 
    MPI_Comm comm;
    mpi_init2d( dg::PER, dg::PER, np, n, Nx, Ny, comm);
    dg::MPI_Grid2d g( 0, lx, 0, lx, n, Nx, Ny, dg::PER, dg::PER, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    //test evaluation and expand functions
    dg::MVec func = dg::evaluate( function, g);
    //test preconditioners
    dg::blas2::symv( 1., dg::create::weights(g), func, 0., func);

    double norm = dg::blas2::dot( dg::create::precond(g), func);

    if(rank==0) cout << "Square normalized 2D norm "<< norm <<"\n";
    double solution2 = (exp(4.)-exp(0))/2.*(exp(4.) -exp(0))/2.;
    if(rank==0) cout << "Correct square norm is    "<<solution2<<endl;

    MPI_Finalize();
    return 0;
} 
