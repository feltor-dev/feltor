#include <iostream>
#include <cmath>

#include <mpi.h>
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
    int rank;
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm2d, comm3d;
    mpi_init2d( dg::PER, dg::PER, n, Nx, Ny, comm2d);
    mpi_init3d( dg::PER, dg::PER, dg::PER, n, Nx, Ny, Nz, comm3d);
    dg::MPI_Grid2d g2d( 0, lx, 0, lx, n, Nx, Ny, dg::PER, dg::PER, comm2d);
    dg::MPI_Grid3d g3d( 0, lx, 0, lx, 0, lz, n, Nx, Ny, Nz, dg::PER, dg::PER, dg::PER, dg::cylindrical, comm3d);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    //test evaluation and expand functions
    dg::MVec func2d = dg::evaluate( function, g2d);
    dg::MVec func3d = dg::evaluate( function, g3d);
    //test preconditioners
    dg::blas2::symv( 1., dg::create::weights(g2d), func2d, 0., func2d);
    dg::blas2::symv( 1., dg::create::weights(g3d), func3d, 0., func3d);

    double norm2d = dg::blas2::dot( dg::create::inv_weights(g2d), func2d);
    double norm3d = dg::blas2::dot( dg::create::inv_weights(g3d), func3d);

    if(rank==0) cout << "Square normalized 2D norm "<< norm2d <<"\n";
    double solution2 = (exp(4.)-exp(0))/2.*(exp(4.) -exp(0))/2.;
    if(rank==0) cout << "Correct square norm is    "<<solution2<<endl;

    if(rank==0) cout << "Square normalized 3DXnorm   "<< norm3d<<"\n";
    double solution = (exp(4.) -exp(0))/2.;
    if( g3d.system() == dg::cylindrical)
        solution = (3*exp(4.)+1.)/4.;
    double solution3 = solution2*solution;
    if(rank==0)cout << "Correct square norm is      "<<solution3<<endl;
    if(rank==0)cout << "Relative 3d error is        "<<(norm3X-solution3)/solution3<<"\n";
    MPI_Finalize();
    return 0;
} 
