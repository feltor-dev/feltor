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

typedef dg::MPI_Vector<thrust::host_vector<double> > MHVec;
typedef dg::MPI_Vector<thrust::device_vector<double> > MDVec;

int main(int argc, char** argv)
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm2d, comm3d;
    mpi_init2d( dg::PER, dg::PER, n, Nx, Ny, comm2d);
    dg::MPIGrid2d g2d( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER, comm2d);

    mpi_init3d( dg::PER, dg::PER, dg::PER, n, Nx, Ny, Nz, comm3d);
    dg::MPIGrid3d g3d( 0, lx, 0, ly, 0, lz, n, Nx, Ny, Nz, dg::PER, dg::PER, dg::PER, comm3d);

    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    //test evaluation and expand functions
    MDVec func2d = dg::evaluate( function, g2d);
    MDVec func3d = dg::evaluate( function, g3d);
    //test preconditioners
    const MDVec w2d = dg::create::weights(g2d);
    const MDVec w3d = dg::create::weights(g3d);
    const MDVec v2d = dg::create::inv_weights(g2d);
    const MDVec v3d = dg::create::inv_weights(g3d);
    dg::blas2::symv( 1., w2d, func2d, 0., func2d);
    dg::blas2::symv( 1., w3d, func3d, 0., func3d);

    double norm2d = dg::blas2::dot( v2d, func2d);
    double norm3d = dg::blas2::dot( v3d, func3d);

    if(rank==0)std::cout << "Square normalized 2D norm "<< norm2d <<"\n";
    double solution2 = (exp(4.)-exp(0))/2.*(exp(4.) -exp(0))/2.;
    if(rank==0)std::cout << "Correct square norm is    "<<solution2<<std::endl;
    if(rank==0)std::cout << "Relative 2d error is      "<<(norm2d-solution2)/solution2<<"\n\n";

    if(rank==0)std::cout << "Square normalized 3DXnorm "<< norm3d<<"\n";
    double solution = (exp(4.) -exp(0))/2.;
    double solution3 = solution2*solution;
    if(rank==0)std::cout << "Correct square norm is    "<<solution3<<std::endl;
    if(rank==0)std::cout << "Relative 3d error is      "<<(norm3d-solution3)/solution3<<"\n";

    if(rank==0)
    {
        int globalIdx, localIdx, PID, result;
        std::cout << "Type in global vector index: \n";
        std::cin >> globalIdx;
        if( g2d.global2localIdx( globalIdx, localIdx, PID) )
            std::cout <<"2d Local Index "<<localIdx<<" with rank "<<PID<<"\n";
        g2d.local2globalIdx( localIdx, PID, result);
        if( globalIdx !=  result)
            std::cerr <<"Inversion failed "<<result<<"\n";
        if( g3d.global2localIdx( globalIdx, localIdx, PID) )
            std::cout <<"3d Local Index "<<localIdx<<" with rank "<<PID<<"\n";
        g3d.local2globalIdx( localIdx, PID, result);
        if( globalIdx != result)
            std::cerr <<"Inversion failed "<<result<<"\n";
    }

    MPI_Finalize();
    return 0;
} 
