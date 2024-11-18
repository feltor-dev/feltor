#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "dg/backend/mpi_init.h"
#include "dg/blas.h"
#include "average.h"
#include "mpi_evaluation.h"

const double lx = M_PI/2.;
const double ly = M_PI;
const double lz = M_PI/2.;
double function( double x, double y, double z) {return cos(x)*sin(z);}
double z_average( double x, double y) {return cos(x)*2./M_PI;}
double x_average( double x, double y, double z) {return sin(z)*2./M_PI;}

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    if(rank==0)std::cout << "Program to test the average in x and z direction\n";
    MPI_Comm comm;
    mpi_init3d( dg::PER, dg::PER, dg::PER, comm);
    unsigned n = 3, Nx = 32, Ny = 48, Nz = 64;
    //![doxygen]
    dg::MPIGrid3d g( 0, lx, 0, ly, 0, lz, n, Nx, Ny, Nz, comm);
    //g.display();

    dg::Average<dg::MIDMatrix, dg::MDVec > avg(g, dg::coo3d::z);

    const dg::MDVec vector = dg::evaluate( function ,g);
    dg::MDVec average_z;
    if(rank==0)std::cout << "Averaging z ... \n";
    avg( vector, average_z, false);
    //![doxygen]
    dg::MPIGrid2d gxy{ g.gx(), g.gy()};
    const dg::MDVec w2d = dg::create::weights( gxy);
    dg::MDVec solution = dg::evaluate( z_average, gxy);
    dg::blas1::axpby( 1., solution, -1., average_z);
    int64_t binary[] = {4406193765905047925,4395311848786989976};
    dg::exblas::udouble res;
    res.d = sqrt( dg::blas2::dot( average_z, w2d, average_z));
    if(rank==0)std::cout << "Distance to solution is: "<<res.d<<"\t"<<res.i<<std::endl;
    if(rank==0)std::cout << "(Converges with 2nd order).\n";
    if(rank==0)std::cout << "Averaging x ... \n";
    dg::Average<dg::MIDMatrix, dg::MDVec > tor(g, dg::coo3d::x);
    average_z = vector;
    tor( vector, average_z);
    solution = dg::evaluate( x_average, g);
    dg::blas1::axpby( 1., solution, -1., average_z);
    const dg::MDVec w3d = dg::create::weights( g);
    res.d = sqrt( dg::blas2::dot( average_z, w3d, average_z));
    if(rank==0)std::cout << "Distance to solution is: "<<res.d<<"\t"<<res.i-binary[1]<<std::endl;
    //if(rank==0)std::cout << "\n Continue with \n\n";

    MPI_Finalize();
    return 0;
}
