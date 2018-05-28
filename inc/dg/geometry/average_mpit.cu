#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "dg/backend/mpi_init.h"
#include "dg/blas.h"
#include "average_mpi.h"
#include "mpi_evaluation.h"

const double lx = M_PI/2.;
const double ly = M_PI;
double function( double x, double y) {return cos(x)*sin(y);}
double pol_average( double x, double y) {return cos(x)*2./M_PI;}
double tor_average( double x, double y) {return sin(y)*2./M_PI;}

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    if(rank==0)std::cout << "Program to test the average in x and y direction\n";
    MPI_Comm comm;
    mpi_init2d( dg::PER, dg::PER, comm);
    unsigned n = 3, Nx = 32, Ny = 48;
    //![doxygen]
    dg::MPIGrid2d g( 0, lx, 0, ly, n, Nx, Ny, comm);

    dg::Average<dg::MDVec > pol(g, dg::coo2d::y);

    const dg::MDVec vector = dg::evaluate( function ,g);
    dg::MDVec average_y( vector);
    if(rank==0)std::cout << "Averaging y ... \n";
    pol( vector, average_y);
    //![doxygen]
    const dg::MDVec w2d = dg::create::weights( g);
    dg::MDVec solution = dg::evaluate( pol_average, g);
    dg::blas1::axpby( 1., solution, -1., average_y);
    int64_t binary[] = {4406193765905047925,4395311848786989976};
    exblas::udouble res;
    res.d = sqrt( dg::blas2::dot( average_y, w2d, average_y));
    if(rank==0)std::cout << "Distance to solution is: "<<res.d<<"\t"<<res.i-binary[0]<<std::endl;
    if(rank==0)std::cout << "Averaging x ... \n";
    dg::Average< dg::MDVec> tor( g, dg::coo2d::x);
    tor( vector, average_y);
    solution = dg::evaluate( tor_average, g);
    dg::blas1::axpby( 1., solution, -1., average_y);
    res.d = sqrt( dg::blas2::dot( average_y, w2d, average_y));
    if(rank==0)std::cout << "Distance to solution is: "<<res.d<<"\t"<<res.i-binary[1]<<std::endl;
    //if(rank==0)std::cout << "\n Continue with \n\n";

    MPI_Finalize();
    return 0;
}
