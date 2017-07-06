#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "blas.h"
#include "mpi_evaluation.h"
#include "mpi_derivatives.h"
#include "mpi_matrix.h"
#include "mpi_precon.h"
#include "mpi_init.h"
#include "../average.h"

const double lx = 2.*M_PI;
const double ly = M_PI;

double function( double x, double y) {return cos(x)*sin(y);}
double pol_average( double x, double y) {return cos(x)*2./M_PI;}

dg::bc bcx = dg::PER; 
dg::bc bcy = dg::PER;

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny; 

    MPI_Comm comm;
    mpi_init2d( bcx, bcy, n, Nx, Ny, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::MPIGrid2d g( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy, comm);
    

    if(rank==0)std::cout << "constructing polavg" << std::endl;
    dg::PoloidalAverage<dg::MDVec, dg::MDVec > pol(g);
    if(rank==0)std::cout << "constructing polavg end" << std::endl;
    dg::MDVec vector = dg::evaluate( function ,g), average_y( vector);
    const dg::MDVec solution = dg::evaluate( pol_average, g);
    if(rank==0)std::cout << "Averaging ... \n";
    pol( vector, average_y);
    dg::blas1::axpby( 1., solution, -1., average_y, vector);

    dg::MDVec w2d = dg::create::weights(g);
    double norm = dg::blas2::dot(vector, w2d, vector);
    if(rank==0)std::cout << "Distance to solution is: "<<        sqrt(norm)<<std::endl;

    MPI_Finalize();
    return 0;
}
