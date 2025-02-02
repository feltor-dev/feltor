#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "dg/backend/timer.h"
#include "dg/backend/mpi_init.h"
#include "dg/blas.h"

#include "mpi_evaluation.h"
#include "average.h"

const double lx = 2.*M_PI;
const double ly = M_PI;

double function( double x, double y) {return cos(x)*sin(y);}
double pol_average( double x, double y) {return cos(x)*2./M_PI;}

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny;

    MPI_Comm comm;
    mpi_init2d( dg::PER, dg::PER, n, Nx, Ny, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::MPIGrid2d g( 0, lx, 0, ly, n, Nx, Ny, comm);
    dg::Timer t;

    dg::Average<dg::MIDMatrix,dg::MDVec > pol(g, dg::coo2d::y);
    dg::MDVec vector = dg::evaluate( function ,g), average_y( vector);
    const dg::MDVec solution = dg::evaluate( pol_average, g);
    t.tic();
    for( unsigned i=0; i<100; i++)
        pol( vector, average_y);
    t.toc();
    if(rank==0)std::cout << "Assembly of average (simple) vector took:      "<<t.diff()/100.<<"s\n";

    dg::blas1::axpby( 1., solution, -1., average_y, vector);
    dg::MDVec w2d = dg::create::weights(g);
    double norm = dg::blas2::dot(vector, w2d, vector);
    if(rank==0)std::cout << "Distance to solution is: "<<        sqrt(norm)<<std::endl;

    MPI_Finalize();
    return 0;
}
