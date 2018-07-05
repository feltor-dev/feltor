#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <mpi.h>

#include "backend/timer.h"

#include "arakawa.h"
#include "backend/mpi_init.h"
#include "backend/typedefs.h"



const double lx = 2*M_PI;
const double ly = 2*M_PI;
//const double lx = 1.;
//const double ly = 1.;


//choose some mean function (attention on lx and ly)
//THESE ARE NOT PERIODIC
/*
double left( double x, double y) { return sin(x)*cos(y);}
double right( double x, double y){ return exp(0.1*(x+y)); }
double jacobian( double x, double y)
{
    return exp( x-M_PI)*(sin(x)+cos(x))*sin(y) * exp(y-M_PI)*sin(x)*(sin(y) + cos(y)) - sin(x)*exp(x-M_PI)*cos(y) * cos(x)*sin(y)*exp(y-M_PI);
}
*/

dg::bc bcx = dg::PER;
dg::bc bcy = dg::PER;
double left( double x, double y) {return sin(x)*cos(y);}
double right( double x, double y) {return cos(x)*sin(y);}
double jacobian( double x, double y)
{
    return cos(x)*cos(y)*cos(x)*cos(y) - sin(x)*sin(y)*sin(x)*sin(y);
}
////These are for comparing to FD arakawa results
//double left( double x, double y) {return sin(2.*M_PI*(x-hx/2.));}
//double right( double x, double y) {return y;}
//double jacobian( double x, double y) {return 2.*M_PI*cos(2.*M_PI*(x-hx/2.));}

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny;
    MPI_Comm comm;
    dg::mpi_init2d( bcx, bcy, n, Nx, Ny, comm);
    dg::MPIGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    dg::Timer t;
    dg::MDVec w2d = dg::create::weights( grid);
    dg::MDVec lhs = dg::evaluate ( left, grid), jac(lhs);
    dg::MDVec rhs = dg::evaluate ( right,grid);
    const dg::MDVec sol = dg::evaluate( jacobian, grid );
    dg::MDVec eins = dg::evaluate( dg::one, grid );
    std::cout<< std::setprecision(3);

    dg::ArakawaX<dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec> arakawa( grid);
    unsigned multi=1000;
    t.tic();
    for( unsigned i=0; i<multi; i++)
        arakawa( lhs, rhs, jac);
    t.toc();
    if(rank==0)std::cout << "Arakawa took "<<t.diff()*1000/(double)multi<<"ms\n\n";

    double result = dg::blas2::dot( eins, w2d, jac);
    if(rank==0)std::cout << std::scientific;
    if(rank==0)std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs,  w2d, jac);
    if(rank==0)std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs,  w2d, jac);
    if(rank==0)std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = sqrt( dg::blas2::dot( w2d, jac));
    if(rank==0)std::cout << "Distance to solution "<<result<<std::endl;


    MPI_Finalize();
    return 0;
}
