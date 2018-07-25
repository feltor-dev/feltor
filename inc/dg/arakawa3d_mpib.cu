#include <iostream>
#include <iomanip>

#include <mpi.h>
#include "arakawa.h"

#include "backend/mpi_init.h"
#include "backend/timer.h"



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
const double lz = 1.;
double left( double x, double y, double z) {return left(x,y)*z;}
double right( double x, double y, double z) {return right(x,y)*z;}
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y, double z)
{
    return jacobian(x,y)*z*z;
}

typedef dg::MDMatrix Matrix;
typedef dg::MDVec Vector;
int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz;
    MPI_Comm comm;
    mpi_init3d( bcx, bcy,dg::PER, n, Nx, Ny,Nz, comm);
    dg::MPIGrid3d grid( 0, lx, 0, ly, 0,lz, n, Nx, Ny, Nz, bcx, bcy, dg::PER, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    dg::Timer t;
    Vector w3d = dg::create::weights( grid);
    Vector lhs = dg::evaluate ( left, grid), jac(lhs);
    Vector rhs = dg::evaluate ( right,grid);
    const Vector sol = dg::evaluate( jacobian, grid );
    Vector eins = dg::evaluate( dg::one, grid );
    std::cout<< std::setprecision(3);

    dg::ArakawaX<dg::CartesianMPIGrid3d, Matrix, Vector> arakawa( grid);
    unsigned multi=20;
    t.tic();
    for( unsigned i=0; i<multi; i++)
        arakawa( lhs, rhs, jac);
    t.toc();
    if(rank==0) std::cout << "\nArakawa took "<<t.diff()*1000/(double)multi<<"ms\n\n";
    if(rank==0) std::cout <<   "which is     "<<t.diff()/0.02/Nz<<"ms per z plane \n\n";

    double result = dg::blas2::dot( eins, w3d, jac);
    std::cout << std::scientific;
    if(rank==0) std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs,  w3d, jac);
    if(rank==0) std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs,  w3d, jac);
    if(rank==0) std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = sqrt( dg::blas2::dot( w3d, jac));
    if(rank==0) std::cout << "Distance to solution "<<result<<std::endl;


    MPI_Finalize();
    return 0;
}
