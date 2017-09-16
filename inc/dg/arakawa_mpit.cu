#include <iostream>
#include <iomanip>

#include <mpi.h>

#include "arakawa.h"
#include "backend/mpi_init.h"




//choose some mean function (attention on lx and ly)
/*
//THESE ARE NOT PERIODIC
double left( double x, double y) { return sin(x/2)*sin(x/2)*exp(x)*sin(y/2.)*sin(y/2.)*log(y+1); }
double right( double x, double y){ return sin(y/2.)*sin(y/2.)*exp(y)*sin(x/2)*sin(x/2)*log(x+1); }
*/
/*
double left( double x, double y) {return sin(x)*exp(x-M_PI)*sin(y);}
double right( double x, double y) {return sin(x)*sin(y);}
double right2( double x, double y) {return exp(y-M_PI);}
double jacobian( double x, double y) 
{
    return exp( x-M_PI)*(sin(x)+cos(x))*sin(y) * exp(y-M_PI)*sin(x)*(sin(y) + cos(y)) - sin(x)*exp(x-M_PI)*cos(y) * cos(x)*sin(y)*exp(y-M_PI); 
}
*/

/*
double left( double x, double y) {return sin(x)*cos(y);}
double right( double x, double y) {return sin(y)*cos(x);} 
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::PER; 
dg::bc bcy = dg::PER;
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y) 
{
    return cos(x)*cos(y)*cos(x)*cos(y) - sin(x)*sin(y)*sin(x)*sin(y); 
}
*/
////These are for comparing to FD arakawa results
//double left( double x, double y) {return sin(2.*M_PI*(x-hx/2.));}
//double right( double x, double y) {return y;}
//double jacobian( double x, double y) {return 2.*M_PI*cos(2.*M_PI*(x-hx/2.));}
const double lx = M_PI/2.;
const double ly = M_PI/2.;
double left( double x, double y) {return sin(x)*sin(y);}
double right( double x, double y) {return sin(2*x)*sin(2*y);} 
dg::bc bcx = dg::DIR_NEU; 
dg::bc bcy = dg::DIR_NEU;
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y) 
{
    return cos(x)*sin(y)*2*sin(2*x)*cos(2*y)-sin(x)*cos(y)*2*cos(2*x)*sin(2*y);
}


int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny; 
    MPI_Comm comm;
    dg::mpi_init2d( bcx, bcy, n, Nx, Ny, comm);
    dg::MPIGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::MHVec w2d = dg::create::weights( grid);
    if(rank==0)std::cout <<std::fixed<< std::setprecision(2)<<std::endl;
    dg::MHVec lhs = dg::evaluate( left, grid), jac(lhs);
    dg::MHVec rhs = dg::evaluate( right, grid);
    const dg::MHVec sol = dg::evaluate ( jacobian, grid);
    dg::MHVec eins = dg::evaluate( dg::one, grid);

    dg::ArakawaX<dg::CartesianMPIGrid2d, dg::MHMatrix, dg::MHVec> arakawa( grid);
    arakawa( lhs, rhs, jac);
    //if(rank==0) std::cout << lhs<<"\n";
    //if(rank==0) std::cout << rhs<<"\n";
    //if(rank==0) std::cout << jac<<"\n";


    double result = dg::blas2::dot( eins, w2d, jac);
    if(rank==0)std::cout << std::scientific;
    if(rank==0)std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs,  w2d, jac);
    if(rank==0)std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs,  w2d, jac);
    if(rank==0)std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = sqrt( dg::blas2::dot( w2d, jac));
    if(rank==0)std::cout << "Distance to solution "<<result<<std::endl; //don't forget sqrt when comuting errors
    MPI_Finalize();
    return 0;
}

