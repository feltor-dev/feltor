#include <iostream>
#include <iomanip>

#include <mpi.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/timer.cuh"
#include "backend/mpi_evaluation.h"
#include "backend/mpi_derivatives.h"
#include "backend/mpi_init.h"

#include "cg.h"
#include "elliptic.h"


const double R_0 = 1000;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double lz = 2.*M_PI;
double fct( double x, double y, double z){ return sin(x-R_0)*sin(y);}
double derivative( double x, double y, double z){return cos(x-R_0)*sin(y);}
double laplace_fct( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y) + 2.*sin(y)*sin(x-R_0);}
dg::bc bcx = dg::DIR;
double initial( double x, double y, double z) {return sin(0);}


int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
    mpi_init3d( bcx, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);

    dg::MPI_Grid3d grid( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz, bcx, dg::PER, dg::PER, dg::cylindrical, comm);
    const dg::MPrecon w3d = dg::create::weights( grid);
    const dg::MPrecon v3d = dg::create::inv_weights( grid);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    double eps;
    if(rank==0)std::cout << "Type epsilon! \n";
    if(rank==0)std::cin >> eps;
    MPI_Bcast(  &eps,1 , MPI_DOUBLE, 0, comm);
    /////////////////////////////////////////////////////////////////
    if(rank==0)std::cout<<"TEST CYLINDRIAL LAPLACIAN!\n";
    dg::Timer t;
    dg::MVec x = dg::evaluate( initial, grid);

    if(rank==0)std::cout << "Create Laplacian\n";
    t.tic();
    dg::Elliptic<dg::MMatrix, dg::MVec, dg::MPrecon> laplace(grid);
    dg::MMatrix DX = dg::create::dx( grid);
    t.toc();
    if(rank==0)std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::CG< dg::MVec > pcg( x, n*n*Nx*Ny);

    if(rank==0)std::cout<<"Expand right hand side\n";
    const dg::MVec solution = dg::evaluate ( fct, grid);
    const dg::MVec deriv = dg::evaluate( derivative, grid);
    dg::MVec b = dg::evaluate ( laplace_fct, grid);
    //compute W b
    dg::blas2::symv( w3d, b, b);
    
    if(rank==0)std::cout << "For a precision of "<< eps<<" ..."<<std::endl;
    t.tic();
    unsigned number = pcg( laplace,x,b,v3d,eps);
    if(rank==0)std::cout << "Number of pcg iterations "<< number<<std::endl;
    t.toc();
    if(rank==0)std::cout << "... on the device took "<< t.diff()<<"s\n";
    dg::MVec  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w3d, error);
    double norm = dg::blas2::dot( w3d, solution);
    if(rank==0)std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr/norm)<<std::endl;
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w3d, error); 
    norm = dg::blas2::dot( w3d, deriv);
    if(rank==0)std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;
    //both function and derivative converge with order P 

    MPI_Finalize();
    return 0;
}
