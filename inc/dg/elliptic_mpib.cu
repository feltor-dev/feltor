#include <iostream>
#include <iomanip>

#include <mpi.h>

#include "cg.h"
#include "elliptic.h"

#include "backend/timer.cuh"
#include "backend/mpi_init.h"


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
    dg::mpi_init3d( bcx, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);

    dg::CylindricalMPIGrid3d grid( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz, bcx, dg::PER, dg::PER, comm);
    const dg::MDVec w3d = dg::create::volume( grid);
    const dg::MDVec v3d = dg::create::inv_volume( grid);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    double eps=1e-6;
    if(rank==0)std::cout << "Type epsilon! \n";
    if(rank==0)std::cin >> eps;
    MPI_Bcast(  &eps,1 , MPI_DOUBLE, 0, comm);
    /////////////////////////////////////////////////////////////////
    if(rank==0)std::cout<<"TEST CYLINDRIAL LAPLACIAN!\n";
    dg::Timer t;
    dg::MDVec x = dg::evaluate( initial, grid);

    if(rank==0)std::cout << "Create Laplacian\n";
    t.tic();
    dg::Elliptic<dg::CylindricalMPIGrid3d, dg::MDMatrix, dg::MDVec> laplace(grid, dg::not_normed, dg::centered);
    dg::MDMatrix DX = dg::create::dx( grid);
    t.toc();
    if(rank==0)std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::CG< dg::MDVec > pcg( x, n*n*Nx*Ny);

    if(rank==0)std::cout<<"Expand right hand side\n";
    const dg::MDVec solution = dg::evaluate ( fct, grid);
    const dg::MDVec deriv = dg::evaluate( derivative, grid);
    dg::MDVec b = dg::evaluate ( laplace_fct, grid);
    //compute W b
    dg::blas2::symv( w3d, b, b);
    
    if(rank==0)std::cout << "For a precision of "<< eps<<" ..."<<std::endl;
    t.tic();
    unsigned number = pcg( laplace,x,b,v3d,eps);
    if(rank==0)std::cout << "Number of pcg iterations "<< number<<std::endl;
    t.toc();
    if(rank==0)std::cout << " took "<< t.diff()<<"s\n";
    dg::MDVec  error(  solution);
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
