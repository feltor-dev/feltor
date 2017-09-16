#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <mpi.h>

#include "elliptic.h"
#include "cg.h"
#include "backend/timer.cuh"
#include "backend/mpi_init.h"


const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double lz = 1.;

dg::bc bcx = dg::DIR;
double initial( double x, double y, double z) {return sin(0);}
double fct(double x, double y, double z){ return sin(y)*sin(x)*sin(2.*M_PI*z);}
double laplace_fct( double x, double y, double z) { return 2*sin(y)*sin(x)*sin(2.*M_PI*z);}


int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
    dg::mpi_init3d( bcx, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    double eps;
    if(rank==0)std::cout << "Type epsilon! \n";
    if(rank==0)std::cin >> eps;
    MPI_Bcast(  &eps,1 , MPI_DOUBLE, 0, comm);

    dg::CartesianMPIGrid3d grid( 0., lx, 0, ly, 0, lz, n, Nx, Ny,Nz, bcx, dg::PER,dg::PER, comm);
    const dg::MDVec w3d = dg::create::weights( grid);
    const dg::MDVec v3d = dg::create::inv_weights( grid);
    if(rank==0)std::cout<<"Expand initial condition\n";
    dg::MDVec x = dg::evaluate( initial, grid);

    if(rank==0)std::cout << "Create Laplacian\n";
    dg::Timer t;
    t.tic();
    dg::Elliptic<dg::CartesianMPIGrid3d, dg::MDMatrix, dg::MDVec> A ( grid, dg::not_normed); 
    t.toc();
    if(rank==0)std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::CG< dg::MDVec > pcg( x, n*n*Nx*Ny*Nz);
    if(rank==0)std::cout<<"Evaluate right hand side\n";
    const dg::MDVec solution = dg::evaluate ( fct, grid);
    dg::MDVec b = dg::evaluate ( laplace_fct, grid);
    //compute W b
    dg::blas2::symv( w3d, b, b);
    
    t.tic();
    int number = pcg( A, x, b, v3d, eps);
    t.toc();
    if( rank == 0)
    {
        std::cout << "# of pcg itersations   "<<number<<std::endl;
        std::cout << "... for a precision of "<< eps<<std::endl;
        std::cout << "...               took "<< t.diff()<<"s\n";
    }

    dg::MDVec  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w3d, error);
    double norm = dg::blas2::dot( w3d, solution);
    if( rank == 0) std::cout << "L2 Norm of relative error is:  " <<sqrt( normerr/norm)<<std::endl;

    MPI_Finalize();
    return 0;
}
