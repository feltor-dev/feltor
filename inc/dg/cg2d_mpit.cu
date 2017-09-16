#include <iostream>
#include <iomanip>
#include "mpi.h"

#include "elliptic.h"
#include "cg.h"

#include "backend/mpi_init.h"

const double ly = 2.*M_PI;
const double lx = 2.*M_PI;

const double eps = 1e-6; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double fct(double x, double y){ return sin(y)*sin(x);}
double derivative( double x, double y){return cos(x)*sin(y);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}

dg::bc bcx = dg::PER;

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny; 
    MPI_Comm comm;
    dg::mpi_init2d( bcx, dg::PER, n, Nx, Ny, comm);

    dg::MPIGrid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER, comm);
    const dg::MDVec w2d = dg::create::weights( grid);
    const dg::MDVec v2d = dg::create::inv_weights( grid);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank == 0) std::cout<<"Expand initial condition\n";
    dg::MDVec x = dg::evaluate( initial, grid);

    if( rank == 0) std::cout << "Create symmetric Laplacian\n";
    dg::Elliptic<dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec> A ( grid, dg::not_normed); 

    dg::CG< dg::MDVec > pcg( x, n*n*Nx*Ny);
    if( rank == 0) std::cout<<"Expand right hand side\n";
    const dg::MDVec solution = dg::evaluate ( fct, grid);
    const dg::MDVec deriv = dg::evaluate( derivative, grid);
    dg::MDVec b = dg::evaluate ( laplace_fct, grid);
    //compute W b
    dg::blas2::symv( w2d, b, b);
    //////////////////////////////////////////////////////////////////////
    
    int number = pcg( A, x, b, v2d, eps);
    if( rank == 0)
    {
        std::cout << "# of pcg itersations   "<<number<<std::endl;
        std::cout << "... for a precision of "<< eps<<std::endl;
    }

    dg::MDVec  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w2d, error);
    double norm = dg::blas2::dot( w2d, solution);
    if( rank == 0) std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr/norm)<<std::endl;
    dg::MDMatrix DX = dg::create::dx( grid);
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w2d, error); 
    norm = dg::blas2::dot( w2d, deriv);
    if( rank == 0) std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;

    MPI_Finalize();
    return 0;
}
