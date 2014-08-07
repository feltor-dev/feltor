#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include "backend/mpi_config.h"
#include "backend/timer.cuh"
#include "backend/mpi_evaluation.h"
#include "cg.h"
#include "backend/mpi_derivatives.h"

#include "backend/mpi_init.h"

//leo3 can do 350 x 350 but not 375 x 375
const double ly = 2.*M_PI;

const double eps = 1e-6; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

const double lx = 2.*M_PI;
double fct(double x, double y){ return sin(y)*sin(x);}
double derivative( double x, double y){return cos(x)*sin(y);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::DIR;
double initial( double x, double y) {return sin(0);}


int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny; 
    MPI_Comm comm;
    mpi_init2d( bcx, bcy, n, Nx, Ny, comm);

    dg::MPI_Grid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, bcy, comm);
    const dg::MPrecon w2d = dg::create::weights( grid);
    const dg::MPrecon v2d = dg::create::precond( grid);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank == 0) std::cout<<"Expand initial condition\n";
    dg::MVec x = dg::evaluate( initial, grid);

    if( rank == 0) std::cout << "Create symmetric Laplacian\n";
    dg::Timer t;
    t.tic();
    dg::MMatrix A = dg::create::laplacianM( grid, dg::not_normed, dg::forward); 
    t.toc();
    if( rank == 0) std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::CG< dg::MVec > pcg( x, n*n*Nx*Ny);
    if( rank == 0) std::cout<<"Expand right hand side\n";
    const dg::MVec solution = dg::evaluate ( fct, grid);
    const dg::MVec deriv = dg::evaluate( derivative, grid);
    dg::MVec b = dg::evaluate ( laplace_fct, grid);
    //compute W b
    dg::blas2::symv( w2d, b, b);
    //////////////////////////////////////////////////////////////////////
    
    t.tic();
    int number = pcg( A, x, b, v2d, eps);
    t.toc();
    if( rank == 0)
    {
        std::cout << "# of pcg itersations   "<<number<<std::endl;
        std::cout << "... for a precision of "<< eps<<std::endl;
        std::cout << "...               took "<< t.diff()<<"s\n";
    }

    dg::MVec  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w2d, error);
    double norm = dg::blas2::dot( w2d, solution);
    if( rank == 0) std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr/norm)<<std::endl;
    dg::MMatrix DX = dg::create::dx( grid);
    dg::MVec mod_solution = dg::evaluate ( fct, grid);
    dg::blas2::gemv( DX, mod_solution, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w2d, error); 
    norm = dg::blas2::dot( w2d, deriv);
    if( rank == 0) std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;

    MPI_Finalize();
    return 0;
}
