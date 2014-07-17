#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include "timer.cuh"
#include "mpi_evaluation.h"

#include "cg.h"

//leo3 can do 350 x 350 but not 375 x 375
const double ly = 2.*M_PI;

const double eps = 1e-6; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

const double lx = M_PI;
double fct(double x, double y){ return sin(y)*sin(x);}
double derivative( double x, double y){return cos(x)*sin(y);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
dg::bc bcx = dg::DIR;
double initial( double x, double y) {return sin(0);}

namespace dg{
typedef MPI_Vector MVec;
typedef MPI_Matrix MMatrix;
typedef MPI_Precon MPrecon;
}


int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int np[2];
    int periods[2] = {0,1};
    if( bcx == dg::PER) periods[0] = 1;
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    if( rank == 0)
    {
        std::cout << "Type npx and npy\n";
        std::cin >> np[0] >> np[1];
        std::cout<< "You typed "<<np[0] <<" and "<<np[1]<<std::endl;
        std::cout << "Size is "<<size<<std::endl;
        assert( size == np[0]*np[1]);
    }
    MPI_Bcast( np, 2, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);
    unsigned n, Nx, Ny; 
    if( rank == 0)
    {
        std::cout << "Type n, Nx and Ny\n";
        std::cin >> n >> Nx >> Ny;
    }
    MPI_Bcast(  &n,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast( &Nx,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast( &Ny,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    dg::MPI_Grid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER, comm);
    const dg::MPrecon w2d = dg::create::weights( grid);
    const dg::MPrecon v2d = dg::create::precond( grid);
    if( rank == 0)
        std::cout<<"Expand initial condition\n";
    dg::MVec x = dg::evaluate( initial, grid);

    if( rank == 0)
        std::cout << "Create symmetric Laplacian\n";
    dg::Timer t;
    t.tic();
    dg::MMatrix A = dg::create::laplacianM( grid, dg::not_normed, dg::forward); 
    t.toc();
    if( rank == 0)
        std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::CG< dg::MVec > pcg( x, n*n*Nx*Ny);
    if( rank == 0)
        std::cout<<"Expand right hand side\n";
    const dg::MVec solution = dg::evaluate ( fct, grid);
    const dg::MVec deriv = dg::evaluate( derivative, grid);
    dg::MVec b = dg::evaluate ( laplace_fct, grid);
    //compute W b
    dg::blas2::symv( w2d, b, b);
    //////////////////////////////////////////////////////////////////////
    if( rank == 0)
    {
        std::cout << "# of polynomial coefficients: "<< n <<std::endl;
        std::cout << "# of 2d cells                 "<< Nx*Ny <<std::endl;
    }
    
    t.tic();
    int number = pcg( A, x, b, v2d, eps);
    t.toc();
    if( rank == 0)
    {
        std::cout << "# of pcg itersations   "<<number<<std::endl;
        std::cout << "... for a precision of "<< eps<<std::endl;
        std::cout << "... on the device took "<< t.diff()<<"s\n";
    }

    dg::MVec  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w2d, error);
    double norm = dg::blas2::dot( w2d, solution);
    if( rank == 0)
        std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr/norm)<<std::endl;
    dg::MMatrix DX = dg::create::dx( grid);
    dg::MVec mod_solution = dg::evaluate ( fct, grid);
    dg::blas2::gemv( DX, mod_solution, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w2d, error); 
    norm = dg::blas2::dot( w2d, deriv);
    if( rank == 0)
        std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;

    return 0;
}
