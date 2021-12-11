#include <iostream>
#include <iomanip>
#include "mpi.h"

#include "elliptic.h"
#include "pcg.h"

#include "backend/mpi_init.h"

const double ly = 2.*M_PI;
const double lx = 2.*M_PI;

const double eps = 1e-6;

double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}

dg::bc bcx = dg::PER;

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny;
    MPI_Comm comm;
    dg::mpi_init2d( bcx, dg::PER, n, Nx, Ny, comm);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::MPIGrid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER, comm);
    const dg::MDVec w2d = dg::create::weights( grid);
    if( rank == 0) std::cout<<"Expand initial condition\n";
    dg::MDVec x = dg::evaluate( initial, grid);

    if( rank == 0) std::cout << "Create Laplacian\n";
    dg::Elliptic<dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec> A ( grid);
    dg::PCG< dg::MDVec > pcg( x, n*n*Nx*Ny);
    if( rank == 0) std::cout<<"Evaluate right hand side\n";
    dg::MDVec b = dg::evaluate ( laplace_fct, grid);
    const dg::MDVec solution = dg::evaluate ( fct, grid);

    int number = pcg.solve( A, x, b, 1., w2d, eps);
    if( rank == 0)
    {
        std::cout << "# of pcg itersations   "<<number<<std::endl;
        std::cout << "... for a precision of "<< eps<<std::endl;
    }

    dg::MDVec  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    dg::MDVec Ax(x), resi( b);
    dg::blas2::symv(  A, x, Ax);
    dg::blas1::axpby( 1.,Ax,-1.,resi);

    dg::exblas::udouble res;
    res.d = sqrt(dg::blas2::dot( w2d, x));
    if(rank==0)std::cout << "L2 Norm of x0 is              " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot(w2d , solution));
    if(rank==0)std::cout << "L2 Norm of Solution is        " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot(w2d , error));
    if(rank==0)std::cout << "L2 Norm of Error is           " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot( w2d, resi));
    if(rank==0)std::cout << "L2 Norm of Residuum is        " << res.d<<"\t"<<res.i << std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)

    MPI_Finalize();
    return 0;
}
