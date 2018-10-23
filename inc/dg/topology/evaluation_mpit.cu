#include <iostream>
#include <iomanip>
#include <cmath>

#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "dg/blas.h"
#include "mpi_evaluation.h"
#include "mpi_weights.h"


double function( double x)
{
    return exp(x);
}

double function( double x, double y)
{
        return exp(x)*exp(y);
}
double function( double x, double y, double z)
{
        return exp(x)*exp(y)*exp(z);
}

typedef dg::MPI_Vector<thrust::host_vector<double> > MHVec;
typedef dg::MPI_Vector<thrust::device_vector<double> > MDVec;

int main(int argc, char** argv)
{
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)std::cout << "This program tests the exblas::dot function. The tests succeed only if the evaluation and grid functions but also the weights and especially the exblas::dot function are correctly implemented and compiled. Furthermore, the compiler implementation of the exp function in the math library must be consistent across platforms to get reproducible results.\n";
    if(rank==0)std::cout << "A TEST is PASSED if the number in the second column shows EXACTLY 0!\n";
    unsigned n = 3, Nx = 12, Ny = 28, Nz = 100;
    if(rank==0)std::cout << "On Grid "<<n<<" x "<<Nx<<" x "<<Ny<<" x "<<Nz<<"\n";
    MPI_Comm comm2d, comm3d;
    mpi_init2d( dg::PER, dg::PER, comm2d);
    dg::MPIGrid2d g2d( 1, 2., 3, 4, n, Nx, Ny, dg::PER, dg::PER, comm2d);
    mpi_init3d( dg::PER, dg::PER, dg::PER, comm3d);
    dg::MPIGrid3d g3d( 1, 2, 3, 4, 5, 6, n, Nx, Ny, Nz, dg::PER, dg::PER, dg::PER, comm3d);

    //test evaluation and expand functions
    MDVec func2d = dg::construct<MDVec>(dg::evaluate( function, g2d));
    MDVec func3d = dg::construct<MDVec>(dg::evaluate( function, g3d));
    //test weights
    const MDVec w2d = dg::construct<MDVec>(dg::create::weights(g2d));
    const MDVec w3d = dg::construct<MDVec>(dg::create::weights(g3d));
    exblas::udouble res;

    double integral2d = dg::blas1::dot( w2d, func2d); res.d = integral2d;
    if(rank==0)std::cout << "2D integral               "<<std::setw(6)<<integral2d <<"\t" << res.i - 4639875759346476257 << "\n";
    double sol2d = (exp(2.)-exp(1))*(exp(4.)-exp(3));
    if(rank==0)std::cout << "Correct integral is       "<<std::setw(6)<<sol2d<<std::endl;
    if(rank==0)std::cout << "Absolute 2d error is      "<<(integral2d-sol2d)<<"\n\n";

    double integral3d = dg::blas1::dot( w3d, func3d); res.d = integral3d;
    if(rank==0)std::cout << "3D integral               "<<std::setw(6)<<integral3d <<"\t" << res.i - 4675882723962622631<< "\n";
    double sol3d = sol2d*(exp(6.)-exp(5));
    if(rank==0)std::cout << "Correct integral is       "<<std::setw(6)<<sol3d<<std::endl;
    if(rank==0)std::cout << "Absolute 3d error is      "<<(integral3d-sol3d)<<"\n\n";

    double norm2d = dg::blas2::dot( w2d, func2d); res.d = norm2d;
    if(rank==0)std::cout << "Square normalized 2D norm "<<std::setw(6)<<norm2d<<"\t" << res.i - 4674091193523851724<<"\n";
    double solution2d = (exp(4.)-exp(2))/2.*(exp(8.) -exp(6))/2.;
    if(rank==0)std::cout << "Correct square norm is    "<<std::setw(6)<<solution2d<<std::endl;
    if(rank==0)std::cout << "Relative 2d error is      "<<(norm2d-solution2d)/solution2d<<"\n\n";

    double norm3d = dg::blas2::dot( func3d, w3d, func3d); res.d = norm3d;
    if(rank==0)std::cout << "Square normalized 3D norm "<<std::setw(6)<<norm3d<<"\t" << res.i - 4746764681002108278<<"\n";
    double solution3d = solution2d*(exp(12.)-exp(10))/2.;
    if(rank==0)std::cout << "Correct square norm is    "<<std::setw(6)<<solution3d<<std::endl;
    if(rank==0)std::cout << "Relative 3d error is      "<<(norm3d-solution3d)/solution3d<<"\n";
    if(rank==0)std::cout << "\nFINISHED! Continue with topology/derivatives_mpit.cu !\n\n";

    MPI_Finalize();
    return 0;
}
