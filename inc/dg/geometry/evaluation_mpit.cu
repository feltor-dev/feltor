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
    unsigned n = 3, Nx = 10, Ny = 20, Nz = 100; 
    MPI_Comm comm2d, comm3d;
    mpi_init2d( dg::PER, dg::PER, comm2d);
    dg::MPIGrid2d g2d( 1, 2., 3, 4, n, Nx, Ny, dg::PER, dg::PER, comm2d);
    mpi_init3d( dg::PER, dg::PER, dg::PER, comm3d);
    dg::MPIGrid3d g3d( 1, 2, 3, 4, 5, 6, n, Nx, Ny, Nz, dg::PER, dg::PER, dg::PER, comm3d);

    //test evaluation and expand functions
    MDVec func2d = dg::blas1::transfer<MDVec>(dg::evaluate( function, g2d));
    MDVec func3d = dg::blas1::transfer<MDVec>(dg::evaluate( function, g3d));
    //test weights
    const MDVec w2d = dg::blas1::transfer<MDVec>(dg::create::weights(g2d));
    const MDVec w3d = dg::blas1::transfer<MDVec>(dg::create::weights(g3d));
    exblas::udouble res; 

    double integral2d = dg::blas1::dot( w2d, func2d); res.d = integral2d;
    if(rank==0)std::cout << "2D integral               "<<std::setw(6)<<integral2d <<"\t" << res.i << "\n";
    double sol2d = (exp(2.)-exp(1))*(exp(4.)-exp(3));
    if(rank==0)std::cout << "Correct integral is       "<<std::setw(6)<<sol2d<<std::endl;
    if(rank==0)std::cout << "Absolute 2d error is      "<<(integral2d-sol2d)<<"\n\n";

    double integral3d = dg::blas1::dot( w3d, func3d); res.d = integral3d;
    if(rank==0)std::cout << "3D integral               "<<std::setw(6)<<integral3d <<"\t" << res.i << "\n";
    double sol3d = sol2d*(exp(6.)-exp(5));
    if(rank==0)std::cout << "Correct integral is       "<<std::setw(6)<<sol3d<<std::endl;
    if(rank==0)std::cout << "Absolute 3d error is      "<<(integral3d-sol3d)<<"\n\n";

    double norm2d = dg::blas2::dot( w2d, func2d); res.d = norm2d;
    if(rank==0)std::cout << "Square normalized 2D norm "<<std::setw(6)<<norm2d<<"\t" << res.i <<"\n";
    double solution2d = (exp(4.)-exp(2))/2.*(exp(8.) -exp(6))/2.;
    if(rank==0)std::cout << "Correct square norm is    "<<std::setw(6)<<solution2d<<std::endl;
    if(rank==0)std::cout << "Relative 2d error is      "<<(norm2d-solution2d)/solution2d<<"\n\n";

    double norm3d = dg::blas2::dot( func3d, w3d, func3d); res.d = norm3d;
    if(rank==0)std::cout << "Square normalized 3D norm "<<std::setw(6)<<norm3d<<"\t" << res.i <<"\n";
    double solution3d = solution2d*(exp(12.)-exp(10))/2.;
    if(rank==0)std::cout << "Correct square norm is    "<<std::setw(6)<<solution3d<<std::endl;
    if(rank==0)std::cout << "Relative 3d error is      "<<(norm3d-solution3d)/solution3d<<"\n";

    //if(rank==0)
    //{
    //    int globalIdx, localIdx=0, PID, result=0;
    //    std::cout << "Type in global vector index: \n";
    //    std::cin >> globalIdx;
    //    if( g2d.global2localIdx( globalIdx, localIdx, PID) )
    //        std::cout <<"2d Local Index "<<localIdx<<" with rank "<<PID<<"\n";
    //    g2d.local2globalIdx( localIdx, PID, result);
    //    if( globalIdx !=  result)
    //        std::cerr <<"Inversion failed "<<result<<"\n";
    //    if( g3d.global2localIdx( globalIdx, localIdx, PID) )
    //        std::cout <<"3d Local Index "<<localIdx<<" with rank "<<PID<<"\n";
    //    g3d.local2globalIdx( localIdx, PID, result);
    //    if( globalIdx != result)
    //        std::cerr <<"Inversion failed "<<result<<"\n";
    //}

    MPI_Finalize();
    return 0;
} 
