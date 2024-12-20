#include <iostream>
#include <sstream>
#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "dg/backend/transpose.h"
#include "dg/blas.h"
#include "dg/blas1.h"
#include "mpi_weights.h"
#include "mpi_projection.h"
#include "mpi_evaluation.h"
#include "fast_interpolation.h"

double sine( double x){ return sin(x);}
double sine( double x, double y){return sin(x)*sin(y);}
double sine( double x, double y, double z){return sin(x)*sin(y);}
//Actually this file is a test for fast_interpolation

template<class Matrix, class Grid>
void projection_test(const Matrix& proj, const Matrix& inte,
            const Grid& g2o, const Grid& g2n ,
            const dg::MHVec& w2do, const dg::MHVec& w2dn )
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    const dg::MHVec sinO( dg::evaluate( sine, g2o)),
                    sinN( dg::evaluate( sine, g2n));

    dg::MHVec sinP( sinN);
    dg::blas2::gemv( proj, sinO, sinP);

    double value0 = sqrt(dg::blas2::dot( sinO, w2do, sinO));
    if(rank==0)std::cout << "Original vector     "<<value0 << "\n";
    double value1 = sqrt(dg::blas2::dot( sinP, w2dn, sinP));
    if(rank==0)std::cout << "Projected vector    "<<value1 << "\n";
    if(rank==0)std::cout << "Difference in Norms "<<
                value0 - value1 << std::endl;
    dg::blas1::axpby( 1., sinN, -1., sinP);
    double value2 = sqrt( dg::blas2::dot( sinP, w2dn, sinP)/
                          dg::blas2::dot( sinN, w2dn, sinN));
    if(rank==0)std::cout << "Difference between projection and evaluation      "
                         <<value2<<"\n";

    dg::MHVec sinI( sinO);
    dg::blas2::gemv( inte, sinN, sinI);

    value0 = sqrt(dg::blas2::dot( sinN, w2dn, sinN));
    if(rank==0)std::cout << "Original vector     "<<value0 << "\n";
    value1 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
    if(rank==0)std::cout << "Interpolated vec    "<<value1 << "\n";
    if(rank==0)std::cout << "Difference in Norms "<<
                 value0 - value1 << "\n";

    dg::blas1::axpby( 1., sinO, -1., sinI);
    value2 = sqrt( dg::blas2::dot( sinI, w2do, sinI)/
                   dg::blas2::dot( sinO, w2do, sinO));
    if(rank==0)std::cout << "Difference between interpolation and evaluation   "
                         <<value2<<"\n"<<std::endl;
}


int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    int dims[3] = {0,0,0};
    MPI_Dims_create( size, 3, dims);
    std::stringstream ss;
    ss<< dims[0]<<" "<<dims[1]<<" "<<dims[2];
    MPI_Comm comm;
    mpi_init3d( dg::PER, dg::PER, dg::PER, comm, ss);

    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0)std::cout << "TEST 2D and 3D\n";
    unsigned n_old = 9, n_new = 3, N_old = 20, N_new = 10;
    if(rank==0)std::cout << "Fine   Grid "<< n_old << " x "<<N_old <<"\n";
    if(rank==0)std::cout << "Coarse Grid "<< n_new << " x "<<N_new <<"\n";
    dg::MPIGrid3d g2o (0, M_PI, 0, M_PI, 0,1, n_old, N_old, N_old, 4, comm);
    dg::MPIGrid3d g2n (0, M_PI, 0, M_PI, 0,1, n_new, N_new, N_new, 4, comm);
    const dg::MHVec w2do = dg::create::weights( g2o);
    const dg::MHVec w2dn = dg::create::weights( g2n);
    if(rank==0)std::cout << "TEST FAST PROJECT-INTERPOLATE\n";
    auto fast_proj2d = dg::create::fast_projection( g2o, n_old/n_new, N_old/N_new,
            N_old/N_new);
    auto fast_inte2d = dg::create::fast_interpolation( g2n, n_old/n_new,
            N_old/N_new, N_old/N_new);
    projection_test( fast_proj2d, fast_inte2d, g2o, g2n, w2do, w2dn);
    if(rank==0)std::cout << "TEST ORIGINAL PROJECT-INTERPOLATE\n";
    auto proj2d = dg::create::projection( g2n, g2o);
    auto inte2d = dg::create::interpolation( g2o, g2n);
    projection_test( proj2d, inte2d, g2o, g2n, w2do, w2dn);

    if(rank==0)std::cout << "TEST FAST TRANSFORM\n";
    auto forward = dg::create::fast_transform( dg::DLT<double>::forward(
                n_old), dg::DLT<double>::forward( n_old), g2o);
    auto backward = dg::create::fast_transform( dg::DLT<double>::backward(
                n_old), dg::DLT<double>::backward( n_old), g2o);
    const dg::MHVec sinO( dg::evaluate( sine, g2o));
    dg::MHVec sinF(sinO), sinI(sinO);
    dg::blas2::gemv( forward, sinO, sinF);
    dg::blas2::gemv( backward, sinF, sinI);
    dg::blas1::axpby( 1., sinO, -1., sinI);
    double value = sqrt(dg::blas2::dot( sinI, w2do, sinI));
    if(rank==0)std::cout << "Forward-Backward Error   "<<value
                         << " (Must be zero)\n" << std::endl;
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    if(rank==0)std::cout << "TEST correct mpi matrix conversion \n";
    for( auto method : {"dg", "linear", "cubic"})
    {
        if(rank==0) std::cout << method <<"\n";
        dg::MIHMatrix inte2d = dg::create::interpolation(
                g2o, g2n, method);
        dg::IHMatrix inte2dg = dg::create::interpolation(
                g2o.global(), g2n.global(), method);
        dg::MHVec sinP = dg::evaluate( sine, g2n);
        dg::blas2::gemv( inte2d, sinP, sinI);
        double value0 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
        dg::HVec sinNg = dg::evaluate( sine, g2n.global()), sinOg(dg::evaluate(
                    sine, g2o.global()));
        const dg::HVec w2dog = dg::create::weights( g2o.global());
        dg::blas2::gemv( inte2dg, sinNg, sinOg);
        double value1 = sqrt( dg::blas2::dot( sinOg, w2dog, sinOg));
        if(rank==0)std::cout << "MPI Interpolation: difference in Norms "<<value0 - value1 << "\n" << std::endl;

        dg::MIHMatrix project2d = dg::create::projection( g2n, g2o, method);
        dg::IHMatrix project2dg = dg::create::projection( g2n.global(),
                g2o.global(), method);
        sinI = dg::evaluate( sine, g2o);
        dg::blas2::gemv( project2d, sinI, sinP);
        value0 = sqrt(dg::blas2::dot( sinP, w2dn, sinP));
        sinOg = dg::evaluate( sine, g2o.global());
        const dg::HVec w2dng = dg::create::weights( g2n.global());
        dg::blas2::gemv( project2dg, sinOg, sinNg);
        value1 = sqrt( dg::blas2::dot( sinNg, w2dng, sinNg));
        if(rank==0)std::cout << "MPI Projection   : difference in Norms "<<value0 - value1 << "\n" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
