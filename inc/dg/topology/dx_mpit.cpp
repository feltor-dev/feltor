#include <iostream>
#include <mpi.h>

#include "dg/backend/mpi_init.h"
#include "dg/blas.h"
#include "mpi_evaluation.h"
#include "mpi_derivatives.h"
#include "mpi_weights.h"
#include "derivativesT.h"


double function( double x) { return sin(x);}
double derivative( double x) { return cos(x);}
double zero( double x) { return 0;}

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    MPI_Comm comm1dPER, comm1d;
    dg::mpi_init1d( dg::PER, comm1dPER);
    dg::mpi_init1d( dg::DIR, comm1d);

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    unsigned n, N;
    dg::mpi_init1d( n, N, MPI_COMM_WORLD);
    dg::MPIGrid1d gPER( 0.1, 2*M_PI+0.1, n, N, dg::PER, comm1dPER);
    dg::MPIGrid1d gDIR( 0, M_PI, n, N, dg::DIR, comm1d);
    dg::MPIGrid1d gNEU( M_PI/2., 3*M_PI/2., n, N, dg::NEU, comm1d);
    dg::MPIGrid1d gDIR_NEU( 0, M_PI/2., n, N, dg::DIR_NEU, comm1d);
    dg::MPIGrid1d gNEU_DIR( M_PI/2., M_PI, n, N, dg::NEU_DIR, comm1d);
    dg::MPIGrid1d g[] = {gPER, gDIR, gNEU, gDIR_NEU,gNEU_DIR};

    DG_RANK0 std::cout << "TEST NORMAL TOPOLOGY: YOU SHOULD SEE CONVERGENCE FOR ALL OUTPUTS!!!\n";
    for( unsigned i=0; i<5; i++)
    {
        dg::MHMatrix hs = dg::create::dx( g[i], g[i].bcx(), dg::centered);
        dg::MHMatrix hf = dg::create::dx( g[i], g[i].bcx(), dg::forward);
        dg::MHMatrix hb = dg::create::dx( g[i], g[i].bcx(), dg::backward);
        dg::MHMatrix js = dg::create::jumpX( g[i], g[i].bcx());
        const dg::MHVec func = dg::evaluate( function, g[i]);
        dg::MHVec error = func;
        const dg::MHVec w1d = dg::create::weights( g[i]);
        const dg::MHVec deri = dg::evaluate( derivative, g[i]);
        const dg::MHVec null = dg::evaluate( zero, g[i]);

        dg::blas2::symv( hs, func, error);
        dg::blas1::axpby( 1., deri, -1., error);
        double err = sqrt( dg::blas2::dot( w1d, error));
        DG_RANK0 std::cout << "Distance to true solution (symmetric): "<<err<<"\n";
        dg::blas2::symv( hf, func, error);
        dg::blas1::axpby( 1., deri, -1., error);
        err = sqrt( dg::blas2::dot( w1d, error));
        DG_RANK0 std::cout << "Distance to true solution (forward  ): "<<err<<"\n";
        dg::blas2::symv( hb, func, error);
        dg::blas1::axpby( 1., deri, -1., error);
        err = sqrt( dg::blas2::dot( w1d, error));
        DG_RANK0 std::cout << "Distance to true solution (backward ): "<<err<<"\n";
        dg::blas2::symv( js, func, error);
        dg::blas1::axpby( 1., null , -1., error);
        err = sqrt( dg::blas2::dot( w1d, error));
        DG_RANK0 std::cout << "Distance to true solution (jump     ): "<<err<<"\n\n";
    }
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5
    MPI_Finalize();



    return 0;
}
