#include <iostream>
#include <mpi.h>

#include "dg/blas.h"
#include "dg/functors.h"
#include "dg/backend/mpi_init.h"

#include "mpi_projection.h"
#include "mpi_evaluation.h"
#include "mpi_weights.h"
#include "filter.h"

double function( double x, double y){return sin(x)*sin(y);}
double function( double x, double y, double z){return sin(x)*sin(y)*sin(z);}

unsigned n = 3, Nx = 8, Ny = 10, Nz = 6;

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    if(size!=4)
    {
        std::cerr << "Please run with 4 processes!\n";
        MPI_Finalize();
        return 0;
    }
    MPI_Comm comm;
    std::stringstream ss;
    ss<< "2 2 " << n << " "<<Nx<<" "<<Ny;
    mpi_init2d( dg::PER, dg::PER, n, Nx, Ny, comm, ss);
    MPI_Comm_rank( comm, &rank);

    {
    if(rank==0)std::cout << "Test 2d exponential filter: \n";
    dg::MPIGrid2d g3( -M_PI, 0, -5*M_PI, -4*M_PI, 3, Nx, Ny, comm);
    dg::MPIGrid2d g2( -M_PI, 0, -5*M_PI, -4*M_PI, 2, Nx, Ny, comm);

    const dg::MDVec vec = dg::evaluate( function, g3);
    const dg::MDVec weights = dg::create::weights( g3);
    dg::MDVec filtered_vec(vec), projected_vec(dg::evaluate( dg::zero, g2)), inter_vec( vec);
    dg::ModalFilter<dg::MDMatrix, dg::MDVec> filter( dg::ExponentialFilter(36, 0.5, 8, g3.nx()), g3);
    dg::MIDMatrix project = dg::create::projection( g2,g3);
    dg::MIDMatrix interpo = dg::create::interpolation( g3,g2);

    dg::blas2::symv( project, vec, projected_vec);
    dg::blas2::symv( interpo, projected_vec, inter_vec);
    filter( vec, filtered_vec);
    dg::blas1::axpby( 1., filtered_vec, -1., inter_vec);
    double error = sqrt(dg::blas2::dot( inter_vec, weights, inter_vec)/ dg::blas2::dot( vec, weights, vec));
    if(rank==0)std::cout << "Error by filtering: "<<error<<std::endl;

    if(rank==0){
    if( error > 1e-14)
        std::cout << "2D TEST FAILED!\n";
    else
        std::cout << "2D TEST PASSED!\n";
    }
    }
    MPI_Comm_free(&comm); // Segmentation fault without this line for unknown reasons!?
    {
    MPI_Comm comm;
    std::stringstream ss;
    ss<< "2 2 1 " << n << " "<<Nx<<" "<<Ny<<" "<<Nz;
    mpi_init3d( dg::PER, dg::PER, dg::PER, n, Nx, Ny, Nz, comm, ss);
    MPI_Comm_rank( comm, &rank);
    if(rank==0)std::cout << "Test 3d exponential filter: \n";
    dg::MPIGrid3d g3( -M_PI, 0, -5*M_PI, -4*M_PI, 0., 2.*M_PI, 3, Nx, Ny, Nz, comm);
    dg::MPIGrid3d g2( -M_PI, 0, -5*M_PI, -4*M_PI, 0., 2.*M_PI, 2, Nx, Ny, Nz, comm);

    const dg::MDVec vec = dg::evaluate( function, g3);
    const dg::MDVec weights = dg::create::weights( g3);
    dg::MDVec filtered_vec(vec), projected_vec(dg::evaluate( dg::zero, g2)), inter_vec( vec);
    dg::ModalFilter<dg::MDMatrix, dg::MDVec> filter( dg::ExponentialFilter(36, 0.5, 8, g3.nx()), g3);
    dg::MIDMatrix project = dg::create::projection( g2,g3);
    dg::MIDMatrix interpo = dg::create::interpolation( g3,g2);

    dg::blas2::symv( project, vec, projected_vec);
    dg::blas2::symv( interpo, projected_vec, inter_vec);
    filter( vec, filtered_vec);
    dg::blas1::axpby( 1., filtered_vec, -1., inter_vec);
    double error = sqrt(dg::blas2::dot( inter_vec, weights, inter_vec)/ dg::blas2::dot( vec, weights, vec));
    if(rank==0)std::cout << "Error by filtering: "<<error<<std::endl;

    if(rank==0){
    if( error > 1e-14)
        std::cout << "3D TEST FAILED!\n";
    else
        std::cout << "3D TEST PASSED!\n";
    }
    }
    MPI_Finalize();
    return 0;
}
