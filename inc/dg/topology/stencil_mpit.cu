#include <iostream>
#include <mpi.h>

#include "dg/backend/mpi_init.h"
#include "mpi_evaluation.h"
#include "stencil.h"
#include "filter.h"
#include "../blas2.h"


int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    if(rank==0)std::cout << "Test window stencil\n";
    std::vector<dg::bc> bcs = {dg::DIR, dg::NEU, dg::PER};
    for( auto bc : bcs)
    {
        MPI_Comm comm2d;
        dg::mpi_init2d( bc, bc, comm2d);
        dg::MPIGrid2d g2d( 0,1, 0,2, 3, 4, 2, bc, bc, comm2d);
        auto x = dg::evaluate( dg::one, g2d), y(x);
        if(rank==0)std::cout << "Test "<<dg::bc2str( bc)<<" boundary:\n";
        auto stencil = dg::create::window_stencil( {3,3}, g2d, bc, bc);
        dg::blas2::symv( stencil, x, y);
        auto l2d = g2d.local();
        for( int r = 0; r<size; r++)
        {
            for( unsigned i=0; i<l2d.Ny()*l2d.ny(); i++)
            {
                for( unsigned k=0; k<l2d.Nx()*l2d.nx(); k++)
                    if(rank==r)std::cout << y.data()[i*l2d.Nx()*l2d.nx()+k] << " ";
                if(rank==r)std::cout<< std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if(rank==0)std::cout << "Test filtered symv\n";
        dg::blas2::stencil( dg::CSRSymvFilter(), stencil, x, y);
        for( int r = 0; r<size; r++)
        {
            for( unsigned i=0; i<l2d.Ny()*l2d.ny(); i++)
            {
                for( unsigned k=0; k<l2d.Nx()*l2d.nx(); k++)
                    if(rank==r)std::cout << y.data()[i*l2d.Nx()*l2d.nx()+k] << " ";
                if(rank==r)std::cout<< std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}
