#pragma once

namespace dg{
typedef MPI_Vector MVec;
typedef MPI_Matrix MMatrix;
typedef MPI_Precon MPrecon;
}

void mpi_init2d( dg::bc bcx, dg::bc bcy, int np[], unsigned& n, unsigned& Nx, unsigned& Ny, MPI_Comm& comm  )
{
    int periods[2] = {false,false};
    if( bcx == dg::PER) periods[0] = true;
    if( bcy == dg::PER) periods[1] = true;
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
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);
    if( rank == 0)
    {
        std::cout << "Type n, Nx and Ny\n";
        std::cin >> n >> Nx >> Ny;
    }
    MPI_Bcast(  &n,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast( &Nx,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast( &Ny,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);

}
