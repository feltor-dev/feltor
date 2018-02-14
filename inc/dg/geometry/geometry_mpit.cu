#include <iostream>

#include <cusp/print.h>

#include <mpi.h>
#include "geometry.h"
#include "../blas2.h"


double R_0 = 4.*M_PI;


double sine( double R, double Z,double phi){ return sin(R-R_0)*sin(Z)*sin(phi)/sqrt(R);}

namespace dg { typedef dg::MPI_Vector<thrust::device_vector<double> > MDVec; }

//TEST geometry.h for every container and geometry that you want to use
int main( int argc, char* argv[] )
{
    MPI_Init(&argc, &argv);
    int np[3];
    int periods[3] = {0,0,0};
    periods[0] = 1;
    periods[1] = 1;
    periods[2] = 1;
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank == 0)
    {
        std::cout << "Type npx, npy and npz\n";
        std::cin >> np[0] >> np[1] >> np[2];
        std::cout<< "You typed "<<np[0] <<" and "<<np[1]<<" and "<<np[2]<<std::endl;
    }
    MPI_Bcast( np, 3, MPI_INT, 0, MPI_COMM_WORLD);

    int size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    if( rank == 0)
    {
        std::cout << "Size is "<<size<<std::endl;
        assert( size == np[0]*np[1]*np[2]);
    }

    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 3, np, periods, true, &comm);
    dg::CylindricalMPIGrid3d grid( R_0 , R_0+ 2.*M_PI, 0.,2.*M_PI, 0., 2.*M_PI,  3,32,24,16, dg::PER, dg::PER, dg::PER, comm);
    dg::SparseElement<dg::MDVec> vol = dg::tensor::volume(grid.metric());

    dg::MDVec b = dg::evaluate( sine, grid);
    dg::MDVec vol3d = dg::create::volume( grid);
    double test = dg::blas2::dot( b, vol3d, b);
    double sol = M_PI*M_PI*M_PI;
    if(rank==0)std::cout << "Test of volume:         "<<test<< " sol = "<<sol<<"\t";
    if(rank==0)std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";
    dg::MDVec temp = dg::create::weights( grid);
    dg::tensor::pointwiseDot( temp, vol, temp);
    test = dg::blas2::dot( b, temp, b);
    if(rank==0)std::cout << "Test of multiplyVolume: "<<test<< " sol = "<<sol<<"\t";
    if(rank==0)std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";

    dg::MDVec inv3d = dg::create::inv_volume( grid);
    dg::blas1::pointwiseDot( vol3d, b, b);
    test = dg::blas2::dot( b, inv3d, b);
    if(rank==0)std::cout << "Test of inv_volume:     "<<test<< " sol = "<<sol<<"\t";
    if(rank==0)std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";
    temp = dg::create::inv_weights( grid);
    dg::tensor::pointwiseDivide(temp, vol, temp );
    test = dg::blas2::dot( b, temp, b);
    if(rank==0)std::cout << "Test of divideVolume:   "<<test<< " sol = "<<sol<<"\t";
    if(rank==0)std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";

    MPI_Finalize();

    return 0;
}
