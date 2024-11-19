#include <iostream>

#include <cusp/print.h>

#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "geometry.h"
#include "../blas2.h"


double R_0 = 4.*M_PI;


double sine( double R, double Z,double phi){ return sin(R-R_0)*sin(Z)*sin(phi)/sqrt(R);}

namespace dg { typedef dg::MPI_Vector<thrust::device_vector<double> > MDVec; }

//TEST geometry.h for every container and geometry that you want to use
int main( int argc, char* argv[] )
{
    MPI_Init(&argc, &argv);
    int dims[3] = {0,0,0};
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Dims_create( size, 3, dims);
    MPI_Comm comm;
    std::stringstream ss;
    ss<< dims[0]<<" "<<dims[1]<<" "<<dims[2];
    dg::mpi_init3d( dg::PER, dg::PER, dg::PER, comm, ss);
    dg::CylindricalMPIGrid3d grid( R_0 , R_0+ 2.*M_PI, 0.,2.*M_PI, 0., 2.*M_PI,
        3,32,24,16, dg::PER, dg::PER, dg::PER, comm);
    dg::MDVec vol = dg::tensor::volume(grid.metric());

    dg::MDVec b = dg::evaluate( sine, grid);
    dg::MDVec vol3d = dg::create::volume( grid);
    double test = dg::blas2::dot( b, vol3d, b);
    double sol = M_PI*M_PI*M_PI;
    if(rank==0)std::cout << "Test of volume:         "<<test<< " sol = "<<sol<<"\t";
    if(rank==0)std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";
    dg::MDVec temp = dg::create::weights( grid);
    dg::blas1::pointwiseDot( temp, vol, temp);
    test = dg::blas2::dot( b, temp, b);
    if(rank==0)std::cout << "Test of multiplyVolume: "<<test<< " sol = "<<sol<<"\t";
    if(rank==0)std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";

    dg::MDVec inv3d = dg::create::inv_volume( grid);
    dg::blas1::pointwiseDot( vol3d, b, b);
    test = dg::blas2::dot( b, inv3d, b);
    if(rank==0)std::cout << "Test of inv_volume:     "<<test<< " sol = "<<sol<<"\t";
    if(rank==0)std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";
    temp = dg::create::inv_weights( grid);
    dg::blas1::pointwiseDivide(temp, vol, temp );
    test = dg::blas2::dot( b, temp, b);
    if(rank==0)std::cout << "Test of divideVolume:   "<<test<< " sol = "<<sol<<"\t";
    if(rank==0)std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";

    MPI_Finalize();

    return 0;
}
