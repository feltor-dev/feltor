#include <iostream>
#include <vector>

#include <mpi.h>
#include <thrust/device_vector.h>
#include "blas1.h"
#include "backend/mpi_evaluation.h"


//test program that calls every blas1 function for every specialization
double two( double x, double y){return 2.;}
double three( double x, double y){return 3.;}

//typedef dg::MPI_Vector<thrust::device_vector<double> > MHVec;
typedef dg::MPI_Vector<cusp::array1d<double, cusp::device_memory> > MHVec;

struct EXP{ __host__ __device__ double operator()(double x){return exp(x);}};

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int np[2];
    int periods[2] = {0,0};
    periods[0] = 1;
    periods[1] = 1;
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank == 0)
    {
        std::cout << "Type npx and npy\n";
        std::cin >> np[0] >> np[1];
        std::cout<< "You typed "<<np[0] <<" and "<<np[1]<<std::endl;
    }
    MPI_Bcast( np, 2, MPI_INT, 0, MPI_COMM_WORLD);

    int size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    if( rank == 0)
    {
        std::cout << "Size is "<<size<<std::endl;
        assert( size == np[0]*np[1]);
    }

    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);
    dg::MPI_Grid2d g( 0,1,0,1, 3,12,12, comm);
    if( rank == 0)
        g.display();
    MHVec v1 = dg::evaluate( two, g);
    MHVec v2 = dg::evaluate( three, g); 
    MHVec v3(v1);
    unsigned gsize = g.global().n()*g.global().n()*g.global().Nx()*g.global().Ny();

    double temp = dg::blas1::dot(v1,v2);
    if(rank==0)std::cout << "(2*3) = "<<temp/gsize << " (6)\n"; 
    dg::blas1::axpby( 2., v1, 3., v2, v3);
    if(rank==0)std::cout << "2*2+ 3*3 = " << v3[0] <<" (13)\n";
    dg::blas1::axpby( 0., v1, 3., v2, v3);
    if(rank==0)std::cout << "0*2+ 3*3 = " << v3[0] <<" (9)\n";
    dg::blas1::axpby( 2., v1, 0., v2, v3);
    if(rank==0)std::cout << "2*2+ 0*3 = " << v3[0] <<" (4)\n";
    dg::blas1::pointwiseDot( v1, v2, v3);
    if(rank==0)std::cout << "2*3 = "<<v3[0]<<" (6)\n";
    dg::blas1::pointwiseDot( 2., v1, v2, -4., v3);
    if(rank==0)std::cout << "2*2*3 -4*6 = "<<v3[0]<<" (-12)\n";
    dg::blas1::axpby( 2., v1, 3., v2);
    if(rank==0)std::cout << "2*2+ 3*3 = " << v2[0] <<" (13)\n";
    dg::blas1::axpby( 2.5, v1, 0., v2);
    if(rank==0)std::cout << "2.5*2+ 0 = " << v2[0] <<" (5)\n";
    dg::blas1::copy( v2, v1);
    if(rank==0)std::cout << "5 = " << v1[0] <<" (5)"<< std::endl;
    dg::blas1::scal( v1, 0.4);
    if(rank==0)std::cout << "5*0.4 = " << v1[0] <<" (2)"<< std::endl;
    dg::blas1::transform( v1, v3, EXP());
    if(rank==0)std::cout << "e^2 = " << v3[0] <<" (7.389056...)"<< std::endl;
    dg::blas1::scal( v2, 0.6);

    //v1 = 2, v2 = 3

    if(rank==0)std::cout << "Test std::vector \n";
    std::vector<MHVec > w1( 2, v1), w2(2, v2), w3( w2);
    temp = dg::blas1::dot( w1, w2);
    if(rank==0)std::cout << "2*(2*3) = "<<temp/gsize << " (12)\n"; 
    dg::blas1::axpby( 2., w1, 3., w2, w3);
    if(rank==0)std::cout << "2*2+ 3*3 = " << w3[0][0] <<" (13)\n";
    dg::blas1::axpby( 0., w1, 3., w2, w3);
    if(rank==0)std::cout << "0*2+ 3*3 = " << w3[0][0] <<" (9)\n";
    dg::blas1::axpby( 2., w1, 0., w2, w3);
    if(rank==0)std::cout << "2*2+ 0*3 = " << w3[0][0] <<" (4)\n";
    dg::blas1::pointwiseDot( w1, w2, w3);
    if(rank==0)std::cout << "2*3 = "<<w3[0][0]<<" (6)\n";
    dg::blas1::pointwiseDot( 2., w1, w2, -4., w3);
    if(rank==0)std::cout << "2*2*3 -4*6 = "<<w3[0][0]<<" (-12)\n";
    dg::blas1::axpby( 2., w1, 3., w2);
    if(rank==0)std::cout << "2*2+ 3*3 = " << w2[0][0] <<" (13)\n";
    dg::blas1::axpby( 2.5, w1, 0., w2);
    if(rank==0)std::cout << "2.5*2+ 0 = " << w2[0][0] <<" (5)\n";
    dg::blas1::copy( w2, w1);
    if(rank==0)std::cout << "5 = " << w1[0][0] <<" (5)"<< std::endl;
    dg::blas1::scal( w1, 0.4);
    if(rank==0)std::cout << "5*0.5 = " << w1[0][0] <<" (2)"<< std::endl;
    dg::blas1::transform( w1, w3, EXP());
    if(rank==0)std::cout << "e^2 = " << w3[0][0] <<" (7.389056...)"<< std::endl;
    dg::blas1::scal( w2, 0.6);
    if(rank==0)std::cout << "FINISHED\n\n";



    MPI_Finalize();
    return 0;

}
