#include <iostream>
#include <vector>

#include <mpi.h>
#include <thrust/device_vector.h>
#include "backend/mpi_init.h"
#include "blas1.h"
#include "functors.h"


//test program that calls every blas1 function for every specialization

typedef dg::MPI_Vector<thrust::device_vector<double> > MVec;
//typedef dg::MPI_Vector<cusp::array1d<double, cusp::device_memory> > MVec;

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm;
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    if(rank==0)std::cout << "This program tests the blas1 functions up to binary reproducibility with the exception of the dot function, which is tested in the dg/topology/evaluation_mpit program\n";
    //mpi_init2d( dg::PER, dg::PER, comm);
    comm = MPI_COMM_WORLD;
    int dims[2] = {1, size};
    int periods[2] = {false, false};
    MPI_Comm comm_cart, cartX, cartY;
    dg::mpi_cart_create( comm, 2, dims, periods, false, &comm_cart);
    int remainsX[2] = {1,0}, remainsY[2] = {0,1};
    dg::mpi_cart_sub( comm_cart, remainsX, &cartX);
    dg::mpi_cart_sub( comm_cart, remainsY, &cartY);
    {
    thrust::device_vector<double> v1p( 500, 2.0002), v2p( 500, 3.00003), v3p(500,5.0005), v4p(500,4.00004);
    MVec v1(v1p, comm), v2(v2p, comm), v3(v3p, comm), v4(v4p, comm), v5(v4p, comm);
    if(rank==0)std::cout << "A TEST IS PASSED IF THE RESULT IS ZERO.\n";
    dg::exblas::udouble ud;
    dg::blas1::scal( v3, 3e-10); ud.d = v3.data()[0];
    if(rank==0)std::cout << "scal (x=ax)           "<<ud.i-4474825110624711575<<std::endl;
    dg::blas1::plus( v3, 3e-10); ud.d = v3.data()[0];
    if(rank==0)std::cout << "plus (x=x+a)          "<<ud.i-4476275821608249130<<std::endl;
    dg::blas1::axpby( 3e+10, v3, 1 , v4); ud.d = v4.data()[0];
    if(rank==0)std::cout << "fma (y=ax+y)          "<<ud.i-4633360230582305548<<std::endl;
    dg::blas1::axpby( 3e-10, v1, -2e-10 , v2); ud.d = v2.data()[0];
    if(rank==0)std::cout << "axpby (y=ax+by)       "<<ud.i-4408573477492505937<<std::endl;
    dg::blas1::axpbypgz( 2.5, v1, 7.e+10, v2, -0.125, v3); ud.d = v3.data()[0];
    if(rank==0)std::cout << "axpbypgz (y=ax+by+gz) "<<ud.i-4617320336812948958<<std::endl;
    dg::blas1::pointwiseDot( v1, v2, v3); ud.d = v3.data()[0];
    if(rank==0)std::cout << "pDot (z=xy)           "<<ud.i-4413077932784031586<<std::endl;
    dg::blas1::pointwiseDot( 0.2, v1, v2, +0.4e10, v3); ud.d = v3.data()[0];
    if(rank==0)std::cout << "pDot ( z=axy+bz)      "<<ud.i-4556605413983777388<<std::endl;
    dg::blas1::pointwiseDot( -0.2, v1, v2, 0.4, v3, v4, 0.1, v5); ud.d = v5.data()[0];
    if(rank==0)std::cout << "pDot (z=axy+bfh+gz)   "<<ud.i-4601058031075598447<<std::endl;
    dg::blas1::pointwiseDot( 0.2, v1, v2,v4, 0.4, v3); ud.d = v3.data()[0];
    if(rank==0)std::cout << "pDot (z=awxy + bz)    "<<ud.i-4550507856334720009<<std::endl;
    dg::blas1::pointwiseDivide( 5.,v1,v2,-1,v3); ud.d = v3.data()[0];
    if(rank==0)std::cout << "pDivide (z=ax/y+bz)   "<<ud.i-4820274520177585116<<std::endl;
    dg::blas1::transform( v1, v3, dg::EXP<>()); ud.d = v3.data()[0];
    if(rank==0)std::cout << "transform y=exp(x)    "<<ud.i-4620007020034741378<<std::endl;

    std::vector<double> xs{1,2,3};
    std::vector<double> ys{10*(double)rank + 10};
    double zs{100};
    double ws{1000};
    std::vector<thrust::complex<double>> y(xs.size()*ys.size());
    MVec xsd(xs, cartX), ysd(ys, cartY);
    dg::MPI_Vector<thrust::device_vector<thrust::complex<double>>> yd(y, comm);
    dg::blas1::kronecker( yd, dg::equals(), []DG_DEVICE(double x, double y,
                double z, double u){ return thrust::complex<double>{x+y+z+u,1};}, xsd, ysd, zs, ws);
    thrust::copy( yd.data().begin(), yd.data().end(), y.begin());
    for( int i=0; i<size; i++)
        if(rank==i)std::cout << "Kronecker test (X ox Y) " << y[1]-thrust::complex<double>{(1112+10*i),1} <<"\n";

    auto ydd = dg::kronecker( []DG_DEVICE( double x, double y, double z, double
    u){ return thrust::complex<double>{x+y+z+u,1};}, xsd, ysd, zs, ws);
    thrust::copy( ydd.data().begin(), ydd.data().end(), y.begin());
    if(rank==0)std::cout << "Kronecker test (X ox Y) " << y[1]-thrust::complex<double>{1112,1} <<"\n";


    }
    thrust::device_vector<double> v1p( 5, 2.), v2p( 5, 3.), v3p(5,5.), v4p(5,4.);
    MVec v1(v1p, comm), v2(v2p, comm), v3(v3p, comm), v4(v4p, comm);

    if(rank==0)std::cout << "Human readable test RecursiveVector (passed if ouput equals value in brackets) \n";
    std::vector<MVec > w1( 2, v1), w2(2, v2), w3( w2), w4( 2, v4);
    dg::blas1::axpby( 2., w1, 3., w2, w3);
    if(rank==0)std::cout << "2*2+ 3*3 = " << w3[0].data()[0] <<" (13)\n";
    dg::blas1::axpby( 0., w1, 3., w2, w3);
    if(rank==0)std::cout << "0*2+ 3*3 = " << w3[0].data()[0] <<" (9)\n";
    dg::blas1::axpby( 2., w1, 0., w2, w3);
    if(rank==0)std::cout << "2*2+ 0*3 = " << w3[0].data()[0] <<" (4)\n";
    dg::blas1::pointwiseDot( w1, w2, w3);
    if(rank==0)std::cout << "2*3 = "<<w3[0].data()[0]<<" (6)\n";
    dg::blas1::pointwiseDot( 2., w1, w2, -4., w3);
    if(rank==0)std::cout << "2*2*3 -4*6 = "<<w3[0].data()[0]<<" (-12)\n";
    dg::blas1::pointwiseDot( 2., w1, w2,w4, -4., w3);
    if(rank==0)std::cout << "2*2*3 -4*(-12) = "<<w3[0].data()[0]<<" (96)\n";
    dg::blas1::axpby( 2., w1, 3., w2);
    if(rank==0)std::cout << "2*2+ 3*3 = " << w2[0].data()[0] <<" (13)\n";
    dg::blas1::axpby( 2.5, w1, 0., w2);
    if(rank==0)std::cout << "2.5*2+ 0 = " << w2[0].data()[0] <<" (5)\n";
    dg::blas1::copy( w2, w1);
    if(rank==0)std::cout << "5 = " << w1[0].data()[0] <<" (5)"<< std::endl;
    dg::blas1::scal( w1, 0.4);
    if(rank==0)std::cout << "5*0.5 = " << w1[0].data()[0] <<" (2)"<< std::endl;
    dg::blas1::evaluate( w4, dg::equals(), dg::AbsMax<>(), w1, w2);
    if(rank==0)std::cout << "absMax( 2, 5) = " << w4[0].data()[0] <<" (5)"<< std::endl;
    dg::blas1::transform( w1, w3, dg::EXP<>());
    if(rank==0)std::cout << "e^2 = " << w3[0].data()[0] <<" (7.389056...)"<< std::endl;
    dg::blas1::scal( w2, 0.6);
    dg::blas1::plus( w3, -7.0);
    if(rank==0)std::cout << "e^2-7 = " << w3[0].data()[0] <<" (0.389056...)"<< std::endl;
    if(rank==0)std::cout << "\nFINISHED! Continue with topology/evaluation_mpit.cu !\n\n";



    MPI_Finalize();
    return 0;

}
