#include <iostream>
#include <vector>

#include <mpi.h>
#include <thrust/device_vector.h>
#include "backend/mpi_init.h"
#include "geometry/mpi_evaluation.h"
#include "blas1.h"
#include "functors.h"


//test program that calls every blas1 function for every specialization
double two( double x, double y){return 2;}
double three( double x, double y){return 3;}
double four( double x, double y){return 4;}
double five( double x, double y){return 5;}
double two2( double x, double y){return 2.0002;}
double three3( double x, double y){return 3.00003;}
double four4( double x, double y){return 4.00004;}
double five5( double x, double y){return 5.0005;}

typedef dg::MPI_Vector<thrust::device_vector<double> > MVec;
//typedef dg::MPI_Vector<cusp::array1d<double, cusp::device_memory> > MVec;

struct EXP{ __host__ __device__ double operator()(double x){return exp(x);}};

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm;
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)std::cout << "This program tests the blas1 functions up to binary reproducibility with the exception of the dot function, which is tested in the dg/geometry/evaluation_mpit program\n";
    mpi_init2d( dg::PER, dg::PER, comm);
    dg::MPIGrid2d g( 0,1,0,1, 3,120,120, comm);
    {
    MVec v1 = dg::evaluate( two2, g);
    MVec v2 = dg::evaluate( three3, g);
    MVec v3 = dg::evaluate( five5, g);
    MVec v4 = dg::evaluate( four4, g), v5(v4);
    if(rank==0)std::cout << "A TEST IS PASSED IF THE RESULT IS ZERO.\n";
    exblas::udouble ud;
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
    dg::blas1::transform( v1, v3, EXP()); ud.d = v3.data()[0];
    if(rank==0)std::cout << "transform y=exp(x)    "<<ud.i-4620007020034741378<<std::endl;
    }
    MVec v1 = dg::evaluate( two, g);
    MVec v2 = dg::evaluate( three, g);
    MVec v3 = dg::evaluate( five, g);
    MVec v4 = dg::evaluate( four, g);

    if(rank==0)std::cout << "Human readable test VectorVector (passed if ouput equals value in brackets) \n";
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
    dg::blas1::transform( w1, w3, EXP());
    if(rank==0)std::cout << "e^2 = " << w3[0].data()[0] <<" (7.389056...)"<< std::endl;
    dg::blas1::scal( w2, 0.6);
    dg::blas1::plus( w3, -7.0);
    if(rank==0)std::cout << "e^2-7 = " << w3[0].data()[0] <<" (0.389056...)"<< std::endl;
    if(rank==0)std::cout << "FINISHED\n\n";



    MPI_Finalize();
    return 0;

}
