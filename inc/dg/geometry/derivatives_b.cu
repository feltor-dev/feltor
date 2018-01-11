#include <iostream>
#include <thrust/device_vector.h>
#include "timer.cuh"
#include "blas.h"
#include "derivatives.h"
#include "evaluation.cuh"
#include "typedefs.cuh"
#include "sparseblockmat.cuh"

const double lx = 2*M_PI;
double sinx(   double x, double y, double z) { return sin(x);}
double cosx(   double x, double y, double z) { return cos(x);}
double siny(   double x, double y, double z) { return sin(y);}
double cosy(   double x, double y, double z) { return cos(y);}
double sinz(   double x, double y, double z) { return sin(z);}
double cosz(   double x, double y, double z) { return cos(z);}
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::DIR;
dg::bc bcz = dg::DIR;

typedef dg::EllSparseBlockMatDevice<double> Matrix;
typedef dg::DVec Vector;
//typedef dg::EllSparseBlockMatDevice<float> Matrix;
//typedef thrust::device_vector<float> Vector;

int main()
{
    unsigned n, Nx, Ny, Nz;
    std::cout << "Type in n, Nx and Ny and Nz!\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::Grid3d g( 0, lx, 0, lx, 0., lx, n, Nx, Ny, Nz, bcx, bcy, bcz);
    const Vector w3d = dg::create::weights( g);
    dg::Timer t;
    std::cout << "TEST DX \n";
    {
    Matrix dx = dg::create::dx( g, bcx, dg::forward);
    Vector v = dg::evaluate( sinx, g);
    Vector w = v;
    const Vector u = dg::evaluate( cosx, g);

    if( thrust::detail::is_same< dg::VectorTraits<Vector>::value_type, float>::value )
        std::cout << "Value type is float! "<<std::endl;
    else
        std::cout << "Value type is double! "<<std::endl;

    t.tic();
    dg::blas2::symv( dx, v, w);
    t.toc();
    std::cout << "Dx took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., u, -1., w);
    std::cout << "DX: Distance to true solution: "<<sqrt(dg::blas2::dot(w, w3d, w))<<"\n";
    }
    std::cout << "TEST DY \n";
    {
    const Vector func = dg::evaluate( siny, g);
    const Vector deri = dg::evaluate( cosy, g);

    Matrix dy = dg::create::dy( g, dg::forward); 
    Vector temp( func);
    t.tic();
    dg::blas2::gemv( dy, func, temp);
    t.toc();
    std::cout << "Dy took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., deri, -1., temp);
    std::cout << "DY(1):           Distance to true solution: "<<sqrt(dg::blas2::dot(temp, w3d, temp))<<"\n";
    }
    std::cout << "TEST DZ \n";
    {
    const Vector func = dg::evaluate( sinz, g);
    const Vector deri = dg::evaluate( cosz, g);

    Matrix dz = dg::create::dz( g); 
    Vector temp( func);
    t.tic();
    dg::blas2::gemv( dz, func, temp);
    t.toc();
    std::cout << "Dz took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., deri, -1., temp);
    std::cout << "DZ(1):           Distance to true solution: "<<sqrt(dg::blas2::dot(temp, w3d, temp))<<"\n";
    }
    std::cout << "JumpX and JumpY \n";
    {
    const Vector func = dg::evaluate( sinx, g);

    Matrix jumpX = dg::create::jumpX( g); 
    Matrix jumpY = dg::create::jumpY( g); 
    Matrix jumpZ = dg::create::jumpZ( g); 
    Vector temp( func);
    t.tic();
    dg::blas2::gemv( jumpX, func, temp);
    t.toc();
    std::cout << "JumpX took "<<t.diff()<<"s\n";
    t.tic();
    dg::blas2::gemv( jumpY, func, temp);
    t.toc();
    std::cout << "JumpY took "<<t.diff()<<"s\n";
    t.tic();
    dg::blas2::gemv( jumpZ, func, temp);
    t.toc();
    std::cout << "JumpZ took "<<t.diff()<<"s\n";
    }
    return 0;
}
