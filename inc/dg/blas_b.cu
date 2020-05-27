#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/timer.h"
#include "blas.h"
#include "topology/derivatives.h"
#include "topology/evaluation.h"
#include "topology/fast_interpolation.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
double left( double x, double y, double z) {return sin(x)*cos(y)*z;}
double right( double x, double y, double z) {return cos(x)*sin(y)*z;}
struct Expression{
   DG_DEVICE
   void operator() ( double& u, double v, double w, double param){
       u = param*u*v + w;
   }
};
struct test_routine{
    test_routine( double mu, double alpha):m_mu(mu), m_alpha(alpha){}
    DG_DEVICE
    void operator()( double g11, double g12, double g22, double in1, double in2, double& out1, double& out2){
        out1 = (g11)*(in1) + (g12)*(in2) + m_mu;
        out2 = (g12)*(in1) + (g22)*(in2) + m_alpha;
    }
private:
    double m_mu, m_alpha;
};

struct test_inplace{
    DG_DEVICE
    void operator()( double g11, double g12, double g22, double& inout1, double& inout2){
        double t = g11*inout1 + g12*inout2;
        inout2 = g12*inout1 + g22*inout2;
        inout1 = t;
    }
};

using value_type= double;
using Vector    = thrust::device_vector<double>;
using Matrix    = dg::DMatrix;
using ArrayVec  = std::array<Vector, 3>;

int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny, Nz;
    std::cout << "This program benchmarks basic vector and matrix-vector operations on the machine. These operations should be memory bandwidth bound. ";
    std::cout << "We therefore convert the measured time into a bandwidth using the given vector size and the STREAM convention for counting memory operations (each read and each write count as one memop. ";
    std::cout << "In an ideal case all operations perform with the same speed (that of the AXPBY operation, which is certainly memory bandwidth bound). With fast memory (GPU, XeonPhi...) the matrix-vector multiplications can be slower however\n";
    std::cout << "Input parameters are: \n";
    std::cout << "    n: # of polynomial coefficients = block size in matrices\n";
    std::cout << "   Nx: # of cells in x (must be multiple of 2)\n";
    std::cout << "   Ny: # of cells in y (must be multiple of 2)\n";
    std::cout << "   Nz: # of cells in z\n";
    std::cout << "Type n (3), Nx (512) , Ny (512) and Nz (10) \n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::Grid3d grid(      0., lx, 0, ly, 0, ly, n, Nx, Ny, Nz);
    dg::Grid3d grid_half = grid; grid_half.multiplyCellNumbers(0.5, 0.5);
    Vector w2d = dg::construct<Vector>( dg::create::weights(grid));

    //std::cout<<"Evaluate a function on the grid\n";
    t.tic();
    ArrayVec x = dg::construct<ArrayVec>( dg::evaluate( left, grid));
    t.toc();
    //std::cout<<"Evaluation of a function took    "<<t.diff()<<"s\n";
    //std::cout << "Sizeof value type is "<<sizeof(value_type)<<"\n";
    value_type gbytes=(value_type)x.size()*x[0].size()*sizeof(value_type)/1e9;
    std::cout << "Size of vectors is "<<gbytes<<" GB\n";
    dg::MultiMatrix<Matrix, ArrayVec> inter, project;
    dg::blas2::transfer(dg::create::fast_interpolation( grid_half, 2,2), inter);
    dg::blas2::transfer(dg::create::fast_projection( grid, 2,2), project);
    //dg::IDMatrix inter = dg::create::interpolation( grid, grid_half);
    //dg::IDMatrix project = dg::create::projection( grid_half, grid);
    int multi=100;
    //t.tic();
    std::cout<<"\nNo communication\n";
    ArrayVec y(x), z(x), u(x), v(x), w(x), h(x);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    std::cout<<"AXPBY (1*y-1*x=x)                "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpbypgz( 1., x, -1., 1, 2., z);
    t.toc();
    std::cout<<"AXPBYPGZ (1*x-1*1+2*z=z)         "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpbypgz( 1., x, -1., y, 3., x);
    t.toc();
    std::cout<<"AXPBYPGZ (1*x-1.*y+3*x=x) (A)    "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot(  y, x, x);
    t.toc();
    std::cout<<"pointwiseDot (yx=x) (A)          "<<t.diff()/multi<<"s\t" <<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 1., y, x, 2.,u,v,0.,  z);
    t.toc();
    std::cout<<"pointwiseDot (1*yx+2*uv=z)       "<<t.diff()/multi<<"s\t" <<6*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 1., y, x, 2.,u,v,0.,  v);
    t.toc();
    std::cout<<"pointwiseDot (1*yx+2*uv=v) (A)   "<<t.diff()/multi<<"s\t" <<5*gbytes*multi/t.diff()<<"GB/s\n";
    //Test new evaluate
    std::array<double, 3> array_p{ 1,2,3};
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::subroutine( Expression(), u, v, x, array_p);
    t.toc();
    std::cout<<"Subroutine (p*yx+w)              "<<t.diff()/multi<<"s\t" <<4*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::subroutine( test_routine(2.,4.), x, y, z, u, v, w, h);
    t.toc();
    std::cout<<"Subroutine ( G Cdot x = y)       "<<t.diff()/multi<<"s\t"<<9*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::subroutine( test_inplace(), x, y, z, u, v);
    t.toc();
    std::cout<<"Subroutine ( G Cdot x = x)       "<<t.diff()/multi<<"s\t"<<7*gbytes*multi/t.diff()<<"GB/s\n";
    /////////////////////SYMV////////////////////////////////
    std::cout<<"\nLocal communication\n";
    Matrix M;
    dg::blas2::transfer(dg::create::dx( grid, dg::backward), M);
    dg::blas2::symv( M, x, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"forward x derivative took        "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::backward), M);
    dg::blas2::symv( M, x, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"forward y derivative took        "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dx( grid, dg::centered), M);
    dg::blas2::symv( M, x, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"centered x derivative took       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::centered), M);
    dg::blas2::symv( M, x, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"centered y derivative took       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    if( grid.Nz() > 1)
    {
        dg::blas2::transfer(dg::create::dz( grid, dg::centered), M);
        dg::blas2::symv( M, x, y);//warm up
        t.tic();
        for( int i=0; i<multi; i++)
            dg::blas2::symv( M, x, y);
        t.toc();
        std::cout<<"centered z derivative took       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    }

    dg::blas2::transfer(dg::create::jumpX( grid), M);
    dg::blas2::symv( M, x, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"jump X took                      "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    ArrayVec x_half = dg::construct<ArrayVec>(dg::evaluate( dg::zero, grid_half));
    dg::blas2::gemv( inter, x_half, x); //warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( inter, x_half, x); //internally 2 multiplications: quarter-> half, half -> full
    t.toc();
    std::cout<<"Interpolation quarter to full    "<<t.diff()/multi<<"s\t"<<3.75*gbytes*multi/t.diff()<<"GB/s\n";
    dg::blas2::gemv( project, x, x_half); //warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( project, x, x_half); //internally 2 multiplications: full -> half, half -> quarter
    t.toc();
    std::cout<<"Projection full to quarter       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    //////////////////////these functions are more mean to dot
    std::cout<<"\nGlobal communication\n";
    x = dg::construct<ArrayVec>( dg::evaluate( left, grid));
    y = dg::construct<ArrayVec>( dg::evaluate( right, grid));
    value_type norm=0;
    norm += dg::blas1::dot( x,y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas1::dot( x,y);
    t.toc();
    std::cout<<"DOT1(x,y) took                   " <<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";
    norm += dg::blas2::dot( w2d, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas2::dot( w2d, y);
    t.toc();
    std::cout<<"DOT2(y,w,y) (A) took             " <<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";
    norm += dg::blas2::dot( x, w2d, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas2::dot( x, w2d, y);
    t.toc();
    std::cout<<"DOT2(x,w,y) took                 " <<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n"; //DOT should be faster than axpby since it is only loading vectors and not writing them

    std::cout << "\nSequential recursive calls";
    unsigned size_rec = 1e4;
    std::vector<double> test_recursive(size_rec, 0.1);
    gbytes=(double)size_rec*sizeof(double)/1e9;
    std::cout << " with size "<<gbytes<<"GB\n";
    norm += dg::blas1::dot( 1., test_recursive);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas1::dot( 1., test_recursive);//warm up
    t.toc();
    std::cout<<"recursive dot took               " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    thrust::host_vector<double> test_serial((int)size_rec, (double)0.1);
    norm += dg::blas1::dot( 1., test_serial);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas1::dot( test_serial, test_serial);//warm up
    t.toc();
    std::cout<<"Serial dot took                  " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    //maybe test how fast a recursive axpby is compared to serial axpby
    dg::blas1::axpby( 1., test_recursive, 2., test_recursive);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpby( 1., test_recursive, 2., test_recursive);//warm up
    t.toc();
    std::cout<<"recursive axpby took             " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    //
    dg::blas1::axpby( 1., test_serial, 2., test_serial);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpby( 1., test_serial, 2., test_serial);//warm up
    t.toc();
    std::cout<<"serial axpby took                " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    return 0;
}
