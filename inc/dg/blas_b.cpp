#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifdef WITH_MPI
#include <mpi.h>
#include "backend/mpi_init.h"
#include "topology/mpi_evaluation.h"
#include "topology/mpi_weights.h"
#endif

#include "backend/timer.h"
#include "blas.h"
#include "topology/filter.h"
#include "topology/fem.h"
#include "topology/tensor.h"
#include "topology/multiply.h"
#include "topology/average.h"
#include "topology/derivativesA.h"
#include "topology/evaluation.h"
#include "topology/fast_interpolation.h"
#include "topology/stencil.h"

using value_type = double;
using Vector     = dg::x::DVec;
using Matrix     = dg::x::DMatrix;
//using value_type = float;
//using Vector     = dg::x::fDVec;
//using Matrix     = dg::x::fDMatrix;

using ArrayVec   = std::array<Vector, 3>;

const value_type lx = 2.*M_PI;
const value_type ly = 2.*M_PI;
value_type left( value_type x, value_type y, value_type z) {return sin(x)*cos(y)*z;}
value_type right( value_type x, value_type y, value_type z) {return cos(x)*sin(y)*z;}

struct Expression{
   DG_DEVICE
   void operator() ( value_type& u, value_type v, value_type w, value_type param){
       u = param*u*v + w;
   }
};
struct test_routine{
    test_routine( value_type mu, value_type alpha):m_mu(mu), m_alpha(alpha){}
    DG_DEVICE
    void operator()( value_type g11, value_type g12, value_type g22, value_type in1, value_type in2, value_type& out1, value_type& out2){
        out1 = (g11)*(in1) + (g12)*(in2) + m_mu;
        out2 = (g12)*(in1) + (g22)*(in2) + m_alpha;
    }
private:
    value_type m_mu, m_alpha;
};
struct test_inplace{
    DG_DEVICE
    void operator()( value_type g11, value_type g12, value_type g22, value_type& inout1, value_type& inout2){
        value_type t = g11*inout1 + g12*inout2;
        inout2 = g12*inout1 + g22*inout2;
        inout1 = t;
    }
};

int main( int argc, char* argv[])
{
#ifdef WITH_MPI
    dg::mpi_init( argc, argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    dg::Timer t;
    unsigned n, Nx, Ny, Nz;
    DG_RANK0 std::cout << "This program benchmarks basic vector and matrix-vector operations on the machine. These operations should be memory bandwidth bound. ";
    DG_RANK0 std::cout << "We therefore convert the measured time into a bandwidth using the given vector size and the STREAM convention for counting memory operations (each read and each write count as one memop. ";
    DG_RANK0 std::cout << "In an ideal case all operations perform with the same speed (that of the AXPBY operation, which is certainly memory bandwidth bound). With fast memory (GPU, XeonPhi...) the matrix-vector multiplications can be slower however\n";
    DG_RANK0 std::cout << "Input parameters are: \n";
#ifdef WITH_MPI
    if(rank==0)std::cout << "    npx: # of processes in x (must divide Nx and total # of processes!\n";
    if(rank==0)std::cout << "    npy: # of processes in y (must divide Ny and total # of processes!\n";
    if(rank==0)std::cout << "    npz: # of processes in z (must divide Nz and total # of processes!\n";
    MPI_Comm comm = dg::mpi_cart_create( {dg::PER, dg::PER, dg::PER}, std::cin,
        MPI_COMM_WORLD, true, true, std::cout);
#endif
    DG_RANK0 std::cout << "    n: # of polynomial coefficients = block size in matrices\n";
    DG_RANK0 std::cout << "   Nx: # of cells in x (must be multiple of 2)\n";
    DG_RANK0 std::cout << "   Ny: # of cells in y (must be multiple of 2)\n";
    DG_RANK0 std::cout << "   Nz: # of cells in z\n";
#ifdef WITH_MPI
    dg::mpi_read_grid( n, {&Nx, &Ny, &Nz}, comm, std::cin, true, std::cout);
    dg::RealMPIGrid3d<value_type> grid(   0., lx, 0, ly, 0, ly, n, Nx, Ny, Nz, comm);
#else
    DG_RANK0 std::cout << "Type n (3), Nx (512) , Ny (512) and Nz (10) \n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::RealGrid3d<value_type> grid(      0., lx, 0, ly, 0, ly, n, Nx, Ny, Nz);
#endif
    dg::x::RealGrid3d<value_type> grid_half = grid;
    grid_half.multiplyCellNumbers(0.5, 0.5);

    Vector w2d = dg::construct<Vector>( dg::create::weights(grid));
    Vector XX( w2d);

    //std::cout<<"Evaluate a function on the grid\n";
    t.tic();
    ArrayVec x = dg::construct<ArrayVec>( dg::evaluate( left, grid));
    t.toc();
    //std::cout<<"Evaluation of a function took    "<<t.diff()<<"s\n";
    //std::cout << "Sizeof value type is "<<sizeof(value_type)<<"\n";
    value_type gbytes=(value_type)x.size()*grid.size()*sizeof(value_type)/1e9;
    DG_RANK0 std::cout << "Size of vectors is "<<gbytes<<" GB\n";
    dg::MultiMatrix<Matrix, ArrayVec> inter, project;
    dg::blas2::transfer(dg::create::fast_interpolation( grid_half, 1,2,2), inter);
    dg::blas2::transfer(dg::create::fast_projection( grid, 1,2,2), project);
    dg::x::IDMatrix stencil = dg::create::window_stencil( {3,3}, grid,
            grid.bcx(), grid.bcy());
    int multi=100;
    //t.tic();
    DG_RANK0 std::cout<<"\nNo communication\n";
    ArrayVec y(x), z(x), u(x), v(x), w(x), h(x);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    DG_RANK0 std::cout<<"AXPBY (1*y-1*x=x)                "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpbypgz( 1., x, -1., 1, 2., z);
    t.toc();
    DG_RANK0 std::cout<<"AXPBYPGZ (1*x-1*1+2*z=z)         "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpbypgz( 1., x, -1., y, 3., x);
    t.toc();
    DG_RANK0 std::cout<<"AXPBYPGZ (1*x-1.*y+3*x=x) (A)    "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot(  y, x, x);
    t.toc();
    DG_RANK0 std::cout<<"pointwiseDot (yx=x) (A)          "<<t.diff()/multi<<"s\t" <<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 1., y, x, 2.,u,v,0.,  z);
    t.toc();
    DG_RANK0 std::cout<<"pointwiseDot (1*yx+2*uv=z)       "<<t.diff()/multi<<"s\t" <<6*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 1., y, x, 2.,u,v,0.,  v);
    t.toc();
    DG_RANK0 std::cout<<"pointwiseDot (1*yx+2*uv=v) (A)   "<<t.diff()/multi<<"s\t" <<5*gbytes*multi/t.diff()<<"GB/s\n";
    Vector absx = grid.abscissas(0);
    Vector absy = grid.abscissas(1);
    Vector absz = grid.abscissas(2);
    t.tic();
    for( int i=0; i<multi*(int)x.size(); i++)
        dg::blas1::kronecker( XX, dg::equals(), []DG_DEVICE(double x, double y,
                    double z){ return x*y*z;}, absx, absy, absz);
    t.toc();
    DG_RANK0 std::cout<<"Kronecker (x X y X z)            "<<t.diff()/multi<<"s\t" <<2*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::subroutine( test_routine(2.,4.), x, y, z, u, v, w, h);
    t.toc();
    DG_RANK0 std::cout<<"Subroutine ( G Cdot x = y)       "<<t.diff()/multi<<"s\t"<<9*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::subroutine( test_inplace(), x, y, z, u, v);
    t.toc();
    DG_RANK0 std::cout<<"Subroutine ( G Cdot x = x)       "<<t.diff()/multi<<"s\t"<<7*gbytes*multi/t.diff()<<"GB/s\n";
    std::vector<ArrayVec> matrix( 10, x);
    std::vector<const ArrayVec*> matrix_ptrs = dg::asPointers(matrix);
    std::vector<value_type> coeffs( 10, 0.5);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( 1., dg::asDenseMatrix(matrix_ptrs), coeffs, 0.,  x);
    t.toc();
    DG_RANK0 std::cout<<"Dense Matrix Symv (Mc = x)       "<<t.diff()/multi
             <<"s\t"<<(coeffs.size()+2)*gbytes*multi/t.diff()<<"GB/s\n";

    dg::SparseTensor<Vector> g(grid);
    dg::tensor::multiply2d( 1., g, x[0], y[0], 0., x[0], y[0]);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::tensor::multiply2d( 1., g, x[0], y[0], 0., x[0], y[0]);
    t.toc();
    DG_RANK0 std::cout<<"Multiply2d Unit tensor inplace took       "<<t.diff()/multi<<"s\t"<<6/3*gbytes*multi/t.diff()<<"GB/s\n";

    g.idx(0,0) = 0, g.idx(0,1) = g.idx(1,0) = 1, g.idx(1,1) = 2;
    g.values().resize(3);
    g.values()[0] = g.values()[1] = g.values()[2] = w2d;
    t.tic();
    for( int i=0; i<multi; i++)
        dg::tensor::multiply2d( 1., g, x[0], y[0], 0., x[0], y[0]);
    t.toc();
    DG_RANK0 std::cout<<"multiply_inplace(g,x,y) took              "<<t.diff()/multi<<"s\t"<<7/3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::tensor::multiply2d( 1., g, x[0], y[0], 0., u[0], v[0]);
    t.toc();
    DG_RANK0 std::cout<<"multiply2d(g,x,y,u,v) took                "<<t.diff()/multi<<"s\t"<<9/3*gbytes*multi/t.diff()<<"GB/s\n";
    /////////////////////SYMV////////////////////////////////
    /// Note: here we re-initialize x and y, note that if we didn't do that we
    /// get different timings on cpu and omp for symv. Why?
    x = dg::construct<ArrayVec>( dg::evaluate( left, grid)), y=x;
    DG_RANK0 std::cout<<"\nLocal communication\n";
    Matrix M;
    dg::blas2::transfer(dg::create::dx( grid, dg::backward), M);
    dg::blas2::symv( M, x, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    DG_RANK0 std::cout<<"forward x derivative took        "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::backward), M);
    dg::blas2::symv( M, x, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    DG_RANK0 std::cout<<"forward y derivative took        "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dx( grid, dg::centered), M);
    dg::blas2::symv( M, x, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    DG_RANK0 std::cout<<"centered x derivative took       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::centered), M);
    dg::blas2::symv( M, x, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    DG_RANK0 std::cout<<"centered y derivative took       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";

    if( grid.Nz() > 1)
    {
        dg::blas2::transfer(dg::create::dz( grid, dg::centered), M);
        dg::blas2::symv( M, x, y);//warm up
        t.tic();
        for( int i=0; i<multi; i++)
            dg::blas2::symv( M, x, y);
        t.toc();
        DG_RANK0 std::cout<<"centered z derivative took       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    }

#ifndef WITH_MPI
    // Only for shared memory
    using MassMatrix = dg::KroneckerTriDiagonal2d<Vector>;
    using InvMassMatrix = dg::InverseKroneckerTriDiagonal2d<Vector>;
    MassMatrix fem_mass = dg::create::fem_mass2d( grid);
    dg::blas2::symv( fem_mass, x[0], y[0]);
    t.tic();
    for( unsigned i=0; i<multi*x.size(); i++)
        dg::blas2::symv( fem_mass, x[0], y[0]);
    t.toc();
    DG_RANK0 std::cout<<"SYMV (y=Sx) FEM                  "
             <<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    InvMassMatrix inv_fem_mass = dg::create::inv_fem_mass2d( grid);
    dg::blas2::symv( inv_fem_mass, y[0], x[0]);
    t.tic();
    for( unsigned i=0; i<multi*x.size(); i++)
        dg::blas2::symv( inv_fem_mass, y[0], x[0]);
    t.toc();
    DG_RANK0 std::cout<<"Thomas SYMV (y=S^{-1}x) FEM      "
             <<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    unsigned ysize = y[0].size();
    for( int i=0; i<multi; i++)
        dg::blas2::parallel_for( [ysize]DG_DEVICE( unsigned i, double* x, const double* y){
                x[i] = y[(i+1)%ysize] - y[i];
            }, ysize, x[0], y[0]);
    t.toc();
    DG_RANK0 std::cout<<"Stencil forward derivative took  "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()/x.size()<<"GB/s\n";

    dg::x::IDMatrix limiter_stencil = dg::create::limiter_stencil( dg::coo3d::x, grid,
            grid.bcx());
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::stencil( dg::CSRSlopeLimiter<double>(), limiter_stencil, x[0], y[0]);
    t.toc();
    DG_RANK0 std::cout<<"stencil Slope Limiter      took  "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()/x.size()<<"GB/s\n";
#endif
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::stencil( dg::CSRMedianFilter(), stencil, x[0], y[0]);
    t.toc();
    DG_RANK0 std::cout<<"stencil Median             took  "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()/x.size()<<"GB/s\n";

    dg::blas2::transfer(dg::create::jumpX( grid), M);
    dg::blas2::symv( M, x, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    DG_RANK0 std::cout<<"jump X took                      "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    ArrayVec x_half = dg::construct<ArrayVec>(dg::evaluate( dg::zero, grid_half));
    dg::blas2::gemv( inter, x_half, x); //warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( inter, x_half, x); //internally 2 multiplications: quarter-> half, half -> full
    t.toc();
    DG_RANK0 std::cout<<"Interpolation quarter to full    "<<t.diff()/multi<<"s\t"<<3.75*gbytes*multi/t.diff()<<"GB/s\n";
    dg::blas2::gemv( project, x, x_half); //warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( project, x, x_half); //internally 2 multiplications: full -> half, half -> quarter
    t.toc();
    DG_RANK0 std::cout<<"Projection full to quarter       "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    //////////////////////these functions are more mean to dot
    DG_RANK0 std::cout<<"\nGlobal communication\n";
    x = dg::construct<ArrayVec>( dg::evaluate( left, grid));
    y = dg::construct<ArrayVec>( dg::evaluate( right, grid));
    value_type norm=0;
    norm += dg::blas1::dot( x,y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas1::dot( x,y);
    t.toc();
    DG_RANK0 std::cout<<"DOT1(x,y) took                   " <<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";
    norm += dg::blas2::dot( w2d, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas2::dot( w2d, y);
    t.toc();
    DG_RANK0 std::cout<<"DOT2(y,w,y) (A) took             " <<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";
    norm += dg::blas2::dot( x, w2d, y);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas2::dot( x, w2d, y);
    t.toc();
    DG_RANK0 std::cout<<"DOT2(x,w,y) took                 " <<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n"; //DOT should be faster than axpby since it is only loading vectors and not writing them

    dg::x::cDVec cc3d = dg::construct<dg::x::cDVec>( dg::evaluate( dg::zero, grid));
    dg::blas1::transform( x[0],cc3d, []DG_DEVICE( double x){ return
        thrust::complex{x, x};});
    dg::x::cDVec cc(cc3d);
    thrust::complex<double> cintegral = dg::blas1::dot( cc3d, cc);
    cintegral += dg::blas1::dot( cc3d,cc3d);//warm up
    t.tic();
    for( int i=0; i<multi*(int)x.size(); i++)
        cintegral += dg::blas1::dot( cc3d,cc);
    t.toc();
    DG_RANK0 std::cout<<"vDOT1(x,y) (complex) took        " <<t.diff()/multi<<"s\t"<<4*gbytes*multi/t.diff()<<"GB/s\n";
    cintegral += dg::blas2::dot( cc3d,w2d,cc3d);//warm up
    t.tic();
    for( int i=0; i<multi*(int)x.size(); i++)
        cintegral += dg::blas2::dot( cc3d,w2d,cc);
    t.toc();
    DG_RANK0 std::cout<<"vDOT2(x,w,y) (complex) took      " <<t.diff()/multi<<"s\t"<<5*gbytes*multi/t.diff()<<"GB/s\n";
    ////////////////////////////////////////////////////////////////
    DG_RANK0 std::cout<<"\nAverages\n";
    dg::Average<dg::x::IDMatrix, dg::x::DVec> pol(grid, dg::coo3d::x);
    t.tic();
    for( int i=0; i<multi; i++)
        pol( x[0], y[0]);
    t.toc();
    DG_RANK0 std::cout << "Average vector over x took:      "<<t.diff()/multi<<"s\t"<<3/3*gbytes*multi/t.diff()<<"GB/s\n";
    pol = dg::Average<dg::x::IDMatrix, dg::x::DVec>(grid, dg::coo3d::y);
    t.tic();
    for( int i=0; i<multi; i++)
        pol( x[0], y[0]);
    t.toc();
    DG_RANK0 std::cout << "Average vector over y took:      "<<t.diff()/multi<<"s\t"<<3/3*gbytes*multi/t.diff()<<"GB/s\n";
    if( grid.Nz() > 1)
    {
        pol = dg::Average<dg::x::IDMatrix, dg::x::DVec>(grid, dg::coo3d::z);
        t.tic();
        for( int i=0; i<multi; i++)
            pol( x[0], y[0]);
        t.toc();
        DG_RANK0 std::cout << "Average vector over z took:      "<<t.diff()/multi<<"s\t"<<3/3*gbytes*multi/t.diff()<<"GB/s\n";
    }


#ifndef WITH_MPI
    DG_RANK0 std::cout << "\nSequential recursive calls";
    unsigned size_rec = 1e4;
    std::vector<value_type> test_recursive(size_rec, 0.1);
    gbytes=(value_type)size_rec*sizeof(value_type)/1e9;
    DG_RANK0 std::cout << " with size "<<gbytes<<"GB\n";
    norm += dg::blas1::dot( 1., test_recursive);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas1::dot( 1., test_recursive);//warm up
    t.toc();
    DG_RANK0 std::cout<<"recursive dot took               " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    thrust::host_vector<value_type> test_serial((int)size_rec, (value_type)0.1);
    norm += dg::blas1::dot( test_serial, test_serial);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas1::dot( test_serial, test_serial);//warm up
    t.toc();
    DG_RANK0 std::cout<<"Serial dot took                  " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    //maybe test how fast a recursive axpby is compared to serial axpby
    dg::blas1::axpby( 1., test_recursive, 2., test_recursive);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpby( 1., test_recursive, 2., test_recursive);//warm up
    t.toc();
    DG_RANK0 std::cout<<"recursive axpby took             " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    //
    dg::blas1::axpby( 1., test_serial, 2., test_serial);//warm up
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpby( 1., test_serial, 2., test_serial);//warm up
    t.toc();
    DG_RANK0 std::cout<<"serial axpby took                " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
#endif

    DG_RANK0 std::cout << "\nUse of std::rotate and swap calls ( should not take any time)\n";
    t.tic();
    for( int i=0; i<multi; i++)
        std::rotate( x.rbegin(), x.rbegin()+1, x.rend()); //calls free swap functions
    t.toc();
    DG_RANK0 std::cout<<"Rotation        took             " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
    {
        using std::swap;
        swap( x[0], y[0]); //call free swap functions
    }
    t.toc();
    DG_RANK0 std::cout<<"std::sawp       took             " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
    // This call leads to warnings in MPI for unkown reasons
    //t.tic();
    //for( int i=0; i<multi; i++)
    //    std::iter_swap( x.begin(), x.end()); //calls free swap functions
    //t.toc();
    //DG_RANK0 std::cout<<"Swap            took             " <<t.diff()/multi<<"s\t"<<gbytes*multi/t.diff()<<"GB/s\n";
#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
