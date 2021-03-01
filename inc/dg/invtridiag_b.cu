// #undef DG_BENCHMARK
// #define DG_DEBUG

#include <iostream>
#include <iomanip>

#include "lanczos.h"

#include "backend/timer.h"
#include <cusp/transpose.h>
#include <cusp/print.h>
#include <cusp/array2d.h>
#include <cusp/elementwise.h>
#include <cusp/blas/blas.h>
#include "cg.h"
#include "lgmres.h"

using value_type = double;
using memory_type = cusp::host_memory;
using CooMatrix =  cusp::coo_matrix<int, value_type, memory_type>;
using DiaMatrix =  cusp::dia_matrix<int, value_type, memory_type>;
using Container = dg::HVec;
int main()
{
    dg::Timer t;
    unsigned size = 50;
    std::cout << "#Specify size of vectors (50)\n";
    std::cin >> size;
    //vectors of the tridiagonal matrix
//     std::vector<value_type> a = {1.98242, 4.45423, 5.31867, 7.48144, 7.11534};
//     std::vector<value_type> b = {-0.00710891, -0.054661, -0.0554193, -0.0172191, -0.297645};
//     std::vector<value_type> c = {-1.98242, -4.44712, -5.26401, -7.42602, -7.09812}; 
    std::cout << "#Constructing and filling vectors\n";
    std::vector<value_type> a(size,1.);
    std::vector<value_type> b(size,1.);
    std::vector<value_type> c(size,1.);
    for (unsigned i=0;i<size; i++)
    {
        a[i] = (1.+0.1*i);
        b[i] = 1./(1.+0.1*i);
        c[i] = (1.+0.1*i)+1./(1.+0.1*i);
    }
    std::cout << "#Constructing and filling containers\n";
    const Container d(size,1.);
    Container x(size,0.), x_sol(x), err(x);
    std::cout << "#Constructing Matrix inversion and linear solvers\n";
    value_type eps= 1e-14;
    t.tic();
    dg::CG <Container> pcg( x,  size*size);
    t.toc();
    std::cout << "#Construction of CG took "<< t.diff()<<"s \n";
    t.tic();
    dg::LGMRES <Container> lgmres( x, 30, 3, 1000*size);
    t.toc();
    std::cout << "#Construction of LGMRES took "<< t.diff()<<"s \n";
    t.tic();
    dg::InvTridiag<Container, DiaMatrix, CooMatrix> invtridiag(a);
    t.toc();
    std::cout << "#Construction of Tridiagonal inversion routine took "<< t.diff()<<"s \n";
    
    //Create Tridiagonal and fill matrix
    DiaMatrix T, Tsym; 
    T.resize(size, size, 3*size-2, 3);
    T.diagonal_offsets[0] = -1;
    T.diagonal_offsets[1] =  0;
    T.diagonal_offsets[2] =  1;
    Tsym.resize(size, size, 3*size-2, 3);
    Tsym.diagonal_offsets[0] = -1;
    Tsym.diagonal_offsets[1] =  0;
    Tsym.diagonal_offsets[2] =  1;
    
    for( unsigned i=0; i<size-1; i++)
    {
        T.values(i,1)   =  a[i];  // 0 diagonal
        T.values(i+1,0) =  c[i];  // -1 diagonal
        T.values(i,2)   =  b[i];  // +1 diagonal //dia_rows entry works since its outside of matrix
        Tsym.values(i,1)   =  a[i];  // 0 diagonal
        Tsym.values(i+1,0) =  b[i];  // -1 diagonal
        Tsym.values(i,2)   =  b[i];  // +1 diagonal //dia_rows entry works since its outside of matrix
    }
    T.values(size-1,1) =  a[size-1];
    Tsym.values(size-1,1) =  a[size-1];
//     std::cout << "T matrix\n";
//     cusp::print(T);
//     std::cout << "Tsym matrix\n";
//     cusp::print(Tsym);
    
    //Create and fill Inverse of tridiagonal matrix (the solution)
    CooMatrix Tinv, Tsyminv;
//     cusp::array2d<value_type ,memory_type> H(size,size), error(size,size,0.);
//     H(0,0) = 0.505249;  H(1,0) = 0.000814795;  H(2,0) =  8.4358e-6;     H(3,0) = 6.26392e-8;     H(4,0) =  1.51587e-10;
//     H(0,1) = 0.227217;  H(1,1) = 0.227217;     H(2,1) =  0.00235244;    H(3,1) = 0.0000174678;   H(4,1) =  4.22721e-8;
//     H(0,2) = 0.19139;   H(1,2) = 0.19139;      H(2,2) =  0.19139;       H(3,2) = 0.00142115;     H(4,2) =  3.43918e-6;
//     H(0,3) = 0.134988;  H(1,3) = 0.134988;     H(2,3) =  0.134988;      H(3,3) = 0.134988;       H(4,3) =  0.000326671;
//     H(0,4) = 0.140882;  H(1,4) = 0.140882;     H(2,4) =  0.140882;      H(3,4) = 0.140882;       H(4,4) = 0.140882;
//     CooMatrix Tinv_sol, Tinv_error;
//     cusp::convert(H,Tinv_sol);
//     cusp::convert(error,Tinv_error);
    
    

    std::cout << "####Compute inverse of symmetric tridiagonal matrix\n";
    dg::blas1::scal(x_sol, 0.);
    t.tic();
    unsigned number = pcg( Tsym, x_sol, d, d, eps);
    if(  number == pcg.get_max())
        throw dg::Fail( eps);
    t.toc();
    std::cout <<  "#CG took: "<< t.diff()<<"s \n";
    t.tic();
    Tsyminv = invtridiag(a,b,b);
    t.toc();
    dg::blas2::gemv(Tsyminv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "#Invtridiag with vectors took: "<< t.diff()<<"s \n";
    std::cout <<  "#Relative error to CG: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";

    t.tic();
    Tsyminv = invtridiag(Tsym);
    t.toc();
    dg::blas2::gemv(Tsyminv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "#Invtridiag with Matrix took: "<< t.diff()<<"s \n";
    std::cout <<  "#Relative error to CG: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";
    

    std::cout << "####Compute inverse of non-symmetric tridiagonal matrix\n";
    t.tic();
    number = lgmres.solve( T, x_sol, d , d, d, eps, 1);    
    t.toc();
    std::cout <<  "#lGMRES took: "<< t.diff()<<"s \n";
    
    t.tic();
    Tinv = invtridiag(a,b,c);
    t.toc();
    dg::blas2::gemv(Tinv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "#Invtridiag with vectors took: "<< t.diff()<<"s \n";
    std::cout <<  "#Relative error to lGMRES: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";
    t.tic();
    Tinv = invtridiag(T);
    t.toc();
    dg::blas2::gemv(Tinv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "#Invtridiag with Matrix took: "<< t.diff()<<"s \n";
    std::cout <<  "#Relative error to lGMRES: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";    
    return 0;
}
