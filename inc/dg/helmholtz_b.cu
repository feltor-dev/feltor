#include <iostream>

#include <cusp/precond/diagonal.h>
#include <cusp/precond/ainv.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>

#include "blas.h"

#include "backend/timer.cuh"
#include "helmholtz.h"
#include "backend/xspacelib.cuh"

#include "cg.h"


const double eps = 1e-4;
const double alpha = -0.5; 
double lhs( double x, double y){ return sin(x)*sin(y);}
double rhs( double x, double y){ return (1.-2.*alpha)*sin(x)*sin(y);}
//double rhs( double x, double y){ return lhs(x,y);}
int main()
{
    
    unsigned n, Nx, Ny; 
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n>> Nx >> Ny;
    dg::Grid2d<double> grid( 0, 2.*M_PI, 0, 2.*M_PI, n, Nx, Ny, dg::DIR, dg::PER);
    const dg::DVec w2d = dg::create::weights( grid);
    const dg::DVec v2d = dg::create::inv_weights( grid);
    const dg::DVec one(grid.size(), 1.);
    dg::DVec rho = dg::evaluate( rhs, grid);
    const dg::DVec sol = dg::evaluate( lhs, grid);
    dg::DVec x(rho.size(), 0.);
    //dg::DVec x(rho);

    dg::Helmholtz< dg::DMatrix, dg::DVec, dg::DVec > gamma1( grid, alpha, dg::centered);

    dg::CG< dg::DVec > cg(x, x.size());
    dg::blas2::symv( w2d, rho, rho);
    dg::Timer t;
    t.tic();
    unsigned number = cg( gamma1, x, rho, v2d, eps);
    t.toc();
    dg::blas1::axpby( 1., sol, -1., x);
    std::cout << "DG   performance:\n";
    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "error " << sqrt( dg::blas2::dot( w2d, x))<<std::endl;
    std::cout << "took  " << t.diff()<<"s"<<std::endl;

//cusp matrix solver
    dg::Matrix Temp = dg::create::laplacianM( grid, dg::not_normed), diff;
    dg::Matrix weights( grid.size(), grid.size(), grid.size());
    thrust::sequence(weights.row_indices.begin(), weights.row_indices.end()); 
    thrust::sequence(weights.column_indices.begin(), weights.column_indices.end()); 
    thrust::copy( w2d.begin(), w2d.end(), weights.values.begin());
    for( unsigned i=0; i<Temp.values.size(); i++)
        Temp.values[i] = - alpha*Temp.values[i];
    cusp::add( weights, Temp, diff);
    dg::DMatrix diff_ = diff;

    cusp::array1d< double, cusp::device_memory> x_( x.size(), 0.);
    cusp::array1d< double, cusp::device_memory> b_( rho.begin(), rho.end());
    //cusp::verbose_monitor<double> monitor( b_, x.size(), eps, eps);
    cusp::default_monitor<double> monitor( b_, x.size(), eps, eps);
    //cusp::identity_operator<double, cusp::device_memory> M( diff_.num_rows, diff_.num_rows);
    cusp::precond::diagonal<double, cusp::device_memory> M( diff_);
    //cusp::precond::bridson_ainv<double, cusp::device_memory> M( diff_, 0.1, 10, true, 1);
    t.tic();
    cusp::krylov::cg( diff_, x_, b_, monitor, M);
    t.toc();
    dg::DVec xx_(x_.begin(), x_.end());
    dg::blas1::axpby( 1., sol, -1., xx_);
    std::cout << "CUSP performance:\n";
    std::cout << "error " << sqrt( dg::blas2::dot( w2d, xx_))<<std::endl;
    std::cout << "took  " << t.diff()<<"s"<<std::endl;



    return 0;
}



