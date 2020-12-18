#undef DG_BENCHMARK


#include <iostream>
#include <iomanip>

#include "lanczos.h"
#include "helmholtz.h"

#include "backend/timer.h"

const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::NEU;

double initial( double x, double y) {return sin(x)*sin(y);}


using dia_type =  cusp::dia_matrix<int, double, cusp::host_memory>;
using coo_type =  cusp::coo_matrix<int, double, cusp::host_memory>;

int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n >> Nx >> Ny;
    unsigned iter;
    std::cout << "# of iterations\n";
    std::cin >> iter;
    dg::CartesianGrid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
    
    const dg::HVec w2d = dg::create::weights( grid);
    const dg::HVec v2d = dg::create::inv_weights( grid);
        
    dg::HVec x = dg::evaluate( initial, grid), b(x), zero(x), one(x), bexac(x);
    dg::blas1::scal(zero, 0.0);
    one = dg::evaluate(dg::one, grid);
//     dg::HVec chi = dg::evaluate( dg::LinearX(1.0,1.0), grid);
    dg::Helmholtz<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> A( grid, -0.5, dg::centered); //not_normed
//     A.set_chi(chi);
    
    //Create Lancsos class
    t.tic();
    dg::Lanczos< dg::HVec > lanczos(x, iter);
    t.toc();
    std::cout << "Creation of Lanczos  took "<< t.diff()<<"s   \n";

    //Execute lanczos algorithm
    dia_type T; 
    coo_type V, Vt;
    std::pair<dia_type, coo_type> TVpair; 
    //M-Lanczos method
    double xnorm = sqrt(dg::blas2::dot(x,w2d,x));
    t.tic();
    TVpair = lanczos(A, x, b, w2d, v2d); 
    t.toc();
    //Lanczos method
//     double xnorm = sqrt(dg::blas1::dot(x,x));
//     t.tic();
//     TVpair = lanczos(A, x, b); 
//     t.toc();   
    T = TVpair.first; 
    V = TVpair.second;
    cusp::transpose(V, Vt);
    
    //Compute error
    dg::blas2::symv(A, x, bexac);
    dg::blas2::symv( v2d, bexac, bexac); //normalize operator

    dg::blas1::axpby(-1.0, bexac, 1.0, b);
    std::cout << "# of Lanczos Iterations: "<< iter <<" | time: "<< t.diff()<<"s \n";
    std::cout << "Relative error (method1):" << sqrt(dg::blas2::dot(w2d, b)/dg::blas2::dot(w2d, bexac)) << "  ";
    
    //overwrite b from Lanczos to check V and T (should be exactly the same than method 1)
    dg::HVec e1( iter, 0.), temp(e1);
    e1[0]=1.;
    dg::blas2::symv(T, e1, temp); //T e_1
    dg::blas2::symv(V, temp, b); // V T e_1
    dg::blas1::scal(b, xnorm ); 
    dg::blas1::axpby(-1.0, bexac, 1.0, b);
    std::cout << "Relative error (method2):" << sqrt(dg::blas2::dot(w2d, b)/dg::blas2::dot(w2d, bexac)) << "  ";

    // overwrite b from Lanczos to check V and T and V^T - get worse for more iterations due to loss of orthognality?
    dg::blas2::symv( w2d, x,x); //normalize
    dg::blas2::symv(Vt, x, e1); //V^T x
    dg::blas2::symv(T, e1, temp); //T V^T x
    dg::blas2::symv(V, temp, b); // V T V^T x

    dg::blas1::axpby(-1.0, bexac, 1.0, b);
    std::cout << "Relative error (method3):" << sqrt(dg::blas2::dot(w2d, b)/dg::blas2::dot(w2d, bexac)) << "\n";    
    
    return 0;
}
