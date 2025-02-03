#include <iostream>
#include <iomanip>

#include "andersonacc.h"
#include "elliptic.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;


double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}

int main()
{
    //global relative error in L2 norm is O(h^P)
    //more N means less iterations for same error
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;
    dg::Grid2d grid( 0, lx, 0, ly,n, Nx, Ny, dg::PER, dg::PER);
    std::cout<<"Evaluate initial condition\n";
    dg::HVec x = dg::evaluate( initial, grid);
    unsigned max_iter = n*n*Nx*Ny;
    const dg::HVec& copyable_vector = x;
    double damping;
    unsigned restart;
    std::cout << "Type damping (1e-3) and restart (10) \n";
    std::cin >> damping >> restart;

//! [doxygen]
    // create volume on previously defined grid
    const dg::HVec w2d = dg::create::weights( grid);

    // Create normalized Laplacian
    dg::Elliptic<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> A( grid);

    // allocate memory
    dg::AndersonAcceleration<dg::HVec > acc( copyable_vector, 10);

    // Evaluate right hand side and solution on the grid
    dg::HVec b = dg::evaluate ( laplace_fct, grid);
    const dg::HVec solution = dg::evaluate ( fct, grid);

    const double eps = 1e-6;
    std::cout << "Number of iterations "<< acc.solve( A, x, b, w2d, eps, eps, max_iter, damping, restart, true)<<std::endl;
//! [doxygen]
    std::cout << "For a precision of "<< eps<<std::endl;
    //compute error
    dg::HVec error( solution);
    dg::blas1::axpby( 1.,x,-1.,error);

    dg::HVec Ax(x), resi( b);
    dg::blas2::symv(  A, x, Ax);
    dg::blas1::axpby( 1.,Ax,-1.,resi);

    dg::exblas::udouble res;
    res.d = sqrt(dg::blas2::dot( w2d, x));
    std::cout << "L2 Norm of x0 is              " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot(w2d , solution));
    std::cout << "L2 Norm of Solution is        " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot(w2d , error));
    std::cout << "L2 Norm of Error is           " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot( w2d, resi));
    std::cout << "L2 Norm of Residuum is        " << res.d<<"\t"<<res.i << std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)
    return 0;
}
