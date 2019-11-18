#include <iostream>
#include <iomanip>

#include "cg.h"
#include "bicgstabl.h"
#include "lgmres.h"
#include "elliptic.h"
#include "chebyshev.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;


double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}

template<class Matrix, class Container>
void solve( std::string solver, Matrix& A, Container& x, const Container& b, const dg::Grid2d& grid)
{
    unsigned n = grid.n(), Nx = grid.Nx(), Ny = grid.Ny();
    if( "cheby" == solver)
    {
        dg::ChebyshevIteration<Container> cheby( x);
        double lmin = 1+1, lmax = n*n*Nx*Nx + n*n*Ny*Ny; //Eigenvalues of Laplace
        double hxhy = lx*ly/(n*n*Nx*Ny);
        lmin *= hxhy, lmax *= hxhy; //we multiplied the matrix by w2d
        //std::cout << "Type number of Chebyshev iterations\n";
        unsigned num_iter =200;
        //std::cin >> num_iter;
        cheby.solve( A, x, b, lmin, lmax, num_iter);
        std::cout << "After "<<num_iter<<" Chebyshev iterations we have:\n";
    }
    if( "bicgstabl" == solver)
    {
        dg::BICGSTABl<Container> bicg( x, 100, 2);
        unsigned num_iter = bicg.solve( A, x, b, A.precond(), A.inv_weights(), 1e-6);
        std::cout << "After "<<num_iter<<" BICGSTABl iterations we have:\n";
    }
    if( "lgmres" == solver)
    {
        dg::LGMRES<Container> lgmres( x, 30, 4, 10000);
        unsigned num_iter = lgmres.solve( A, x, b, A.precond(), A.inv_weights(), 1e-6);
        std::cout << "After "<<num_iter<<" LGMRES iterations we have:\n";
    }

}

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

//! [doxygen]
    // create volume and inverse volume on previously defined grid
    const dg::HVec w2d = dg::create::weights( grid);
    const dg::HVec v2d = dg::create::inv_weights( grid);

    // Create unnormalized Laplacian
    dg::Elliptic<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> A( grid);

    // allocate memory in conjugate gradient
    dg::CG<dg::HVec > pcg( copyable_vector, max_iter);

    // Evaluate right hand side and solution on the grid
    dg::HVec b = dg::evaluate ( laplace_fct, grid);
    const dg::HVec solution = dg::evaluate ( fct, grid);

    // normalize right hand side
    dg::blas2::symv( w2d, b, b);

    // use inverse volume as preconditioner in solution method
    const double eps = 1e-6;
    unsigned num_iter = pcg( A, x, b, v2d, eps);
//! [doxygen]
    std::cout << "Number of pcg iterations "<< num_iter<<std::endl;
    std::cout << "For a precision of "<< eps<<std::endl;
    //compute error
    dg::HVec error( solution);
    dg::blas1::axpby( 1.,x,-1.,error);

    dg::HVec Ax(x), resi( b);
    dg::blas2::symv(  A, x, Ax);
    dg::blas1::axpby( 1.,Ax,-1.,resi);

    exblas::udouble res;
    res.d = sqrt(dg::blas2::dot( w2d, x));
    std::cout << "L2 Norm of x0 is              " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot(w2d , solution));
    std::cout << "L2 Norm of Solution is        " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot(w2d , error));
    std::cout << "L2 Norm of Error is           " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot( w2d, resi));
    std::cout << "L2 Norm of Residuum is        " << res.d<<"\t"<<res.i << std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)

    std::vector<std::string> solvers{ "cheby", "bicgstabl", "lgmres"};
    for(auto solver : solvers)
    {
        dg::blas1::copy( 0., x);
        solve( solver, A, x, b, grid);
        dg::blas1::copy( solution, error);
        dg::blas1::axpby( 1.,x,-1.,error);

        dg::blas1::copy( b, resi);
        dg::blas2::symv(  A, x, Ax);
        dg::blas1::axpby( 1.,Ax,-1.,resi);

        res.d = sqrt(dg::blas2::dot( w2d, x));
        std::cout << "L2 Norm of x0 is              " << res.d<<"\n";
        res.d = sqrt(dg::blas2::dot(w2d , solution));
        std::cout << "L2 Norm of Solution is        " << res.d<<"\n";
        res.d = sqrt(dg::blas2::dot(w2d , error));
        std::cout << "L2 Norm of Error is           " << res.d<<"\n";
        res.d = sqrt(dg::blas2::dot( w2d, resi));
        std::cout << "L2 Norm of Residuum is        " << res.d<<"\n\n";
    }


    return 0;
}
