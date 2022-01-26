#include <iostream>
#include <iomanip>

#include "pcg.h"
#include "eve.h"
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
    double lmin = 1+1, lmax = 2*grid.size(); //Eigenvalues of Laplace
    if( "eve cg" == solver)
    {
        std::cout <<" EVE SOLVER:\n";
        dg::EVE<Container> eve( x);
        std::cout << "L_min     "<<lmin<<" L_max     "<<lmax<<"\n";
        double eve_max;
        unsigned counter = eve.solve( A, x, b, 1., A.weights(), eve_max, 1e-10);
        std::cout << "Maximum EV mod "<<eve_max<<" after "<<counter<<" EVE iterations\n";
    }
    if( "eve pcg" == solver)
    {
        std::cout <<" PRECONDITIONED EVE SOLVER:\n";
        dg::EVE<Container> eve( x);
        std::cout << "L_min     "<<lmin<<" L_max     "<<lmax<<"\n";
        double eve_max;
        unsigned counter = eve.solve( A, x, b, A.precond(), A.weights(), eve_max, 1e-10);
        std::cout << "Maximum EV     "<<eve_max<<" after "<<counter<<" EVE iterations\n";
    }
    if( "cheby" == solver)
    {
        std::cout <<" CHEBYSHEV SOLVER:\n";
        dg::ChebyshevIteration<Container> cheby( x);
        unsigned num_iter =200;
        cheby.solve( A, x, b, lmin, lmax/2., num_iter);
        std::cout << "After "<<num_iter<<" Chebyshev iterations we have:\n";
    }
    if( "P cheby" == solver)
    {
        std::cout <<" PRECONDITIONED CHEBYSHEV SOLVER:\n";
        dg::ChebyshevIteration<Container> cheby( x);
        unsigned num_iter =200;
        cheby.solve( A, x, b, A.precond(), lmin, lmax/2., num_iter);
        std::cout << "After "<<num_iter<<" Chebyshev iterations we have:\n";
    }
    if( "bicgstabl" == solver)
    {
        std::cout <<" BICGSTABl SOLVER:\n";
        dg::BICGSTABl<Container> bicg( x, 100, 2);
        unsigned num_iter = bicg.solve( A, x, b, A.precond(), A.weights(), 1e-6);
        std::cout << "After "<<num_iter<<" BICGSTABl iterations we have:\n";
    }
    if( "lgmres" == solver)
    {
        std::cout <<" LGMRES SOLVER:\n";
        dg::LGMRES<Container> lgmres( x, 30, 4, 10000);
        unsigned num_iter = lgmres.solve( A, x, b, A.precond(), A.weights(), 1e-6);
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

    std::cout <<" PCG SOLVER:\n";
//! [doxygen]
    // create volume and inverse volume on previously defined grid
    const dg::HVec w2d = dg::create::weights( grid);

    // Create unnormalized Laplacian
    dg::Elliptic<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> A( grid);

    // allocate memory in conjugate gradient
    dg::PCG<dg::HVec > pcg( copyable_vector, max_iter);

    // Evaluate right hand side and solution on the grid
    dg::HVec b = dg::evaluate ( laplace_fct, grid);
    const dg::HVec solution = dg::evaluate ( fct, grid);

    // use inverse volume as preconditioner in solution method
    const double eps = 1e-6;
    unsigned num_iter = pcg.solve( A, x, b, 1., w2d, eps);
//! [doxygen]
    std::cout << "Number of pcg iterations "<< num_iter<<std::endl;
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
    std::cout << "L2 Norm of Residuum is        " << res.d<<"\t"<<res.i << std::endl<<std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)

    std::vector<std::string> solvers{ "eve cg", "eve pcg", "cheby", "P cheby", "bicgstabl", "lgmres"};
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
