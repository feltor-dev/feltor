#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>

#include "implicit.h"
#include "elliptic.h"


template< class Geometry, class Matrix, class Container>
struct Diffusion
{
    Diffusion( Geometry& g, double nu, unsigned order)
    {
        m_temp[0] = dg::evaluate( dg::zero, g);
        m_temp[1] = dg::evaluate( dg::zero, g);
        enum dg::direction dir = dg::str2direction( "centered");
        m_LaplacianM.construct( g, dg::normed, dir, 1);
        m_nu = nu;
        m_order = order;
    }
    void operator()(double t, const Container& x, Container& y)
    {
        if( m_nu != 0)
        {
            dg::blas1::copy( x, m_temp[1]);
            for( unsigned p=0; p<m_order; p++)
            {
                using std::swap;
                swap( m_temp[0], m_temp[1]);
                dg::blas2::symv( m_nu, m_LaplacianM, m_temp[0], 0., m_temp[1]);
            }
            dg::blas1::axpby( -1., m_temp[1], 0., y);
        }
        else
            dg::blas1::scal( y, 0);
    }
    const Container& weights(){ return m_LaplacianM.weights();}
    const Container& inv_weights(){ return m_LaplacianM.inv_weights();}
    const Container& precond(){ return m_LaplacianM.precond();}
  private:
    unsigned m_order;
    double m_nu;
    Container m_temp[2];
    dg::Elliptic<Geometry, Matrix,Container> m_LaplacianM;
};

const double lx = M_PI;
const double ly = 2.*M_PI;
const double nu = 1e-3;
const unsigned order = 2;
const double alpha = -0.01;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;


double rhs( double x, double y) { return (1.-alpha*pow(2.*nu,order))*sin(x)*sin(y);}
double sol(double x, double y)  { return sin( x)*sin(y);}
double initial( double x, double y) {return rhs(x,y);}
//double initial( double x, double y) {return 0.;}


int main()
{
    unsigned n, Nx, Ny;
    double eps;
	n = 3;
	Nx = Ny = 48;
	eps = 1e-10;
    unsigned max_iter = 1000;

	/*std::cout << "Type n, Nx and Ny and epsilon and jfactor (1)! \n";
    std::cin >> n >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> eps >> jfactor;*/

    std::cout << "Computation on: "<< n <<" x "<< Nx <<" x "<< Ny << std::endl;
	dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
    dg::DVec w2d = dg::create::weights( grid);
    dg::DVec v2d = dg::create::inv_weights( grid);
    //create functions A(chi) x = b
    dg::DVec x =    dg::evaluate( initial, grid);
    const dg::DVec b =    dg::evaluate( rhs, grid);
    Diffusion<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> diff( grid, nu, order);

    //compute error
    const dg::DVec solution = dg::evaluate( sol, grid);
    const double norm = dg::blas2::dot( w2d, solution);
    dg::DVec error( solution);
    dg::exblas::udouble res;
    std::cout << "###########################################\n";
    std::cout << "Default Solver\n";
    dg::DefaultSolver<dg::DVec> solver( x, max_iter, eps);
    x  =    dg::evaluate( initial, grid);
    solver.solve( alpha, diff, 1., x, b);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm); res.d = err;
    std::cout << " Error "<<err << "\t"<<res.i<<"\n";
    //
    std::cout << "###########################################\n";
    std::cout << "Fixed Point Solver (does not converge in this example)\n";
    dg::FixedPointSolver<dg::DVec> fixed_solver( x, 100, eps);
    x  =    dg::evaluate( initial, grid);
    fixed_solver.solve( alpha, diff, 1., x, b);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm); res.d = err;
    std::cout << " Error "<<err << "\t"<<res.i<<"\n";
    //
    std::cout << "###########################################\n";
    std::cout << "Anderson Solver\n";
    unsigned mMax = 8;
    double damping = 1e-5;
    double restart = 8;
    //std::cout << "Type mMAx (8), damping ( 1e-5), restart (8)\n";
    //std::cin >> mMax >> damping >> restart;
    dg::AndersonSolver<dg::DVec> anderson( x, mMax, eps, max_iter, damping,
            restart);
    x  =    dg::evaluate( initial, grid);
    anderson.solve( alpha, diff, 1., x, b);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm); res.d = err;
    std::cout << " Error "<<err << "\t"<<res.i<<"\n";

    return 0;
}

