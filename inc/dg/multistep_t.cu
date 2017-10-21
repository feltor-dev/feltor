#include <iostream>

#include "multistep.h"
#include "elliptic.h"

//![function]
template<class Vector>
void zero(const std::vector<Vector>& y, std::vector<Vector>& yp){ dg::blas1::scal(yp,0.);}

template< class Matrix, class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d& g, double nu): m_nu(nu),
        m_w2d( dg::create::weights(g)), 
        m_v2d( dg::create::inv_weights(g)),
        m_LaplacianM( g, dg::normed) 
        { }

    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        for(unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( m_LaplacianM, x[i], y[i]);
        }
        dg::blas1::scal( y, -m_nu);
    }
    const container& inv_weights(){return m_v2d;}
    const container& weights(){return m_w2d;}
    const container& precond(){return m_v2d;}
  private:
    double m_nu;
    const container m_w2d, m_v2d;
    dg::Elliptic<dg::CartesianGrid2d, Matrix, container> m_LaplacianM;
};
double sine( double x, double y) {return sin(x)*sin(y);}
const double T = 1.0;
const double nu = 0.01;
double sol( double x, double y) {return exp( -2.*nu*T)*sine(x, y);}
//![function]


const unsigned n = 3;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

//const unsigned NT = (unsigned)(nu*T*n*n*N*N/0.01/lx/lx);



int main()
{
    double dt, NT, eps;
    unsigned Nx, Ny;
    std::cout << "Type Nx (20), Ny (20) and timestep (0.1) and eps( 1e-8)!\n";
    std::cin >> Nx >> Ny >> dt >> eps;
    NT = (unsigned)(T/dt);

    std::cout << "Test Karniadakis scheme on diffusion equation\n";
    std::cout << "Number of gridpoints:     "<<Nx*Ny<<std::endl;
    std::cout << "# of timesteps:           "<<NT<<std::endl;

    //![doxygen]
    dg::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
    std::vector<dg::DVec> y0(2, dg::evaluate( sine, grid)), y1(y0);
    Diffusion<dg::DMatrix, dg::DVec> diffusion( grid, nu);
    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), eps);
    karniadakis.init( zero<dg::DVec>, diffusion, y0, dt);
    dg::SIRK< std::vector<dg::DVec> > sirk( y0, y0[0].size(), eps);
    for( unsigned i=0; i<NT; i++)
    {
        karniadakis( zero<dg::DVec>, diffusion, y0);
        //sirk( explicit, diffusion, y0, y1, dt);
        //y0.swap(y1);
    }
    //![doxygen]
    dg::DVec w2d = dg::create::weights( grid);
    double norm_y0 = dg::blas2::dot( w2d, y0[0]);
    std::cout << "Normalized y0 after "<< NT <<" steps is "<< norm_y0 << std::endl;
    dg::DVec solution = dg::evaluate( sol, grid), error( solution);
    double norm_sol = dg::blas2::dot( w2d, solution);
    dg::blas1::axpby( -1., y0[0], 1., error);
    std::cout << "Normalized solution is "<<  norm_sol<< std::endl;
    double norm_error = dg::blas2::dot( w2d, error);
    std::cout << "Relative error is      "<< sqrt( norm_error/norm_sol)<<" (0.0020084 Karniadakis) (0.000148647 SIRK)\n";
    //n = 1 -> p = 1 (Sprung in laplace macht n=1 eine Ordng schlechter) 
    //n = 2 -> p = 2
    //n = 3 -> p = 3
    //n = 4 -> p = 4
    //n = 5 -> p = 5

    return 0;
}
