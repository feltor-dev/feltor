#include <iostream>
#include <cmath>
#include <cusp/print.h>
#include "dg/blas.h"
#include "dg/functors.h"
#include "dg/cg.h"
#include "fem_dx.h"
#include "fem_weights.h"

double function( double x) { return sin(x);}
double derivative( double x) { return cos(x);}
double zero( double x) { return 0;}

typedef dg::HVec Vector;
typedef cusp::coo_matrix<int,double,cusp::host_memory> Matrix;

int main ()
{
    unsigned n, N;
    double eps = 1e-8;
    std::cout << "# Type in n Nx eps!\n";
    std::cin >> n>> N >> eps;
    std::cout << "# of Legendre nodes " << n <<"\n";
    std::cout << "# of cells          " << N <<"\n";
    std::cout << "# eps               " << eps <<"\n";
    dg::Grid1d gPER( 0.1, 2*M_PI+0.1, n, N, dg::PER);
    dg::Grid1d gDIR( 0, M_PI, n, N, dg::DIR);
    dg::Grid1d gNEU( M_PI/2., 3*M_PI/2., n, N, dg::NEU);
    dg::Grid1d gDIR_NEU( 0, M_PI/2., n, N, dg::DIR_NEU);
    dg::Grid1d gNEU_DIR( M_PI/2., M_PI, n, N, dg::NEU_DIR);
    dg::Grid1d g[] = {gPER, gDIR, gNEU, gDIR_NEU,gNEU_DIR};
    std::string names[] = {"PER", "DIR", "NEU", "DIR_NEU", "NEU_DIR"};
    const Vector func = dg::evaluate( function, gDIR);
    const Vector w1d = dg::create::fem_weights( gDIR);
    std::cout << "TEST FEM WEIGHTS:\n";
    std::cout << "Distance: "<<2-dg::blas1::dot( func, w1d)<<"\n";


    std::cout << "TEST FEM TOPOLOGY: YOU SHOULD SEE CONVERGENCE FOR ALL OUTPUTS!!!\n";
    for( unsigned i=0; i<5; i++)
    {
        Matrix hs = dg::create::fem_dx( g[i]);
        //cusp::print(hs);
        Vector func = dg::evaluate( function, g[i]), error(func);
        const Vector w1d = dg::create::fem_weights( g[i]);
        const Vector v1d = dg::create::fem_inv_weights( g[i]);
        const Vector deri = dg::evaluate( derivative, g[i]);
        const Vector null = dg::evaluate( zero, g[i]);

        dg::blas2::symv( hs, func, error);
        Matrix fem_mass = dg::create::fem_mass( g[i]);
        Vector test_weights( w1d);
        Vector one = dg::evaluate( dg::one, g[i]);
        dg::blas2::symv( fem_mass, one, test_weights);
        dg::blas1::axpby( 1., test_weights, -1., w1d, test_weights);
        std::cout << "Distance S 1 = W : "<<sqrt( dg::blas2::dot( w1d, test_weights))<<"\n";
        dg::blas1::pointwiseDot( error, v1d, func);
        //cusp::print(fem_mass);
        dg::CG<Vector> cg( error, 1000);

        unsigned number = 0;
        dg::blas1::pointwiseDot( error, v1d, func);
        number = cg.solve( fem_mass, func, error, v1d, v1d, eps);
        dg::blas1::axpby( 1., deri, -1., func);
        double norm = sqrt(dg::blas2::dot( w1d, func) );
        double deri_n = sqrt(dg::blas2::dot( w1d, deri) );
        std::cout << names[i]<<": Distance to true solution: "<<norm/deri_n<<"\n";
        std::cout << "using "<<number<<" iterations\n";
    }

    return 0;
}
