#include <iostream>

#include "dg/blas.h"
#include "dx.h"
#include "evaluation.h"
#include "weights.h"


double function( double x) { return sin(x);}
double derivative( double x) { return cos(x);}
double zero( double x) { return 0;}

typedef dg::HVec Vector;
typedef dg::EllSparseBlockMat<double> Matrix;

int main ()
{
    unsigned n, N;
    std::cout << "Type in n and Nx!\n";
    std::cin >> n>> N;
    std::cout << "# of Legendre nodes " << n <<"\n";
    std::cout << "# of cells          " << N <<"\n";
    dg::Grid1d gPER( 0.1, 2*M_PI+0.1, n, N, dg::PER);
    dg::Grid1d gDIR( 0, M_PI, n, N, dg::DIR);
    dg::Grid1d gNEU( M_PI/2., 3*M_PI/2., n, N, dg::NEU);
    dg::Grid1d gDIR_NEU( 0, M_PI/2., n, N, dg::DIR_NEU);
    dg::Grid1d gNEU_DIR( M_PI/2., M_PI, n, N, dg::NEU_DIR);
    dg::Grid1d g[] = {gPER, gDIR, gNEU, gDIR_NEU,gNEU_DIR};

    std::cout << "TEST NORMAL TOPOLOGY: YOU SHOULD SEE CONVERGENCE FOR ALL OUTPUTS!!!\n";
    for( unsigned i=0; i<5; i++)
    {
        Matrix hs = dg::create::dx( g[i], dg::centered);
        Matrix hf = dg::create::dx( g[i], dg::forward);
        Matrix hb = dg::create::dx( g[i], dg::backward);
        Matrix js = dg::create::jump( g[i].n(), g[i].N(), g[i].h(), g[i].bcx());
        const Vector func = dg::evaluate( function, g[i]);
        Vector error = func;
        const Vector w1d = dg::create::weights( g[i]);
        const Vector deri = dg::evaluate( derivative, g[i]);
        const Vector null = dg::evaluate( zero, g[i]);

        dg::blas2::symv( hs, func, error);
        dg::blas1::axpby( 1., deri, -1., error);
        std::cout << "Distance to true solution (symmetric): "<<sqrt(dg::blas2::dot( w1d, error) )<<"\n";
        dg::blas2::symv( hf, func, error);
        dg::blas1::axpby( 1., deri, -1., error);
        std::cout << "Distance to true solution (forward  ): "<<sqrt(dg::blas2::dot( w1d, error) )<<"\n";
        dg::blas2::symv( hb, func, error);
        dg::blas1::axpby( 1., deri, -1., error);
        std::cout << "Distance to true solution (backward ): "<<sqrt(dg::blas2::dot( w1d, error) )<<"\n";
        dg::blas2::symv( js, func, error);
        dg::blas1::axpby( 1., null , -1., error);
        std::cout << "Distance to true solution (jump     ): "<<sqrt(dg::blas2::dot( w1d, error) )<<"\n\n";
    }
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5



    return 0;
}
