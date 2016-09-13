#include <iostream>

#include "file/read_input.h"
#include "file/nc_utilities.h"

#include "dg/geometry/refined_grid.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/grid.h"
#include "dg/elliptic.h"
#include "dg/cg.h"

#include "solovev.h"
//#include "guenther.h"
#include "orthogonal.h"



int main(int argc, char**argv)
{
    std::cout << "Type n, Nx, Ny\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;   
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    std::vector<double> v, v2;
    try{ 
        if( argc==1)
        {
            v = file::read_input( "geometry_params_Xpoint.txt"); 
        }
        else
        {
            v = file::read_input( argv[1]); 
        }
    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;}
    //write parameters from file into variables
    solovev::GeomParameters gp(v);
    gp.display( std::cout);
    dg::Timer t;
    solovev::Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    double eps = 1e10, eps_old = 2e10;

    orthogonal::RingGrid2d<dg::DVec> g2d_old(gp, psi_0, psi_1, n, Nx, Ny,dg::NEU);
    dg::Elliptic<orthogonal::RingGrid2d<dg::DVec>, dg::DMatrix, dg::DVec> elliptic_old( g2d_old, dg::DIR_NEU, dg::PER, dg::not_normed, dg::centered);
    dg::DVec x_old = dg::evaluate( dg::zero, g2d_old);
    dg::DVec b = g2d_old.lapy();
    dg::Invert<dg::DVec > invert_old( x_old, n*n*Nx*Ny, 1e-10);
    unsigned number = invert_old( elliptic_old, x_old,b);
    while( (eps < eps_old||eps > 1e-7) && eps > 1e-13)
    {
        eps = eps_old;
        Nx*=2, Ny*=2;
        t.tic();
        orthogonal::RingGrid2d<dg::DVec> g2d(gp, psi_0, psi_1, n, Nx, Ny,dg::NEU);
        t.toc();
        std::cout << "Grid construction took "<<t.diff()<<"\n";
        dg::Elliptic<orthogonal::RingGrid2d<dg::DVec>, dg::DMatrix, dg::DVec> elliptic( g2d, dg::DIR_NEU, dg::PER, dg::not_normed, dg::centered);
        b = g2d.lapy();
        const dg::DVec vol2d = dg::create::weights( g2d);

        const dg::IDMatrix Q = dg::create::interpolation( g2d, g2d_old);
        dg::DVec x = dg::evaluate( dg::zero, g2d), x_diff( x);
        dg::blas2::gemv( Q, x_old, x_diff);

        dg::Invert<dg::DVec > invert( x_diff, n*n*Nx*Ny, 1e-10);
        t.tic();
        number = invert( elliptic, x,b);
        t.toc();
        std::cout << number << " iterations took "<<t.diff()<<"\n";
        double norm = sqrt( dg::blas2::dot( x, vol2d, x));
        dg::blas1::axpby( 1. ,x, -1., x_diff);
        eps = sqrt( dg::blas2::dot( x_diff, vol2d, x_diff) / norm );
        std::cout << "Nx "<<Nx<<" Ny "<<Ny<<" error "<<eps<<"\n";
        g2d_old = g2d;
        x_old = x;
    }
    return 0;
}
