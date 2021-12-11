#include <iostream>
#include <cmath>
#include <cusp/print.h>
#include "dg/algorithm.h"
#include "fem.h"
#include "fem_weights.h"

double function( double x, double y){return sin(x)*cos(y);}

typedef dg::HVec Vector;
typedef cusp::coo_matrix<int,double,cusp::host_memory> Matrix;

int main ()
{
    unsigned n = 3, Nx = 20, Ny = 20, mx = 3;
    double eps = 1e-10;
    //std::cout << "# Type in n Nx Ny mx eps!\n";
    std::cout << "# on grid " << n <<" x "<<Nx<<" x "<<Ny<<"\n";
    std::cout << "# eps and Multiply " << eps <<" " << mx<<"\n";
    dg::CartesianGrid2d gDIR( 0, 2.*M_PI, M_PI/2., 5*M_PI/2., n, Nx, Ny, dg::DIR,
            dg::DIR);
    dg::FemRefinement fem_ref( mx);
    dg::CartesianRefinedGrid2d gDIR_f( fem_ref, fem_ref, gDIR.x0(), gDIR.x1(),
            gDIR.y0(), gDIR.y1(), n, Nx,Ny, dg::DIR, dg::DIR);
    const Vector func = dg::evaluate( function, gDIR);
    const Vector v2d = dg::create::fem_inv_weights( gDIR);
    const Vector w2d = dg::create::fem_weights( gDIR);
    const Vector wf2d = dg::create::volume( gDIR_f);
    double integral = dg::blas2::dot( func, w2d, func);
    std::cout << "error of integral is "
              <<(integral-M_PI*M_PI)/M_PI/M_PI<<std::endl;
    Vector Xf = dg::pullback( dg::cooX2d, gDIR_f);
    Vector Yf = dg::pullback( dg::cooY2d, gDIR_f);
    Matrix inter = dg::create::interpolation( Xf, Yf, gDIR, dg::NEU, dg::NEU, "linear");
    Matrix interT = dg::transpose( inter);
    Matrix Wf = dg::create::diagonal( wf2d), project;
    cusp::multiply( interT, Wf, project);
    //cusp::multiply( project, inter, interT);
    //project = interT;
    project.sort_by_row_and_column();
    //std::cout << "Project matrix\n";
    //cusp::print( project);


    Vector func_f( gDIR_f.size());
    dg::blas2::symv( inter, func, func_f);
    integral = dg::blas2::dot( func_f, wf2d, func_f);
    std::cout << "error of refined integral is "
              <<(integral-M_PI*M_PI)/M_PI/M_PI<<std::endl;
    Vector barfunc(func);
    dg::blas2::symv( project, func_f, barfunc);
    // test now should contain Sf
    Vector test( barfunc);
    dg::blas1::pointwiseDot( barfunc, v2d, test);
    dg::PCG<Vector> cg( test, 1000);
    Matrix fem_mass = dg::create::fem_mass( gDIR);
    //std::cout << "S matrix\n";
    //cusp::print( fem_mass);
    unsigned number = cg.solve( fem_mass, test, barfunc, v2d, eps);
    dg::blas1::axpby( 1., func, -1., test);
    double norm = sqrt(dg::blas2::dot( w2d, test) );
    double func_norm = sqrt(dg::blas2::dot( w2d, func) );
    std::cout <<"Distance to true solution: "<<norm/func_norm<<"\n";
    std::cout << "using "<<number<<" iterations\n";
    return 0;
}
