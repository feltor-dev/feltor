#include <iostream>

#include "refined_grid.h"
#include "../blas.h"
#include "dg/backend/typedefs.cuh"


double function( double x, double y){return sin(x)*cos(y);}


int main ()
{
    std::cout<< "BOTH SIDES:\n";
    thrust::host_vector<double> left, right, both, left_abs, right_abs, both_abs;
    //don't forget to test the case add_x = 0 once in a while!
    dg::Grid1d<double> g( 0,1, 2,4, dg::PER);
    int node;
    std::cout<< "Type node to refine 0,..,4!\n";
    std::cin >> node;
    int new_N = dg::refined::detail::exponential_ref( 3, node, g, both, both_abs);
    double sum = 0;
    for( unsigned i=0; i<new_N*g.n(); i++)
    {
        std::cout << both[i] << "\t" <<both_abs[i] << std::endl;
        sum += 1./both[i]/g.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";
    std::cout<< "LEFT SIDE:\n";
    dg::Grid1d<double> gl( 0,1, 2,5, dg::DIR);
    new_N = dg::refined::detail::exponential_ref( 2, 0, gl, left, left_abs);
    sum = 0;
    for( unsigned i=0; i<new_N*gl.n(); i++)
    {
        std::cout << left[i] <<"\t"<<left_abs[i]<<std::endl;
        sum += 1./left[i]/gl.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";
    std::cout<< "RIGHT SIDE:\n";
    dg::Grid1d<double> gr( 0,1, 1, 5, dg::DIR);
    new_N = dg::refined::detail::exponential_ref( 5, gr.N(), gr, right, right_abs);
    sum =0;
    for( unsigned i=0; i<new_N*gr.n(); i++)
    {
        std::cout << right[i] <<"\t"<<both_abs[i]<< std::endl;
        sum += 1./right[i]/gr.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";

    dg::refined::Grid2d g2d( 0,0,3,3, 0., 2*M_PI, 0., 2*M_PI, 3, 20, 20);
    dg::HVec vec( g2d.size());
    dg::HVec w2d = dg::create::weights( g2d);
    dg::blas1::pointwiseDivide( w2d, g2d.weightsX(), w2d);
    dg::blas1::pointwiseDivide( w2d, g2d.weightsY(), w2d);
    for( unsigned i=0; i<g2d.size(); i++)
        vec[i] = function( g2d.abscissasX()[i], g2d.abscissasY()[i]);
    double integral = dg::blas2::dot( vec, w2d, vec);
    std::cout << "error of integral is "<<integral-M_PI*M_PI<<std::endl;
    dg::IHMatrix Q = dg::create::interpolation( g2d);
    dg::Grid2d<double> g2d_c = g2d.associated();
    dg::HVec w2d_c = dg::create::weights( g2d_c);
    dg::HVec vec_c = dg::evaluate( function, g2d_c);
    integral = dg::blas2::dot( vec_c, w2d_c, vec_c);
    std::cout << "error of integral is "<<integral-M_PI*M_PI<<std::endl;
    dg::HVec vec_cf(vec);
    dg::blas2::gemv( Q, vec_c, vec_cf);//here gemv instead of symv is important
    integral = dg::blas2::dot( vec_cf, w2d, vec_cf);
    std::cout << "error of interpolated integral is "<<integral-M_PI*M_PI<<std::endl;
    dg::IHMatrix P = dg::create::projection( g2d);
    dg::blas2::gemv( P, vec_cf, vec_c);
    integral = dg::blas2::dot( vec_c, w2d_c, vec_c);
    std::cout << "error of projected integral is "<<integral-M_PI*M_PI<<std::endl;

    dg::IHMatrix S = dg::create::smoothing( g2d);
    dg::HVec smoothed(vec);
    dg::blas2::symv( S, vec, smoothed);
    integral = dg::blas2::dot( smoothed, w2d, smoothed);
    std::cout << "error of smoothed integral is "<<integral-M_PI*M_PI<<std::endl;

    return 0;
}
