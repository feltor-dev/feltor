#include <iostream>

#include "refined_grid.h"
#include "../blas.h"
#include "dg/backend/derivatives.h"


double function( double x, double y){return sin(x)*cos(y);}
double derivative( double x, double y){return cos(x)*cos(y);}


int main ()
{
    std::cout<< "BOTH SIDES:\n";
    thrust::host_vector<double> left, right, both, left_abs, right_abs, both_abs;
    //don't forget to test the case add_x = 0 once in a while!
    dg::Grid1d<double> g( 0,1, 2,4, dg::PER);
    int node;
    std::cout<< "Type node to refine 0,..,4!\n";
    std::cin >> node;
    int new_N = dg::refined::detail::equidist_ref( 3, node, g, both, both_abs, 1);
    double sum = 0;
    for( unsigned i=0; i<new_N*g.n(); i++)
    {
        std::cout << both[i] << "\t" <<both_abs[i] << std::endl;
        sum += 1./both[i]/g.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";
    std::cout<< "LEFT SIDE:\n";
    dg::Grid1d<double> gl( 0,1, 2,5, dg::DIR);
    new_N = dg::refined::detail::equidist_ref( 2, 0, gl, left, left_abs,2 );
    sum = 0;
    for( unsigned i=0; i<new_N*gl.n(); i++)
    {
        std::cout << left[i] <<"\t"<<left_abs[i]<<std::endl;
        sum += 1./left[i]/gl.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";
    std::cout<< "RIGHT SIDE:\n";
    dg::Grid1d<double> gr( 0,1, 1, 5, dg::DIR);
    new_N = dg::refined::detail::equidist_ref( 5, gr.N(), gr, right, right_abs, 2);
    sum =0;
    for( unsigned i=0; i<new_N*gr.n(); i++)
    {
        std::cout << right[i] <<"\t"<<both_abs[i]<< std::endl;
        sum += 1./right[i]/gr.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";

    //dg::refined::Grid2d g2d_f( 0,0, 3,3, 2,2, 0., 2*M_PI, 0., 2*M_PI, 3, 20, 20);
    dg::refined::Grid2d g2d_f( 3,3, 0., 2*M_PI, 0., 2*M_PI, 5, 3, 20, 20);
    dg::Grid2d<double> g2d_c = g2d_f.associated();
    dg::HVec w2d_c = dg::create::weights( g2d_c);
    dg::HVec vec_c = dg::evaluate( function, g2d_c);
    dg::HVec vec( g2d_f.size());
    dg::HVec w2d = dg::create::weights( g2d_f);
    dg::blas1::pointwiseDivide( w2d, g2d_f.weightsX(), w2d);
    dg::blas1::pointwiseDivide( w2d, g2d_f.weightsY(), w2d);
    for( unsigned i=0; i<g2d_f.size(); i++)
        vec[i] = function( g2d_f.abscissasX()[i], g2d_f.abscissasY()[i]);
    double integral = dg::blas2::dot( vec, w2d, vec);
    std::cout << "error of fine integral is "<<integral-M_PI*M_PI<<std::endl;
    integral = dg::blas2::dot( vec_c, w2d_c, vec_c);
    std::cout << "error of coarse integral is "<<integral-M_PI*M_PI<<std::endl;
    dg::IHMatrix Q = dg::create::interpolation( g2d_f);
    dg::IHMatrix P = dg::create::projection( g2d_f);
    dg::IHMatrix S = dg::create::smoothing( g2d_f);

    dg::HVec vec_cf(vec);
    dg::blas2::gemv( Q, vec_c, vec_cf);//here gemv instead of symv is important
    integral = dg::blas2::dot( vec_cf, w2d, vec_cf);
    std::cout << "error of interpolated integral is "<<integral-M_PI*M_PI<<std::endl;
    dg::blas2::gemv( P, vec_cf, vec_c);
    integral = dg::blas2::dot( vec_c, w2d_c, vec_c);
    std::cout << "error of projected integral is "<<integral-M_PI*M_PI<<std::endl;

    dg::HVec smoothed(vec);
    dg::blas2::symv( S, vec, smoothed);
    integral = dg::blas2::dot( smoothed, w2d, smoothed);
    std::cout << "error of smoothed integral is "<<integral-M_PI*M_PI<<std::endl;
    //also test P D^f_x Q f_c = D^c_x f 
    std::cout << "TEST OF P D_x^f Q = D_x^c\n";
    const dg::HMatrix dx_f = dg::create::dx( g2d_f);
    const dg::HMatrix dx_c = dg::create::dx( g2d_c);
    vec_c = dg::evaluate( function, g2d_c);
    dg::HVec deri_num(vec_c);
    dg::blas2::symv( dx_c, vec_c, deri_num);
    dg::blas2::gemv( Q, vec_c, vec_cf);
    dg::blas2::symv( dx_f, vec_cf, vec);
    dg::blas1::pointwiseDot( vec, g2d_f.weightsX(), vec);
    dg::blas2::gemv( P, vec, vec_c);
    dg::blas1::axpby( 1., vec_c, -1., deri_num, vec_c);
    double error = dg::blas2::dot( vec_c, w2d_c, vec_c);
    std::cout << "error of derivative is "<<error<<std::endl;

    return 0;
}
