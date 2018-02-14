#include <iostream>

#include "../blas.h"
#include "dg/backend/derivatives.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/projection.cuh"
#include "transform.h"
#include "refined_grid.h"


double function( double x, double y){return sin(x)*cos(y);}
double derivative( double x, double y){return cos(x)*cos(y);}


int main ()
{
    std::cout<< "BOTH SIDES:\n";
    thrust::host_vector<double> left, right, both, left_abs, right_abs, both_abs;
    //don't forget to test the case add_x = 0 once in a while!
    dg::Grid1d g( 0,1, 2,4, dg::PER);
    int node;
    std::cout<< "Type node to refine 0,..,4!\n";
    std::cin >> node;
    dg::EquidistRefinement equi( 3, node, 1);
    int new_N = equi.N_new(g.N(), g.bcx());
    equi.generate( g, both, both_abs);
    double sum = 0;
    for( unsigned i=0; i<new_N*g.n(); i++)
    {
        std::cout << both[i] << "\t" <<both_abs[i] << std::endl;
        sum += 1./both[i]/g.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";
    std::cout<< "LEFT SIDE:\n";
    dg::Grid1d gl( 0,1, 2,5, dg::DIR);
    dg::EquidistRefinement equi2( 2, 0, 2);
    new_N = equi2.N_new(gl.N(), gl.bcx());
    equi2.generate( gl, left, left_abs);
    sum = 0;
    for( unsigned i=0; i<new_N*gl.n(); i++)
    {
        std::cout << left[i] <<"\t"<<left_abs[i]<<std::endl;
        sum += 1./left[i]/gl.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";
    std::cout<< "RIGHT SIDE:\n";
    dg::Grid1d gr( 0,1, 1, 5, dg::DIR);
    dg::EquidistRefinement equi3( 5, gr.N(), 2);
    new_N = equi2.N_new(gr.N(), gl.bcx());
    equi2.generate( gr, right, right_abs);
    sum =0;
    for( unsigned i=0; i<new_N*gr.n(); i++)
    {
        std::cout << right[i] <<"\t"<<both_abs[i]<< std::endl;
        sum += 1./right[i]/gr.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";

    dg::LinearRefinement lin(3);
    dg::CartesianRefinedGrid2d g2d_f( lin,lin, 0., 2*M_PI, 0., 2*M_PI, 5, 20, 20);
    dg::CartesianGrid2d g2d_c( 0., 2*M_PI, 0., 2*M_PI, 3, 20, 20);
    dg::HVec w2d_c = dg::create::weights( g2d_c);
    dg::HVec vec_c = dg::evaluate( function, g2d_c);
    dg::HVec vec = dg::pullback( function, g2d_f);
    dg::HVec w2d = dg::create::volume( g2d_f);
    double integral = dg::blas2::dot( vec, w2d, vec);
    std::cout << "error of fine integral is "<<integral-M_PI*M_PI<<std::endl;
    integral = dg::blas2::dot( vec_c, w2d_c, vec_c);
    std::cout << "error of coarse integral is "<<integral-M_PI*M_PI<<std::endl;
    dg::IHMatrix Q = dg::create::interpolation( g2d_f, g2d_c);
    dg::IHMatrix P = dg::create::projection( g2d_c, g2d_f);

    dg::HVec vec_cf(vec);
    dg::blas2::gemv( Q, vec_c, vec_cf);//here gemv instead of symv is important
    integral = dg::blas2::dot( vec_cf, w2d, vec_cf);
    std::cout << "error of interpolated integral is "<<integral-M_PI*M_PI<<std::endl;
    dg::blas2::gemv( P, vec_cf, vec_c);
    integral = dg::blas2::dot( vec_c, w2d_c, vec_c);
    std::cout << "error of projected integral is "<<integral-M_PI*M_PI<<std::endl;

    //also test P D^f_x Q f_c = D^c_x f 
    std::cout << "TEST OF P D_x^f Q = D_x^c\n";
    const dg::HMatrix dx_f = dg::create::dx( g2d_f);
    const dg::HMatrix dx_c = dg::create::dx( g2d_c);
    vec_c = dg::evaluate( function, g2d_c);
    dg::HVec deri_num(vec_c);
    dg::blas2::symv( dx_c, vec_c, deri_num);
    dg::blas2::gemv( Q, vec_c, vec_cf);
    dg::blas2::symv( dx_f, vec_cf, vec);
    dg::SparseTensor<thrust::host_vector<double> > jac=g2d_f.jacobian();
    dg::blas1::pointwiseDot( vec, jac.value(0,0), vec);
    dg::blas2::gemv( P, vec, vec_c);
    dg::blas1::axpby( 1., vec_c, -1., deri_num, vec_c);
    double error = dg::blas2::dot( vec_c, w2d_c, vec_c);
    std::cout << "error of derivative is "<<error<<std::endl;

    dg::CartesianRefinedGrid3d g3d_f( lin,lin,lin, 0., 2*M_PI, 0., 2*M_PI, 0., 2*M_PI, 5, 20, 20, 20);
    g3d_f.display();

    return 0;
}
