#include <iostream>

#include "../blas.h"
#include "derivatives.h"
#include "derivativesT.h"
#include "interpolation.h"
#include "projection.h"
#include "transform.h"
#include "refined_grid.h"

#include "catch2/catch_all.hpp"

static double function( double x, double y){return sin(x)*cos(y);}

TEST_CASE( "Refinement")
{
    SECTION("both sides")
    {
        //don't forget to test the case add_x = 0 once in a while!
        dg::Grid1d g( 0,1, 2,4, dg::PER);
        int node = 3;
        //INFO("Type node to refine 0,..,4!\n";
        //std::cin >> node;
        dg::EquidistRefinement equi( 3, node, 1);
        int new_N = equi.N_new(g.N(), g.bcx());
        dg::HVec both, both_abs;
        equi.generate( g, both, both_abs);
        double sum = 0;
        for( unsigned i=0; i<new_N*g.n(); i++)
        {
            INFO( both[i] << "\t" <<both_abs[i]);
            sum += 1./both[i]/g.n();
        }
        INFO( "SUM IS: "<<sum<<" ("<<new_N<<")");
        CHECK( sum == new_N);
    }
    SECTION("left side")
    {
        dg::Grid1d gl( 0,1, 2,5, dg::DIR);
        dg::EquidistRefinement equi2( 2, 0, 2);
        int new_N = equi2.N_new(gl.N(), gl.bcx());
        dg::HVec left, left_abs;
        equi2.generate( gl, left, left_abs);
        double sum = 0;
        for( unsigned i=0; i<new_N*gl.n(); i++)
        {
            INFO( left[i] <<"\t"<<left_abs[i]);
            sum += 1./left[i]/gl.n();
        }
        INFO( "SUM IS: "<<sum<<" ("<<new_N<<")");
        CHECK( sum == new_N);
    }
    SECTION("right side")
    {
        dg::Grid1d gr( 0,1, 1, 5, dg::DIR);
        dg::EquidistRefinement equi3( 5, gr.N(), 2);
        int new_N = equi3.N_new(gr.N(), gr.bcx());
        dg::HVec right, right_abs;
        equi3.generate( gr, right, right_abs);
        double sum =0;
        for( unsigned i=0; i<new_N*gr.n(); i++)
        {
            INFO( right[i] <<"\t"<<right_abs[i]);
            sum += 1./right[i]/gr.n();
        }
        INFO( "SUM IS: "<<sum<<" ("<<new_N<<")");
        CHECK( sum  == new_N);
    }
    SECTION( "Linear Refinement")
    {
        dg::LinearRefinement lin(3);
        dg::CartesianRefinedGrid2d g2d_f( lin,lin, 0., 2*M_PI, 0., 2*M_PI, 5, 20, 20);
        dg::CartesianGrid2d g2d_c( 0., 2*M_PI, 0., 2*M_PI, 3, 20, 20);
        dg::HVec w2d_c = dg::create::weights( g2d_c);
        dg::HVec vec_c = dg::evaluate( function, g2d_c);
        dg::HVec vec = dg::pullback( function, g2d_f);
        dg::HVec w2d = dg::create::volume( g2d_f);
        double integral = dg::blas2::dot( vec, w2d, vec);
        INFO( "error of fine integral is "<<integral-M_PI*M_PI);
        CHECK( integral - M_PI*M_PI == 0);
        integral = dg::blas2::dot( vec_c, w2d_c, vec_c);
        INFO( "error of coarse integral is "<<integral-M_PI*M_PI);
        CHECK( integral - M_PI*M_PI == 0);
        dg::IHMatrix Q = dg::create::interpolation( g2d_f, g2d_c);
        dg::IHMatrix P = dg::create::projection( g2d_c, g2d_f);

        dg::HVec vec_cf(vec);
        dg::blas2::gemv( Q, vec_c, vec_cf);//here gemv instead of symv is important
        integral = dg::blas2::dot( vec_cf, w2d, vec_cf);
        INFO( "error of interpolated integral is "<<integral-M_PI*M_PI);
        CHECK( integral - M_PI*M_PI == 0);
        dg::blas2::gemv( P, vec_cf, vec_c);
        integral = dg::blas2::dot( vec_c, w2d_c, vec_c);
        INFO( "error of projected integral is "<<integral-M_PI*M_PI);
        CHECK( integral - M_PI*M_PI == 0);
    }

    //also test P D^f_x Q f_c = D^c_x f
    SECTION( "P D_x^f Q = D_x^c")
    {
        dg::LinearRefinement lin(3);
        dg::CartesianGrid2d g2d_c( 0., 2*M_PI, 0., 2*M_PI, 3, 20, 20);
        dg::CartesianRefinedGrid2d g2d_f( lin,lin, 0., 2*M_PI, 0., 2*M_PI, 5,
                20, 20);
        dg::HVec vec = dg::pullback( function, g2d_f);
        const dg::HMatrix dx_f = dg::create::dx( g2d_f, g2d_f.bcx());
        const dg::HMatrix dx_c = dg::create::dx( g2d_c, g2d_c.bcx());
        dg::IHMatrix Q = dg::create::interpolation( g2d_f, g2d_c);
        dg::IHMatrix P = dg::create::projection( g2d_c, g2d_f);
        auto vec_c = dg::evaluate( function, g2d_c);
        auto w2d_c = dg::create::weights( g2d_c);
        dg::HVec deri_num(vec_c);
        dg::blas2::symv( dx_c, vec_c, deri_num);
        dg::HVec vec_cf(vec);
        dg::blas2::gemv( Q, vec_c, vec_cf);
        dg::blas2::symv( dx_f, vec_cf, vec);
        dg::SparseTensor<thrust::host_vector<double> > jac=g2d_f.jacobian();
        dg::blas1::pointwiseDot( vec, jac.value(0,0), vec);
        dg::blas2::gemv( P, vec, vec_c);
        dg::blas1::axpby( 1., vec_c, -1., deri_num, vec_c);
        double error = sqrt(dg::blas2::dot( vec_c, w2d_c, vec_c));
        INFO( "error of derivative is "<<error);
        CHECK( error < 1e-13);
    }

    SECTION( "Finite element REFINEMENT")
    {
        dg::LinearRefinement lin(3);
        dg::CartesianRefinedGrid3d g3d_f( lin,lin,lin, 0., 2*M_PI, 0., 2*M_PI,
                0., 2*M_PI, 5, 20, 20, 20);
        unsigned n = 5, Nx = 20, Ny = 20;
        for( unsigned m = 1; m<8; m++)
        {
            INFO( "Refinement factor = "<<m);
            dg::FemRefinement fem_ref( m);
            dg::CartesianRefinedGrid2d g2d_f( fem_ref, fem_ref, 0.1, 0.1+2*M_PI,
                    0.1, 0.1+2*M_PI, n, Nx,Ny);
            dg::HVec vec = dg::pullback( function, g2d_f);
            dg::HVec w2d = dg::create::volume( g2d_f);
            double integral = dg::blas2::dot( vec, w2d, vec);
            INFO( "error of fine integral is "
                    <<(integral-M_PI*M_PI)/M_PI/M_PI);
            CHECK( fabs( integral-M_PI*M_PI)/M_PI/M_PI < 1e-10);
        }
    }

}
