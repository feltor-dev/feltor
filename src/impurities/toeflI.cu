#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "toeflI.cuh"
#include "dg/backend/timer.cuh" 
#include "file/read_input.h"
#include "../toefl/parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflI - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/

const unsigned k = 3; //!< a change of k needs a recompilation!

int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::vector<double> v, v2;
    std::stringstream title;
    if( argc == 1)
    {
        v = file::read_input("input.txt");
    }
    else if( argc == 2)
    {
        v = file::read_input( argv[1]);
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }

    v2 = file::read_input( "window_params.txt");
    GLFWwindow* w = draw::glfwInitAndCreateWindow( v2[3], v2[4], "");
    draw::RenderHostData render(v2[1], v2[2]);
    /////////////////////////////////////////////////////////////////////////
    const Parameters p( v, 2);
    p.display( std::cout);
    if( p.k != k)
    {
        std::cerr << "ERROR: k doesn't match: "<<k<<" vs. "<<p.k<<"\n";
        return -1;
    }

    dg::Grid2d<double > grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    dg::ToeflI< dg::DVec > test( grid, p.kappa, p.nu, p.tau, p.a_z, p.mu_z, p.tau_z, p.eps_pol, p.eps_gamma); 

    //create initial vector
    dg::Gaussian g( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); //gaussian width is in absolute values
    std::vector<dg::DVec> y0(3, dg::DVec( grid.size()) ), y1(y0);
    //dg::blas1::axpby( 1., y0[0], 1., (dg::DVec)dg::evaluate( g, grid), y0[0]);//n_e = 1+ gaussian
    dg::Helmholtz<dg::DMatrix, dg::DVec, dg::DVec>& gamma = test.gamma();
    if( v[25] == 1)
    {
        gamma.alpha() = -0.5*p.tau;
        y0[0] = dg::evaluate( g, grid);
        dg::blas2::symv( gamma, y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1 
        dg::DVec v2d=dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[1], y0[1]);
        dg::blas1::axpby( 1./(1.-p.a_z), y0[1], 0., y0[1]); //n_i ~1./a_i n_e
        y0[2] = dg::evaluate( dg::one, grid);
        dg::blas1::axpby( 1., y0[2], 1., y0[0]);
        dg::blas1::axpby( 1., y0[2], 1., y0[1]);
    }
    if( v[25] == 2) 
    {
        //init wall in y0[2]
        dg::GaussianX wall( v[26]*grid.lx(), v[28], v[27]); //position, sigma, amplitude
        dg::DVec wallv = dg::evaluate( wall, grid);
        gamma.alpha() = -0.5*p.tau_z*p.mu_z;
        dg::blas2::symv( gamma, wallv, y0[2]); 
        dg::DVec v2d=dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[2], y0[2]);
        if( p.a_z != 0.)
            dg::blas1::axpby( 1./p.a_z, y0[2], 0., y0[2]); //n_z ~1./a_z

        //init blob in y0[1]
        gamma.alpha() = -0.5*p.tau;
        y0[0] = dg::evaluate( g, grid);
        dg::blas2::symv( gamma, y0[0], y0[1]); 
        dg::blas2::symv( v2d, y0[2], y0[2]);
        dg::blas2::symv( v2d, y0[1], y0[1]);
        if( p.a_z == 1)
        {
            std::cerr << "No blob with trace ions possible!\n";
            return -1;
        }
        dg::blas1::axpby( 1./(1-p.a_z), y0[1], 0., y0[1]); //n_i ~1./a_i n_e

        //sum up
        if( p.a_z != 0)
            dg::blas1::axpby( 1., wallv, 1., y0[0]); //add wall to blob in n_e
        dg::DVec one = dg::evaluate( dg::one, grid);
        for( unsigned i=0; i<3; i++)
            dg::blas1::axpby( 1., one, 1., y0[i]);
        
    }
    if( v[25] == 3) 
    {
        gamma.alpha() = -0.5*p.tau_z*p.mu_z;
        y0[0] = dg::evaluate( g, grid);
        dg::blas2::symv( gamma, y0[0], y0[2]); 
        dg::DVec v2d=dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[2], y0[2]);
        if( p.a_z == 0)
        {
            std::cerr << "No impurity blob with trace impurities possible!\n";
            return -1;
        }
        dg::blas1::axpby( 1./p.a_z, y0[2], 0., y0[2]); //n_z ~1./a_z n_e
        y0[1] = dg::evaluate( dg::one, grid);
        dg::blas1::axpby( 1., y0[1], 1., y0[0]);
        dg::blas1::axpby( 1., y0[1], 1., y0[2]);
    }

    test.log( y0, y0); //transform to logarithmic values

    dg::AB< k, std::vector<dg::DVec> > ab( y0);
    //dg::TVB< std::vector<dg::DVec> > ab( y0);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    ab.init( test, y0, p.dt);
    ab( test, y1);
    const double mass0 = test.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = test.energy(), energy0 = E0, E1 = 0, diff = 0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    while ( !glfwWindowShouldClose( w ))
    {
        //transform field to an equidistant grid
        test.exp( y1, y1);
        title << std::setprecision(2) << std::scientific;
        for( unsigned i=0; i<y1.size(); i++)
        {
            thrust::transform( y1[i].begin(), y1[i].end(), dvisual.begin(), dg::PLUS<double>(-1));

            hvisual = dvisual;
            dg::blas2::gemv( equi, hvisual, visual);
            //compute the color scale
            colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
            if( colors.scale() == 0) 
                colors.scale() = 1.;
            //draw ions
            title <<"n / "<<colors.scale()<<"\t";
            render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        //transform phi
        dg::blas2::gemv( test.laplacianM(), test.potential()[0], y1[1]);
        hvisual = y1[1];
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        title <<"omega / "<<colors.scale()<<"\t";
        title << std::fixed; 
        title << " &&   time = "<<time;
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        glfwSetWindowTitle(w,title.str().c_str());
        title.str("");
        glfwPollEvents();
        glfwSwapBuffers( w);

        //step 
#ifdef DG_BENCHMARK
        t.tic();
#endif//DG_BENCHMARK
        for( unsigned i=0; i<p.itstp; i++)
        {
            step++;
                std::cout << "(m_tot-m_0)/m_0: "<< (test.mass()-mass0)/mass_blob0<<"\t";
                E0 = E1;
                E1 = test.energy();
                diff = (E1 - E0)/p.dt;
                double diss = test.energy_diffusion( );
                std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";

            try{ ab( test, y1);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;
            }
        }
        time += (double)p.itstp*p.dt;
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "\n\t Step "<<step;
        std::cout << "\n\t Average time for one step: "<<t.diff()/(double)p.itstp<<"s\n\n";
#endif//DG_BENCHMARK
    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////

    return 0;

}
