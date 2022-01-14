#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "toeflR.cuh"
#include "dg/algorithm.h"
#include "dg/file/json_utilities.h"
#include "parameters.h"


int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::stringstream title;
    Json::Value js;
    if( argc == 1)
        dg::file::file2Json("input.json", js, dg::file::comments::are_discarded);
    else if( argc == 2)
        dg::file::file2Json(argv[1], js, dg::file::comments::are_discarded);
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const Parameters p( js);
    p.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    dg::file::file2Json("window_params.json", js, dg::file::comments::are_discarded);
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["width"].asDouble(), js["height"].asDouble(), "");
    draw::RenderHostData render(js["rows"].asDouble(), js["cols"].asDouble());
    /////////////////////////////////////////////////////////////////////////

    dg::CartesianGrid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    ep::ToeflR<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > exp( grid, p); 
    ep::Diffusion<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> imp( grid, p.nu);
    //create initial vector
    dg::Gaussian g( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp); //gaussian width is in absolute values
    dg::DVec gauss = dg::evaluate( g, grid);
    std::vector<dg::DVec> y0(2, gauss), y1(y0); // n_e' = gaussian
    dg::Helmholtz<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> gamma(  grid, -0.5*p.tau[0]*p.mu[0], dg::centered);
    dg::blas2::symv( gamma, gauss, y0[0]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    gamma.alpha() = -0.5*p.tau[1]*p.mu[1];
    dg::blas2::symv( gamma, gauss, y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1

    dg::DefaultSolver<std::vector<dg::DVec>> solver( imp, y0, 1000, 1e-9);
    dg::ImExMultistep< std::vector<dg::DVec> > karniadakis( "ImEx-BDF-3-3", y0);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    karniadakis.init( std::tie(exp, imp, solver), time, y0, p.dt);
    const double mass0 = exp.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = exp.energy(), energy0 = E0, E1 = 0, diff = 0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    while ( !glfwWindowShouldClose( w ))
    {
        //transform field to an equidistant grid
        dvisual = y0[0];

        dg::assign( dvisual, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"ne / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //transform phi
        dvisual = exp.potential();
        dg::blas2::gemv( exp.laplacianM(), dvisual, y1[1]);
        dg::assign( y1[1], hvisual);
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
            {
                std::cout << "(m_tot-m_0)/m_0: "<< (exp.mass()-mass0)/mass_blob0<<"\t";
                E0 = E1;
                E1 = exp.energy();
                diff = (E1 - E0)/p.dt;
                double diss = exp.energy_diffusion( );
                std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";

            }
            try{ karniadakis.step( std::tie( exp, imp, solver), time, y0);}
            catch( dg::Fail& fail) {
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;
            }
        }
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
