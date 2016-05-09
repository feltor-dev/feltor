#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "toeflR.cuh"
#include "dg/algorithm.h"
#include "dg/backend/xspacelib.cuh"
#include "file/read_input.h"
#include "json/json.hpp"
#include "parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/


int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::stringstream title;
    nlohmann::json js;
    if( argc == 1)
    {
        std::ifstream fs( "input.txt");
        js = nlohmann::json::parse(fs);
    }
    else if( argc == 2)
    {
        std::ifstream fs(argv[1]);
        js = nlohmann::json::parse(fs);
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    std::cout <<std::setw(4)<<js<<std::endl;

    std::vector<double> v2 = file::read_input( "window_params.txt");
    GLFWwindow* w = draw::glfwInitAndCreateWindow( v2[3], v2[4], "");
    draw::RenderHostData render(v2[1], v2[2]);
    /////////////////////////////////////////////////////////////////////////

    dg::Grid2d<double > grid( 0, js["lx"], 0, js["ly"], js["n"], js["Nx"], js["Ny"], dg::str2bc(js["bc_x"]), dg::str2bc(js["bc_y"]));
    //create RHS 
    dg::ToeflR<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > test( grid, js["curvature"], js["nu_perp"], js["tau"], js["eps_pol"], js["eps_gamma"], 1); 
    dg::Diffusion<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> diffusion( grid, js["nu_perp"], 1);
    //create initial vector
    dg::Gaussian g( js["posX"].get<double>()*grid.lx(), js["posY"].get<double>()*grid.ly(), js["sigma"], js["sigma"], js["amplitude"]); //gaussian width is in absolute values
    std::vector<dg::DVec> y0(2, dg::evaluate( g, grid)), y1(y0); // n_e' = gaussian
    dg::blas2::symv( test.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    {
        dg::DVec v2d = dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[1], y0[1]);
    }


    //dg::AB< k, std::vector<dg::DVec> > ab( y0);
    dg::Karniadakis< std::vector<dg::DVec> > ab( y0, y0[0].size(), 1e-9);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    ab.init( test, diffusion, y0, js["dt"]);
    const double mass0 = test.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = test.energy(), energy0 = E0, E1 = 0, diff = 0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    while ( !glfwWindowShouldClose( w ))
    {
        //transform field to an equidistant grid
        dvisual = y0[0];

        dg::blas1::transfer( dvisual, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"ne / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //transform phi
        dvisual = test.potential()[0];
        dg::blas2::gemv( test.laplacianM(), dvisual, y1[1]);
        dg::blas1::transfer( y1[1], hvisual);
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
        unsigned itstp = js["itstp"];
        for( unsigned i=0; i<itstp; i++)
        {
            step++;
            {
                std::cout << "(m_tot-m_0)/m_0: "<< (test.mass()-mass0)/mass_blob0<<"\t";
                E0 = E1;
                E1 = test.energy();
                diff = (E1 - E0)/js["dt"].get<double>();
                double diss = test.energy_diffusion( );
                std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";

            }
            try{ ab( test, diffusion, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;
            }
        }
        time += (double)itstp*js["dt"].get<double>();
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "\n\t Step "<<step;
        std::cout << "\n\t Average time for one step: "<<t.diff()/(double)js["itstp"]<<"s\n\n";
#endif//DG_BENCHMARK
    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////

    return 0;

}
