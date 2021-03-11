#define SILENT
// #define DG_DEBUG
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"
#include "init.h"
#include "poet.cuh"
#include "dg/algorithm.h"
#include "dg/file/json_utilities.h"
#include "parameters.h"

#include "dg/polarization_init.h"
#include "dg/andersonacc.h"

/*
   - reads parameters from input.json or any other given file,
   - integrates the ToeflR - functor and
   - directly visualizes results on the screen using parameters in window_params.json
*/
using DVec = dg::DVec;
using DMatrix =  dg::DMatrix;



int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    std::stringstream title;
    Json::Value js;
    if( argc == 1)
        dg::file::file2Json( "input.json", js, dg::file::comments::are_forbidden);
    else if( argc == 2)
        dg::file::file2Json( argv[1], js, dg::file::comments::are_forbidden);
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const Parameters p( js);
    p.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    dg::file::file2Json( "window_params.json", js, dg::file::comments::are_discarded);
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["width"].asDouble(), js["height"].asDouble(), "");
    draw::RenderHostData render(js["rows"].asDouble(), js["cols"].asDouble());
    /////////////////////////////////////////////////////////////////////////

    dg::Grid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS
    poet::Explicit<dg::CartesianGrid2d, DMatrix, DVec> ex( grid, p);
    poet::Implicit<dg::CartesianGrid2d, DMatrix, DVec> im( grid, p.nu);
    //////////////////create initial vector///////////////////////////////////////
    std::vector<DVec> y0(2, dg::evaluate( dg::zero, grid)), y1(y0); // n_e' = gaussian

    if (p.init == "blob")
    {
        dg::Gaussian g( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp); 
        y0[0] = dg::evaluate(g, grid);
        ex.gamma1inv_y(y0[0],y0[1]); //no inversion -> smaller accuracy but n_e can be chosen instead of N_i!
//         y0[1] = dg::evaluate(g, grid);
//         ex.gamma1_y(y0[1], y0[0]); //invert Gamma operator for initialization with higher accuracy!
    }
    else if (p.init == "shearlayer")
    {
        ShearLayer layer(M_PI/15., 0.05, p.lx, p.ly); //shear layer
        std::vector<DVec> y0(2, dg::evaluate( layer, grid)), y1(y0);
        dg::blas1::scal(y0[0], p.amp);
        ex.invLap_y(y0[0], y1[0]); //phi 
        dg::blas1::scal(y0[0], 0.);
        ex.solve_Ni_lwl(y0[0], y1[0], y0[1]); //if df
        //Compute exact Ni with fixed point iteration
    //     dg::PolChargeN< dg::CartesianGrid2d, DMatrix, DVec > polN(grid, dg::DIR, dg::PER, dg::normed, dg::centered, 1.0, false);
    //     polN.set_phi(y1[0]);
    //     dg::AndersonAcceleration<DVec> acc( y1[0], 10000);
    // 
    //     dg::blas1::scal(y0[1], 0.0);
    //     dg::blas1::plus(y0[1], 1.0); //x solution must be positive 
    //     dg::blas1::scal(y0[0], 0.);  //ne_tilde = 0
    // 
    //     acc.solve( polN, y0[1], y0[0], im.weights(), 1e-4, 1e-4, grid.size(), 1e-13, 10000, true);    
    //     dg::blas1::plus(y0[1],-1.0);
    }
    else if (p.init == "rot_blob")
    {
//     //double rotating gaussian
//     dg::Gaussian g1( (0.5-p.posX)*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);
//     dg::Gaussian g2( (0.5+p.posX)*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);
// 
//     std::vector<DVec> y0(2, dg::evaluate( g1, grid)); // n_e' = gaussian
//     std::vector<DVec> y1(2, dg::evaluate( g2, grid)); // n_e' = gaussian
//     dg::blas1::axpby(1.0,y0[0],1.0,y1[0],y0[0]);
//     dg::blas1::axpby(10, y0[0], 0.0, y1[1]);
//     ex.invLap_y(y1[1], y1[0]); //phi 
//     ex.solve_Ni_lwl(y0[0], y1[0], y0[1]);
    }

    //////////////////////////////////////////////////////////////////////
    dg::ImExMultistep<std::vector<DVec>> stepper( "ImEx-TVB-3-3", y0, y0[0].size(), p.eps_time);
//     dg::Adaptive<dg::ARKStep<std::vector<DVec>>> stepper( "ARK-4-2-3", y0, y0[0].size(), p.eps_time);
//     dg::Adaptive<dg::ERKStep<std::vector<DVec>>> stepper( "Dormand-Prince-7-4-5", y0);

    DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    stepper.init( ex, im, time, y0, p.dt);
//     double dt = 1e-5;
    const double mass0 = ex.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = ex.energy(), energy0 = E0, E1 = 0, diff = 0;
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
        dvisual = ex.potential()[0];
        dg::blas2::gemv( ex.laplacianM(), dvisual, y1[1]);
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
                std::cout << "(m_tot-m_0)/m_0: "<< (ex.mass()-mass0)/mass_blob0<<"\t";
                E0 = E1;
                E1 = ex.energy();
                diff = (E1 - E0)/p.dt;
                double diss = ex.energy_diffusion( );
                std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";

            }
            try{ stepper.step( ex, im, time, y0);}
//             try{
// //                 std::cout << "Time "<<time<<" dt "<<dt<<" success "<<!stepper.failed()<<"\n";
// //                 stepper.step( ex, im, time, y0, time, y0, dt, dg::pid_control, dg::l2norm, 1e-7, 1e-14);
// //                 stepper.step( ex, time, y0, time, y0, dt, dg::pid_control, dg::l2norm, 1e-7, 1e-14);
//             }
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
