#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "hw.cuh"
#include "dg/multistep.h"
#include "dg/backend/timer.cuh"
#include "file/read_input.h"
#include "../toefl/parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
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
    const Parameters p( v);
    p.display( std::cout);
    if( p.k != k)
    {
        std::cerr << "ERROR: k doesn't match: "<<k<<" vs. "<<p.k<<"\n";
        return -1;
    }

    dg::Grid2d<double > grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    dg::HW< dg::DVec > test( grid, p.kappa, p.tau, p.nu, p.eps_pol, (bool)p.global); 
    dg::DVec one( grid.size(), 1.);
    //create initial vector
    dg::Gaussian gaussian( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); //gaussian width is in absolute values
    dg::Vortex vortex( p.posX*grid.lx(), p.posY*grid.ly(), 0, p.sigma, p.n0);
    std::vector<dg::DVec> y0(2, dg::evaluate( vortex, grid)), y1(y0); // n_e' = gaussian
    dg::DVec w2d( dg::create::weights( grid));

    if( p.bc_x == dg::PER && p.bc_y == dg::PER)
    {
        double meanMass = dg::blas2::dot( y0[0], w2d, one)/(double)(p.lx*p.ly);
        std::cout << "Mean Mass is "<<meanMass<<"\n";
        dg::blas1::axpby( -meanMass, one, 1., y0[0]);
        y0[1] = y0[0];
    }
    //dg::AB< k, std::vector<dg::DVec> > ab( y0);
    //dg::TVB< std::vector<dg::DVec> > ab( y0);
    dg::Karniadakis<std::vector<dg::DVec> > ab( y0, y0[0].size(), 1e-9);
    dg::Diffusion<dg::DVec> diffusion( grid, p.nu);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    ab.init( test, diffusion, y0, p.dt);
    //ab( test, y0, y1, p.dt);
    //y0.swap( y1); 
    double E0 = test.energy(), energy0 = E0, E1 = 0, diff = 0;
    double Ezf0 = test.zonal_flow_energy(), energyzf0 = Ezf0, Ezf1 = 0, diffzf = 0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    while ( !glfwWindowShouldClose( w ))
    {
        if( p.bc_x == dg::PER && p.bc_y == dg::PER)
        {
            double meanMass = dg::blas2::dot( y0[0], w2d, one)/(double)(p.lx*p.ly);
            std::cout << "Mean Mass is "<<meanMass<<"\n";
        }
        //transform field to an equidistant grid
        dvisual = y0[0];

        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"ne / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //transform phi
        dg::blas2::gemv( test.laplacianM(), test.potential(), y1[1]);
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
            {
                if( p.bc_x == dg::PER && p.bc_y == dg::PER)
                {
                    double meanMass = dg::blas2::dot( y0[0], w2d, one)/(double)(p.lx*p.ly);
                    dg::blas1::axpby( -meanMass, one, 1., y0[0]);
                    meanMass = dg::blas2::dot( y0[1], w2d, one)/(double)(p.lx*p.ly);
                    dg::blas1::axpby( -meanMass, one, 1., y0[1]);
                }
                E0 = E1; Ezf0 = Ezf1;
                E1 = test.energy(); Ezf1 = test.zonal_flow_energy();
                diff = (E1 - E0)/p.dt; diffzf = (Ezf1-Ezf0)/p.dt;
                double diss = test.energy_diffusion( ) + test.flux() + test.capital_jot(); double disszf = test.zonal_flow_diffusion() + test.capital_r();
                //std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                std::cout << "Accuracy:   "<< 2.*(diff-diss)/(diff+diss)<<"\n";
                std::cout << "AccuracyZF: "<< 2.*(diffzf-disszf)/(diffzf+disszf)<<"\n";
                std::cout << diff << " "<< test.energy_diffusion() << " " <<test.flux()<<" "<<test.capital_jot()<<"\n";
                std::cout << diffzf << " "<< test.zonal_flow_diffusion() << " " <<test.capital_r()<<"\n";

            }
            try{ ab( test, diffusion, y0);}
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
