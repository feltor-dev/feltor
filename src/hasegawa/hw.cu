#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "hw.cuh"
#include "../toefl/parameters.h"
#include "dg/file/json_utilities.h"


int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    std::stringstream title;
    Json::Value js;
    if( argc == 1)
        dg::file::file2Json( "input.json", js, dg::file::comments::are_discarded);
    else if( argc == 2)
        dg::file::file2Json( argv[1], js, dg::file::comments::are_discarded);
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
    dg::CartesianGrid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    bool mhw = (p.equations == "modified");
    hw::HW<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > test( grid, p.kappa,
            p.tau, p.nu, p.eps_pol, mhw); 
    dg::DVec one( grid.size(), 1.);
    //create initial vector
    dg::Gaussian gaussian( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.amp); //gaussian width is in absolute values
    dg::Vortex vortex( p.posX*grid.lx(), p.posY*grid.ly(), 0, p.sigma, p.amp);
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
    dg::ImExMultistep_s<std::vector<dg::DVec> > ab( "ImEx-BDF-3-3", y0, y0[0].size(), p.eps_time);
    hw::Diffusion<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> diffusion( grid, p.nu);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    ab.init( test, diffusion, time, y0, p.dt);
    double E0 = test.energy(), E1 = 0, diff = 0; //energy0 = E0;
    double Ezf0 = test.zonal_flow_energy(), Ezf1 = 0, diffzf = 0; //energyzf0 = Ezf0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> laplacianM(grid,  dg::centered);
    while ( !glfwWindowShouldClose( w ))
    {
        if( p.bc_x == dg::PER && p.bc_y == dg::PER)
        {
            double meanMass = dg::blas2::dot( y0[0], w2d, one)/(double)(p.lx*p.ly);
            std::cout << "Mean Mass is "<<meanMass<<"\n";
        }
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

        dg::blas2::gemv( laplacianM, test.potential(), y1[1]);
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
            try{ ab.step( test, diffusion, time, y0);}
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
