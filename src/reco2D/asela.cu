#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "asela.cuh"

#include "dg/runge_kutta.h"
#include "dg/multistep.h"
#include "dg/backend/timer.cuh"
#include "parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/

double aparallel( double x, double y)
{
    return 0.1/cosh(x)/cosh(x)*cos(1./8.*y);
}

int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Reader reader;
    Json::Value js;
    if( argc == 1)
    {
        std::ifstream is("input.json");
        reader.parse(is,js,false);
    }
    else if( argc == 2)
    {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const eule::Parameters p( js);
    p.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    std::stringstream title;
    std::ifstream is( "window_params.js");
    reader.parse( is, js, false);
    is.close();
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["width"].asDouble(), js["height"].asDouble(), "");
    draw::RenderHostData render(js["rows"].asDouble(), js["cols"].asDouble());
    /////////////////////////////////////////////////////////////////////////

    dg::Grid2d grid( -p.lxhalf, p.lxhalf, -p.lyhalf, p.lyhalf , p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    eule::Asela< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > asela( grid, p); 
    eule::Diffusion<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> diffusion( grid, p.nu, 1., 1. );
    //create initial vector
    std::vector<dg::DVec> y0(4, dg::evaluate( dg::one, grid)), y1(y0); // n_e' = gaussian
    y0[2] = y0[3] = dg::evaluate( aparallel, grid);
    dg::DVec temp( y0[2]);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> laplaceM(grid, dg::normed, dg::centered);
    dg::blas2::gemv( laplaceM, y0[2], temp); //u_e = \Delta A_parallel
    dg::blas1::axpby( p.dhat[0]*p.dhat[0], temp, 1., y0[2]);//w_e = \Delta A + beta/mue A
   
    for( unsigned i=0; i<2; i++)
        dg::blas1::transform( y0[i], y0[i], dg::LN<double>());

    dg::Karniadakis< std::vector<dg::DVec> > ab( y0, y0[0].size(), p.eps_time);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    //ab.init( asela, y0, p.dt);
    ab.init( asela, diffusion, y0, p.dt);
    //ab( asela, y0, y1, p.dt);
    //y0.swap( y1); 
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    while ( !glfwWindowShouldClose( w ))
    {
        for( unsigned i=0; i<2; i++)
            dg::blas1::transform( y0[i], y0[i], dg::EXP<double>());

        thrust::transform( y1[0].begin(), y1[0].end(), dvisual.begin(), dg::PLUS<double>(-1));
        dg::blas1::transfer(dvisual, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"ne / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        thrust::transform( y1[1].begin(), y1[1].end(), dvisual.begin(), dg::PLUS<double>(-1));
        dg::blas1::transfer(dvisual, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"ni / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        dvisual = asela.potential()[0];
        dg::blas1::transfer(dvisual, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        title <<"Potential / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        //transform phi
        dg::blas2::gemv(laplaceM, asela.potential()[0], dvisual);
        dg::blas1::transfer(dvisual, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        title <<"omega / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        //transform Aparallel
        dvisual = asela.aparallel();
        dg::blas1::transfer(dvisual, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        title <<"Aparallel / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        //transform Aparallel
        dg::blas2::gemv( laplaceM, asela.aparallel(), dvisual);
        dg::blas1::transfer(dvisual, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        title <<"Jpar / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        

        title << std::fixed; 
        title << " &&   time = "<<time;
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
            try{ ab( asela, diffusion, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;
            }
            //y0.swap( y1); //attention on -O3 ?
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
