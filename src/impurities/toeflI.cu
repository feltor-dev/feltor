#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"


#include "dg/backend/xspacelib.cuh"
#include "toeflI.cuh"
#include "dg/backend/timer.cuh" 
#include "parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflI - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/

int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::stringstream title;
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
    const imp::Parameters p( js);
    p.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    std::ifstream is( "window_params.js");
    reader.parse( is, js, false);
    is.close();
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["width"].asDouble(), js["height"].asDouble(), "");
    draw::RenderHostData render(js["rows"].asDouble(), js["cols"].asDouble());
    /////////////////////////////////////////////////////////////////////////
    ////////////////////////////////set up computations///////////////////////////
    dg::CartesianGrid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    dg::Diffusion< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > diffusion( grid, p); 
    dg::ToeflI< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > toeflI( grid, p); 

    //create initial vector
    dg::Gaussian gaussian( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.amp); //gaussian width is in absolute values
    std::vector<dg::DVec> y0(3, dg::DVec( grid.size()) );
    dg::Helmholtz<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> & gamma = toeflI.gamma();
    if( p.mode == 1)
    {
        if( p.vorticity == 0)
        {
            gamma.alpha() = -0.5*p.tau[1];
            y0[0] = dg::evaluate( gaussian, grid);
            dg::blas2::symv( gamma, y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1 
            dg::DVec v2d=dg::create::inv_weights(grid);
            dg::blas2::symv( v2d, y0[1], y0[1]);

            dg::blas1::scal( y0[1], 1./p.a[1]); //n_i ~1./a_i n_e
            y0[2] = dg::evaluate( dg::zero, grid);
        }
        else
        {
            y0[1] = y0[0] = dg::evaluate( gaussian, grid);
            dg::blas1::scal( y0[1], 1/p.a[1]);
            y0[2] = dg::evaluate( dg::zero, grid);
        }
    }
    if( p.mode == 2) 
    {
        //init wall in y0[2]
        dg::GaussianX wall( p.wall_pos*grid.lx(), p.wall_sigma, p.wall_amp); 
        dg::DVec wallv = dg::evaluate( wall, grid);
        gamma.alpha() = -0.5*p.tau[2]*p.mu[2];
        dg::blas2::symv( gamma, wallv, y0[2]); 
        dg::DVec v2d=dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[2], y0[2]);
        if( p.a[2] != 0.)
            dg::blas1::scal( y0[2], 1./p.a[2]); //n_z ~1./a_z

        //init blob in y0[1]
        gamma.alpha() = -0.5*p.tau[1];
        y0[0] = dg::evaluate( gaussian, grid);
        dg::blas2::symv( gamma, y0[0], y0[1]); 
        dg::blas1::pointwiseDot( v2d, y0[1], y0[1]);
        if( p.a[2] == 1)
        {
            std::cerr << "No blob with trace ions possible!\n";
            return -1;
        }
        dg::blas1::scal( y0[1], 1./p.a[1]); //n_i ~1./a_i n_e

        //sum up
        if( p.a[2] != 0)
            dg::blas1::axpby( 1., wallv, 1., y0[0]); //add wall to blob in n_e
        
    }
    if( p.mode == 3) 
    {
        gamma.alpha() = -0.5*p.tau[2]*p.mu[2];
        y0[0] = dg::evaluate( gaussian, grid);
        dg::blas2::symv( gamma, y0[0], y0[2]); 
        dg::DVec v2d=dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[2], y0[2]);
        if( p.a[2] == 0)
        {
            std::cerr << "No impurity blob with trace impurities possible!\n";
            return -1;
        }
        dg::blas1::axpby( 1./p.a[2], y0[2], 0., y0[2]); //n_z ~1./a_z n_e
        y0[1] = dg::evaluate( dg::zero, grid);
    }

    //////////////////initialisation of timestepper and first step///////////////////
    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    karniadakis.init( toeflI, diffusion, y0, p.dt);


    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    const double mass_blob0 = toeflI.mass();
    double E0 = toeflI.energy(), energy0 = E0, E1 = 0, diff = 0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> laplacianM( grid, dg::normed, dg::centered);
    while ( !glfwWindowShouldClose( w ))
    {
        //transform field to an equidistant grid
        title << std::setprecision(2) << std::scientific;
        for( unsigned i=0; i<y0.size(); i++)
        {
            dg::blas1::transfer( y0[i], hvisual);
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
        dg::blas2::gemv( laplacianM, toeflI.potential()[0], y0[1]);
        dg::blas1::transfer( y0[1], hvisual);
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
            std::cout << "(m_tot-m_0)/m_0: "<< (toeflI.mass()-mass_blob0)/mass_blob0<<"\t";
            E0 = E1;
            E1 = toeflI.energy();
            diff = (E1 - E0)/p.dt;
            double diss = toeflI.energy_diffusion( );
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout << diff << " "<<diss<<"\t";
            std::cout << "Accuracy: "<< 2.*fabs((diff-diss)/(diff+diss))<<"\n";

            try{ karniadakis( toeflI, diffusion, y0);}
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
