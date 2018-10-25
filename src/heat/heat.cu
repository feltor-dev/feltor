#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG
#include "json/json.h"

#include "draw/host_window.h"
//#include "draw/device_window.cuh"
#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"

#include "parameters.h"
#include "heat.cuh"

int main( int argc, char* argv[])
{
    //-------Parameter initialisation------------------
    std::stringstream title;
    Json::Value js, gs;
    if( argc == 1)
    {
        std::ifstream is("input.json");
        std::ifstream ks("geometry_params.js");
        is >> js;
        ks >> gs;
    }
    else if( argc == 3)
    {
        std::ifstream is(argv[1]);
        std::ifstream ks(argv[2]);
        is >> js;
        ks >> gs;
    }
    else
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] \n";
        return -1;
    }
    const heat::Parameters p( js); p.display( std::cout);
    const dg::geo::solovev::Parameters gp(gs); gp.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////
    std::ifstream is( "window_params.js");
    is >> js;
    is.close();
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["width"].asDouble(), js["height"].asDouble(), "");
    draw::RenderHostData render(js["rows"].asDouble(), js["cols"].asDouble());
    //////////////////////////////////////////////////////////////////////

    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;

     dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcx, p.bcy, dg::PER);

    //create RHS
    std::cout << "Initialize explicit" << std::endl;
    heat::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ex( grid, p,gp); //initialize before im!
    std::cout << "Initialize implicit" << std::endl;
    heat::Implicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec > im( grid, p,gp);

    //-----------------------------The initial field--------------------
    //initial perturbation
    dg::Gaussian3d init0(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma_z, p.amp);

    // background profile
    dg::geo::Nprofile prof(0., p.nprofileamp, gp, dg::geo::solovev::Psip(gp)); //initial background profile
    dg::DVec y0( dg::evaluate( prof, grid)), y1(y0);

    // field aligning
    std::cout << "T aligning" << std::endl;
    dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
    y1 = ex.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 3); //rounds =2 ->2*2-1 //3 rounds for blob

    dg::blas1::axpby( 1., y1, 1., y0); //initialize y0

    std::cout << "Done!\n";

    //////////////////////////////////////////////////////////////////
    //Adaptive solver
    dg::Adaptive<dg::ARKStep<dg::DVec>> adaptive(
        "ARK-4-2-3", y0, grid.size(), p.eps_time);
    double dt_new = p.dt, dt = dt_new;
//     dg::Karniadakis< dg::DVec > karniadakis( y0, y0.size(),1e-13);
//     karniadakis.init( ex, im, y0, p.dt);

    ex.energies( y0);//now energies and potential are at time 0

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual),avisual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);

    //create timer
    dg::Timer t;
    double time = 0;
    unsigned step = 0;

    const double heat0 = ex.energy();
    double E0 = ex.entropy(), energy0 = E0, E1 = 0, diff = 0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);

    dg::DVec T0 = y0, T1(T0);
    dg::DVec w3d =  dg::create::volume(grid);
    double normT0 = dg::blas2::dot(  w3d, T0);
    while ( !glfwWindowShouldClose( w ))
    {
        dg::assign( y0, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        dg::blas1::scal(avisual, 0.);
        for( unsigned k=0; k<p.Nz;k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::View<dg::HVec> part( visual.data() + k*size, size);
            dg::blas1::axpby( 1.0, part, 1.0, avisual);
        }
        dg::blas1::scal(avisual,1./p.Nz);
        colors.scalemax() = (double)thrust::reduce( avisual.begin(),
                        avisual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();
        title << std::setprecision(2) << std::scientific;
        title <<"Temperature / " << colors.scalemax()<<"\t";
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

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
            ex.energies( y0); //update energetics
            std::cout << "(Q_tot-Q_0)/Q_0: "
                      << (ex.energy()-heat0)/heat0<<"\t";
            E1 = ex.entropy();
            diff = (E1 - E0)/dt; //
            double diss = ex.entropy_diffusion( );
            dg::blas1::axpby( 1., y0, -1.,T0, T1);
            double err = sqrt(dg::blas2::dot( w3d, T1)/normT0);
            std::cout << "(E_tot-E_0)/E_0: "
                      << (E1-energy0)/energy0<<"\t";
            std::cout << "Accuracy: "
                      << 2.*(diff-diss)/(diff+diss)
                      <<" d E/dt = " << diff
                      <<" Lambda =" << diss
                      << " err =" << err << "\n";
            E0 = E1;
            try{
                dt = dt_new;
                // rk( ex, y0, y1, p.dt);
                adaptive.step(ex,im,time,y0,time,y0,dt_new,
                    dg::pid_control, dg::l2norm, p.rtol, 1e-10);
                // karniadakis( ex, im, y0);

              }
              catch( dg::Fail& fail) {
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;}


        }
        time += (double)time;
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
