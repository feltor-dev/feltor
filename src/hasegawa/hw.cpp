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
    std::string input;
    if( argc == 1)
        input = "input.json";
    else if( argc == 2)
        input = argv[1];
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const toefl::Parameters p(  dg::file::file2Json(input));
    p.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    std::stringstream title;
    dg::file::WrappedJsonValue js = dg::file::file2Json( "window_params.json");
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["cols"].asUInt()*js["width"].asUInt()*p.lx/p.ly, js["rows"].asUInt()*js["height"].asUInt(), "");
    draw::RenderHostData render(js["rows"].asUInt(), js["cols"].asUInt());
    /////////////////////////////////////////////////////////////////////////
    dg::CartesianGrid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bcx, p.bcy);
    //create RHS
    bool mhw = (p.model == "modified");
    hw::HW<dg::CartesianGrid2d, dg::IDMatrix, dg::DMatrix, dg::DVec > rhs( grid, p.kappa,
            p.tau, p.nu, p.eps_pol[0], mhw);
    dg::DVec one( grid.size(), 1.);
    //create initial vector
    dg::Gaussian gaussian( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.amp); //gaussian width is in absolute values
    dg::Vortex vortex( p.posX*grid.lx(), p.posY*grid.ly(), 0, p.sigma, p.amp);
    std::vector<dg::DVec> y0(2, dg::evaluate( vortex, grid)), y1(y0); // n_e' = gaussian
    dg::DVec w2d( dg::create::weights( grid));

    if( p.bcx == dg::PER && p.bcy == dg::PER)
    {
        double meanMass = dg::blas2::dot( y0[0], w2d, one)/(double)(p.lx*p.ly);
        std::cout << "Mean Mass is "<<meanMass<<"\n";
        dg::blas1::axpby( -meanMass, one, 1., y0[0]);
        y0[1] = y0[0];
    }

    std::string tableau;
    double rtol, atol, time = 0.;
    try{
        rtol = js["timestepper"].get("rtol", 1e-5).asDouble();
        atol = js["timestepper"].get("atol", 1e-5).asDouble();
        tableau = js[ "timestepper"].get( "tableau",
                "Bogacki-Shampine-4-2-3").asString();
    }catch ( std::exception& error){
        DG_RANK0 std::cerr << "Error in input file " << argv[1]<< std::endl;
        DG_RANK0 std::cerr << error.what() << std::endl;
        dg::abort_program();
    }
    DG_RANK0 std::cout<< "Construct timeloop ...\n";
    dg::Adaptive< dg::ERKStep< std::vector<dg::DVec>>> adapt(tableau, y0);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double E0 = rhs.energy(), E1 = 0, diff = 0; //energy0 = E0;
    double Ezf0 = rhs.zonal_flow_energy(), Ezf1 = 0, diffzf = 0; //energyzf0 = Ezf0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> laplacianM(grid,  dg::centered);
    unsigned itstp = js["output"]["itstp"].asUInt();
    double dt = 1e-5;
    while ( !glfwWindowShouldClose( w ))
    {
        if( p.bcx == dg::PER && p.bcy == dg::PER)
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

        dg::blas2::gemv( laplacianM, rhs.potential(), y1[1]);
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
        for( unsigned i=0; i<itstp; i++)
        {
            step++;
            {
                if( p.bcx == dg::PER && p.bcy == dg::PER)
                {
                    double meanMass = dg::blas2::dot( y0[0], w2d, one)/(double)(p.lx*p.ly);
                    dg::blas1::axpby( -meanMass, one, 1., y0[0]);
                    meanMass = dg::blas2::dot( y0[1], w2d, one)/(double)(p.lx*p.ly);
                    dg::blas1::axpby( -meanMass, one, 1., y0[1]);
                }
                E0 = E1; Ezf0 = Ezf1;
                E1 = rhs.energy(); Ezf1 = rhs.zonal_flow_energy();
                diff = (E1 - E0)/dt; diffzf = (Ezf1-Ezf0)/dt;
                double diss = rhs.energy_diffusion( ) + rhs.flux() + rhs.capital_jot(); double disszf = rhs.zonal_flow_diffusion() + rhs.capital_r();
                //std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                std::cout << "Accuracy:   "<< 2.*(diff-diss)/(diff+diss)<<"\n";
                std::cout << "AccuracyZF: "<< 2.*(diffzf-disszf)/(diffzf+disszf)<<"\n";
                std::cout << diff << " "<< rhs.energy_diffusion() << " " <<rhs.flux()<<" "<<rhs.capital_jot()<<"\n";
                std::cout << diffzf << " "<< rhs.zonal_flow_diffusion() << " " <<rhs.capital_r()<<"\n";

            }
            try{
                adapt.step( rhs, time, y0, time, y0, dt, dg::pid_control,
                        dg::l2norm, rtol, atol);
            }
            catch( std::exception& fail) {
                std::cerr << "ERROR in Timestepper\n";
                std::cerr << fail.what() << std::endl;
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;
            }
        }
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "\n\t Step "<<step;
        std::cout << "\n\t Average time for one step: "<<t.diff()/(double)itstp<<"s\n\n";
#endif//DG_BENCHMARK
    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////

    return 0;

}
