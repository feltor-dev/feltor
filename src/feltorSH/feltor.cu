#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG

#include "draw/host_window.h"
#include "dg/file/json_utilities.h"

#include "feltor.cuh"
#include "parameters.h"


int main( int argc, char* argv[])
{
    ////////////////////////Parameter initialisation//////////////////////////
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
    const eule::Parameters p(  js);
    p.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    dg::file::file2Json( "window_params.json", js, dg::file::comments::are_discarded);
    std::stringstream title;
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["cols"].asUInt()*js["width"].asUInt()*p.lx/p.ly, js["rows"].asUInt()*js["height"].asUInt(), "");
    draw::RenderHostData render(js["rows"].asUInt(), js["cols"].asUInt());
    //////////////////////////////////////////////////////////////////////////

    //Make grid
    dg::Grid2d grid( 0., p.lx, 0.,p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);  
    //create RHS 
    std::cout << "Constructing Explicit...\n";
    eule::Explicit<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > feltor( grid, p); //initialize before rolkar!
    std::cout << "Constructing Implicit...\n";
    eule::Implicit<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > rolkar( grid, p);
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //initial perturbation
    dg::Gaussian init0( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);

    dg::CONSTANT prof(p.bgprofamp );
    std::vector<dg::DVec> y0(4, dg::evaluate( prof, grid)), y1(y0); //Ne,Ni,Te,Ti = prof    
   
   //initialization via N_i,T_I ->n_e, t_i=t_e
    y1[1] = dg::evaluate( init0, grid);
    dg::blas1::pointwiseDot(y1[1], y0[1],y1[1]); //<n>*ntilde    
    dg::blas1::axpby( 1., y1[1], 1., y0[1]); //initialize Ni = <n> + <n>*ntilde
    if (p.iso == 1) dg::blas1::axpby( 1.,y1[2], 0., y0[3]); //initialize Ti = prof
    if (p.iso == 0) dg::blas1::axpby( 1.,y0[1], 0., y0[3]); //initialize Ti = N_i
    dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); //= Ni - bg
    std::cout << "intiialize ne" << std::endl;
    if( p.init == 0)
        feltor.initializene( y0[1],y0[3], y0[0]);    //ne -bg
    else  
        dg::blas1::axpby( 1., y0[1], 0., y0[0], y0[0]); // for Omega*=0
    std::cout << "Done!\n";    
    
    std::cout << "intialize ti=te" << std::endl;
    if (p.iso == 1) {
        dg::blas1::transform(y0[3], y0[3], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); // =Ti - bg
        dg::blas1::axpby( 1.,y0[3], 0., y0[2]); //initialize Ti = N_i
    }
    if (p.iso == 0) {
        dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //Ni
        dg::blas1::pointwiseDot(y0[1],y0[3],y1[3]); // = Ni Ti
        dg::blas1::transform(y1[3], y1[3], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp))); //Pi = Pi - bg^2

        feltor.initializepi(y1[3],y0[3], y0[2]); // = pi-bg^2    
//         dg::blas1::axpby( 1.,y1[3], 0., y0[2]); //initialize pi = P_i

        //compute ti-bg = ((pi-bg^2) +bg^2)/ne -bg
        dg::blas1::transform(y0[2], y0[2], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)));
        dg::blas1::transform(y0[0], y0[0], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //=ne    
        dg::blas1::pointwiseDivide(y0[2],y0[0],y0[2]);

        dg::blas1::transform(y0[2], y0[2], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp)));
        dg::blas1::transform(y0[0], y0[0], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); // =ne-bg
        dg::blas1::transform(y0[3], y0[3], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); // =Ti - bg
        dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); // =Ni - bg 
    }
    std::cout << "Done!\n";

    
    dg::DefaultSolver< std::vector<dg::DVec> > solver( rolkar, y0,
            y0[0].size(), p.eps_time);
    dg::ImExMultistep< std::vector<dg::DVec> > karniadakis( "ImEx-BDF-3-3", y0);
    std::cout << "initialize karniadakis" << std::endl;
    karniadakis.init( std::tie(feltor, rolkar, solver), 0., y0, p.dt);
    std::cout << "Done!\n";

    dg::DVec dvisual( grid.size(), 0.);
    dg::DVec dvisual2( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual),avisual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);
    //create timer
    dg::Timer t;
    double time = 0;
    unsigned step = 0;
    
    const double mass0 = feltor.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = feltor.energy(), energy0 = E0, E1 = 0., diff = 0.;
    
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);

    while ( !glfwWindowShouldClose( w ))
    {
        //draw Ne-1
        dg::assign( y0[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), (double)-1e14, thrust::maximum<double>() );
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
        title <<"ne-1 / " << colors.scalemax() << " " << colors.scalemin()<<"\t";
//          colors.scalemin() =  -colors.scalemax();
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw Ni-1
        dg::assign( y0[1], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
        title <<"ni-1 / " << colors.scalemax() << " " << colors.scalemin()<<"\t";
//          colors.scalemin() =  -colors.scalemax();
        render.renderQuad(visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        
        //draw potential
        dg::assign( feltor.potential()[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"Pot / "<< colors.scalemax() << " " << colors.scalemin()<<"\t";
        colors.scalemin() =  -colors.scalemax();
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        //draw Te-1
        dg::assign( y0[2], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), (double)-1e14, thrust::maximum<double>() );
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
        title <<"Te-1 / " << colors.scalemax() << " " << colors.scalemin()<<"\t";
//          colors.scalemin() =  -colors.scalemax();
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw Ti-1
        dg::assign( y0[3], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
        title <<"Ti-1 / " << colors.scalemax() << " " << colors.scalemin()<<"\t";
//          colors.scalemin() =  -colors.scalemax();
        render.renderQuad(visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        //draw vor
        //transform to Vor
        dvisual=feltor.potential()[0];
        dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
        dg::assign( y1[1], hvisual);
         //hvisual = feltor.potential()[0];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title <<"Omega / "<< colors.scalemax()<< " "<< colors.scalemin()<<"\t";
        colors.scalemin() =  -colors.scalemax();
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
            try{ karniadakis.step( std::tie(feltor, rolkar, solver), time, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;
            }
            step++;
            std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass_blob0<<"\t";
            E1 = feltor.energy();
            diff = (E1 - E0)/p.dt; //
            double diss = feltor.energy_diffusion( );
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout << " Accuracy: "<< 2.*fabs((diff-diss)/(diff+diss))<<
                         " d E/dt = " << diff <<
                         " Lambda =" << diss <<  std::endl;
 
                                     std::cout << E1 << std::endl;

            E0 = E1;

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
