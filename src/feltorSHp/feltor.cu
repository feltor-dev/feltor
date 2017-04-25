#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG

#include "draw/host_window.h"
//#include "draw/device_window.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/backend/timer.cuh"

#include "feltor.cuh"
#include "parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the Feltor - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/


int main( int argc, char* argv[])
{
    ////////////////////////Parameter initialisation//////////////////////////
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
    const eule::Parameters p(  js);
    p.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    std::stringstream title;
    std::ifstream is( "window_params.js");
    reader.parse( is, js, false);
    is.close();
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["cols"].asUInt()*js["width"].asUInt()*p.lx/p.ly, js["rows"].asUInt()*js["height"].asUInt(), "");
    draw::RenderHostData render(js["rows"].asUInt(), js["cols"].asUInt());
    //////////////////////////////////////////////////////////////////////////

    //Make grid
    dg::Grid2d grid( 0., p.lx, 0.,p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);  
    //create RHS 
    std::cout << "Constructing Feltor...\n";
    eule::Feltor<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > feltor( grid, p); //initialize before rolkar!
    std::cout << "Constructing Rolkar...\n";
    eule::Rolkar<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > rolkar( grid, p);
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //initial perturbation
    dg::Gaussian init0( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);

    dg::CONSTANT prof(p.bgprofamp );
    std::vector<dg::DVec> y0(4, dg::evaluate( prof, grid)), y1(y0); //Ne,Ni,Te,Ti = prof    
   
   //initialization via N_i,T_I ->n_e, t_i=t_e
    y1[1] = dg::evaluate( init0, grid);
    dg::blas1::pointwiseDot(y1[1], y0[1],y1[1]); //N_i = <Ni>*Ni_tilde = prof*Gauss
    //for Ni and ne with (scaled) blob structure
    dg::blas1::axpby( 1., y1[1], 1., y0[1]); // Ni = <Ni> (1 + *Ni_tilde)
    if (p.iso == 1) dg::blas1::axpby( 1.,y1[2], 0., y1[3]); //Ti = <Ni> 
    if (p.iso == 0) dg::blas1::axpby( 1.,y1[1], 1., y1[3]); //Ti = <Ni> (1 + *Ni_tilde)
    dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); //Ni_tilde= Ni - bg
    
    std::cout << "intiialize ne" << std::endl;
    if( p.init == 0) feltor.initializene( y0[1],y1[3], y0[0]);    //ne_tilde = Gamma_1 (Ni - bg) for OmegaE=0
    if( p.init == 1) dg::blas1::axpby( 1., y0[1], 0., y0[0], y0[0]); // ne_tilde = Ni_tilde for Omega*=0
    std::cout << "Done!\n";    
    
    std::cout << "intialize ti=te" << std::endl;
    if (p.iso == 1) {
        dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //Ni=Ni_tilde +bg
        dg::blas1::pointwiseDot(y0[1],y1[3],y0[3]); // Pi = Ni Ti
        dg::blas1::transform(y0[3], y0[3], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp))); //Pi_tilde = Pi - bg^2   
        dg::blas1::axpby( 1.,y0[3], 0., y0[2]); //Pi_tilde = pe_tilde = prof
    }
    if (p.iso == 0) {
        dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //Ni=Nitilde+bg
        dg::blas1::pointwiseDot(y0[1],y1[3],y0[3]); // Pi = Ni Ti
        dg::blas1::transform(y0[3], y0[3], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp))); //Pi_tilde = Pi - bg^2
        feltor.initializepi(y0[3],y1[3], y0[2]); // = pi-bg^2    
    }
    dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); // =Ni - bg 
    std::cout << "Done!\n";

    
    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    std::cout << "intiialize karniadakis" << std::endl;
    karniadakis.init( feltor, rolkar, y0, p.dt);
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
        dg::blas1::transfer( y0[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), (double)-1e14, thrust::maximum<double>() );
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
        title <<"ne-1 / " << colors.scalemax() << " " << colors.scalemin()<<"\t";
//          colors.scalemin() =  -colors.scalemax();
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw Ni-1
        dg::blas1::transfer( y0[1], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
        title <<"ni-1 / " << colors.scalemax() << " " << colors.scalemin()<<"\t";
//          colors.scalemin() =  -colors.scalemax();
        render.renderQuad(visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        
        //draw potential
        dg::blas1::transfer( feltor.potential()[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"Pot / "<< colors.scalemax() << " " << colors.scalemin()<<"\t";
        colors.scalemin() =  -colors.scalemax();
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        //draw Te-1
        dg::blas1::transfer( feltor.temptilde()[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), (double)-1e14, thrust::maximum<double>() );
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
        title <<"Te-1 / " << colors.scalemax() << " " << colors.scalemin()<<"\t";
//          colors.scalemin() =  -colors.scalemax();
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw Ti-1
        dg::blas1::transfer( feltor.temptilde()[1], hvisual);
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
        dg::blas1::transfer( y1[1], hvisual);
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
            try{ karniadakis( feltor, rolkar, y0);}
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
