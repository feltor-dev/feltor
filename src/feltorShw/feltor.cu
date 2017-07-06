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
#include "../diag/probes.h"



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

    dg::ExpProfX prof(p.nprofileamp, p.bgprofamp,p.invkappa);
    std::vector<dg::DVec> y0(2, dg::evaluate( prof, grid)), y1(y0); 

    
    if (p.initmode == 0) { 
      dg::Gaussian init0( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);
      y1[1] = dg::evaluate( init0, grid);
    }
    if (p.initmode == 1) {
      dg::SinXSinY init0(p.amp,0.,2*M_PI/p.lx,p.sigma*2*M_PI/p.ly);
      y1[1] = dg::evaluate( init0, grid);
    }
    if (p.initmode == 2) {
      dg::BathRZ init0(16,16,1.,0.,0., 30.,5.,p.amp);
      y1[1] = dg::evaluate( init0, grid);
      dg::DVec  dampr = dg::evaluate(dg::TanhProfX(p.lx*0.95,p.sourcew,-1.0,0.0,1.0),grid);
      dg::DVec  dampl = dg::evaluate(dg::TanhProfX(p.lx*0.05,p.sourcew,1.0,0.0,1.0),grid);
      dg::blas1::pointwiseDot(y1[1],dampr,y1[1]);
      dg::blas1::pointwiseDot(y1[1],dampl,y1[1]);
   
    }  
        
    if (p.modelmode == 0 || p.modelmode == 1)
    {
        dg::blas1::pointwiseDot(y1[1], y0[1],y1[1]); //<n>*ntilde
        dg::blas1::axpby( 1., y1[1], 1., y0[1]); //initialize ni = <n> + <n>*ntilde
        dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); //initialize ni-1
        std::cout << "intiialize ne" << std::endl;
        feltor.initializene( y0[1], y0[0]);    
        std::cout << "Done!\n";
    }
    if (p.modelmode == 2) {
        std::cout << "intiialize ne" << std::endl;
        dg::blas1::axpby(1.0,y1[1],0.,y0[1],y0[1]);
        feltor.initializene( y1[1], y0[0]);    
        std::cout << "Done!\n";
    }
    if (p.modelmode == 3) {
        dg::blas1::pointwiseDot( y0[1],y1[1],y0[1]); //<n>*Ntilde
        dg::blas1::axpby( 1., y0[1], 1.,y1[0], y0[1]); //initialize Ni = <n> + <n>*Ntilde
        dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); //initialize Ni-1
        
        std::cout << "intiialize ne" << std::endl;
        feltor.initializene( y0[1], y0[0]); //n_e-1
        
        
        dg::blas1::transform( y1[1], y0[1], dg::PLUS<>(+1.0)); // (1+Nitilde)
        dg::blas1::transform( y0[1], y0[1], dg::LN<double>()); //ln (1+Nitilde)
        
        dg::blas1::transform(y0[0], y0[0], dg::PLUS<>((p.bgprofamp + p.nprofileamp))); //ne
        dg::blas1::pointwiseDivide(y0[0], y1[0],y0[0]); // 1+ netilde
        dg::blas1::transform( y0[0], y0[0], dg::LN<double>()); //ln (1+netilde)

        std::cout << "Done!\n";
    } 



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
    
    dg::DVec xprobecoords(7,1.);
    for (unsigned i=0;i<7; i++) {
        xprobecoords[i] = p.lx/8.*(1+i) ;
    }
    const dg::DVec yprobecoords(7,p.ly/2.);
    probes<dg::IDMatrix,dg::DMatrix, dg::DVec> pro(xprobecoords,yprobecoords,grid);
    while ( !glfwWindowShouldClose( w ))
    {

        dg::blas1::transfer(y0[0], hvisual);
//         if 
//         dg::blas1::axpby(1.0,hvisual,
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), (double)-1e14, thrust::maximum<double>() );
//         colors.scalemin() = -colors.scalemax();        
        //colors.scalemin() = 1.0;
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );

        title << std::setprecision(2) << std::scientific;
        //title <<"ne / "<<(double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() )<<"  " << colors.scalemax()<<"\t";
        title <<"ne-1 / " << colors.scalemin()<<"\t";

        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw ions
        //thrust::transform( y1[1].begin(), y1[1].end(), dvisual.begin(), dg::PLUS<double>(-0.));//ne-1
        dg::blas1::transfer(y0[1], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );
        //colors.scalemin() = 1.0;        
//         colors.scalemin() = -colors.scalemax();        
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );

        title << std::setprecision(2) << std::scientific;
        //title <<"ni / "<<(double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() )<<"  " << colors.scalemax()<<"\t";
        title <<"ni-1 / " << colors.scalemin()<<"\t";

        render.renderQuad(visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        
        //draw potential
        //transform to Vor
//        dvisual=feltor.potential()[0];
//        dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
//        hvisual = y1[1];
        dg::blas1::transfer(feltor.potential()[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );

        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );

//         //colors.scalemin() = 1.0;        
//          colors.scalemin() = -colors.scalemax();        
//          colors.scalemin() = -colors.scalemax();        
        //colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title <<"Potential / "<< colors.scalemax() << " " << colors.scalemin()<<"\t";

        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw potential
        //transform to Vor
        dvisual=feltor.potential()[0];
        dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
        dg::blas1::transfer(y1[1], hvisual);
         //hvisual = feltor.potential()[0];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );
        //colors.scalemin() = 1.0;        
//          colors.scalemin() = -colors.scalemax();        
        colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title <<"Omega / "<< colors.scalemax()<< " "<< colors.scalemin()<<"\t";

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
//             double coupling = feltor.coupling();
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout << 
                         " Charge= " << feltor.charge() <<
                         " Accuracy: "<< 2.*fabs((diff-diss)/(diff+diss))<<
                         " d E/dt = " << diff <<
                         " Lambda =" << diss <<  std::endl;
            E0 = E1;
        }
        dg::blas1::transform( y0[0], dvisual, dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //npe = N+1
        dvisual2 = feltor.potential()[0];
        pro.fluxes(time,  dvisual,dvisual2);
        pro.profiles(time,dvisual,dvisual2);
//         p.profiles
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
