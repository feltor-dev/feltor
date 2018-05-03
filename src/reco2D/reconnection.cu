#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "reconnection.cuh"


/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/
// double aparallel( double x, double y)
// {
//     return 0.1/cosh(x)/cosh(x)*fabs(sin(1./2.*y));
// }

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
    const asela::Parameters p( js);
    p.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    std::stringstream title;
    std::ifstream is( "window_params.js");
    reader.parse( is, js, false);
    is.close();
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["width"].asDouble(), js["height"].asDouble(), "");
    draw::RenderHostData render(js["rows"].asDouble(), js["cols"].asDouble());
    //////////////////////////////////////////////////////////////////////////
    //Make grid

    dg::Grid2d grid( -p.lxhalf, p.lxhalf, -p.lyhalf, p.lyhalf , p.n, p.Nx, p.Ny, dg::DIR, dg::PER);
    std::cout << "Constructing Explicit...\n";
    asela::Asela<dg::CartesianGrid2d, dg::IDMatrix, dg::DMatrix, dg::DVec> asela( grid, p); //initialize before rolkar!
    std::cout << "Constructing Implicit...\n";
    asela::Implicit<dg::CartesianGrid2d, dg::IDMatrix, dg::DMatrix, dg::DVec> rolkar(  grid, p);
    std::cout << "Done!\n";

   /////////////////////The initial field///////////////////////////////////////////
    std::cout << "intiialize fields" << std::endl;
    std::vector<dg::DVec> y0(4, dg::evaluate( dg::one, grid)), y1(y0); 
    
    //Harris sheet problem
    if( p.init == 0) { 
        //true harris is -lambda ln (cosh(x/lambda))
        dg::InvCoshXsq init0( 1., 2.*M_PI/p.lxhalf);
        dg::CosY perty(   1., 0., p.mY*M_PI/p.lyhalf);
        dg::CosXCosY damp(1., 0., M_PI/p.lxhalf/2.,0.);    
        y0[3] = dg::evaluate( init0, grid);
        y1[2] = dg::evaluate( perty, grid);
        y1[3] = dg::evaluate( damp, grid);
    }
    //Island coalescence problem
    if( p.init == 1) { 
        dg::IslandXY init0( p.lxhalf/(2.*M_PI), 0.2);
        dg::CosY perty(   1., 0., p.mY*M_PI/p.lyhalf);
        dg::CosXCosY damp(1., 0., M_PI/p.lxhalf/2.,0.);    
        y0[3] = dg::evaluate( init0, grid);
        y1[2] = dg::evaluate( perty, grid);
        y1[3] = dg::evaluate( damp, grid);
    }
    
    //Compute initial A_par
    dg::blas1::axpby(-p.amp,y1[2],1.0,y0[3],y0[3]); // = [ A*Cos(y*ky) + 1/Cosh2(x*kx) ] (harris)
    dg::blas1::pointwiseDot(y1[3],y0[3],y0[3]);     // A_par = cos(x *kx') * [ A*Cos(y*ky) + 1/Cosh2(x*kx) ] (harris)

    //Compute u_e, U_i, w_e, W_i
    dg::blas2::gemv( rolkar.laplacianM(),y0[3], y0[2]);        //u_e = -nabla_perp A_par
    dg::blas1::scal(y0[2],-1.0);                               //u_e =  nabla_perp A_par
    dg::blas1::axpby(1., y0[2], p.beta/p.mu[0], y0[3], y0[2]); //w_e =  u_e + beta/mue A_par
    asela.initializene( y0[3], y1[3]);                         //A_G = Gamma_1 A_par
    dg::blas1::axpby(p.beta/p.mu[1], y1[3], 0.0, y0[3]);       //w_i =  beta/mui A_G
    //Compute n_e
    dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-1.)); // =Ni - bg 
    asela.initializene( y0[1], y0[0]);                         //n_e = Gamma_1 N_i

    std::cout << "Done!\n";


    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    std::cout << "intiialize karniadakis" << std::endl;
    karniadakis.init( asela, rolkar, y0, p.dt);
    std::cout << "Done!\n";
//     asela.energies( y0);//now energies and potential are at time 0

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual),avisual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);

    //create timer
    dg::Timer t;
    double time = 0;
    unsigned step = 0;
    const double mass0 = asela.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = asela.energy(), energy0 = E0, E1 = 0, diff = 0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);

    while ( !glfwWindowShouldClose( w ))
    {
        //plot electrons
        dg::blas1::transfer( y0[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title << std::setprecision(2) << std::scientific;
        title <<"ne-1 / " << colors.scalemax()<<"\t";

        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw ions
        dg::blas1::transfer( y0[1], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title << std::setprecision(2) << std::scientific;
        title <<"ni-1 / " << colors.scalemax()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw Potential
        dg::blas1::transfer( asela.potential()[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //transform to Vor
        //dvisual=asela.potential()[0];
        //dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
        //hvisual = y1[1];
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), 0.,thrust::maximum<double>()  );
//         colors.scalemin() =  (float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        colors.scalemin() = -colors.scalemax();
        title <<"Phi / " << colors.scalemax()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        //draw Vor
        dvisual=asela.potential()[0];
        dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
        dg::blas1::transfer( y1[1], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        dg::blas1::scal(visual,-1.0);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), 0.,thrust::maximum<double>()  );
//         colors.scalemin() =  (float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        colors.scalemin() = -colors.scalemax();
        title <<"Vor / " << colors.scalemax()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        //draw U_e
        dg::blas1::transfer( asela.uparallel()[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title <<"Ue / " << colors.scalemax()<<"\t";
         render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw U_i
        dg::blas1::transfer( asela.uparallel()[1], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title <<"Ui / "<< colors.scalemax()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw a parallel
        dg::blas1::transfer(asela.aparallel(), hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(),visual.end(), 0., thrust::maximum<double>()  );
        colors.scalemin() = - colors.scalemax();
        title <<"A / "<<(float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax(),thrust::minimum<double>() )<< "  " << colors.scalemax()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

         //draw j_par
        dvisual=asela.aparallel();
        dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
        dg::blas1::transfer( y1[1], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        
        colors.scalemax() = (float)thrust::reduce( visual.begin(),visual.end(), 0., thrust::maximum<double>()  );
        colors.scalemin() = - colors.scalemax();
        title <<"j / "<<(float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax(),thrust::minimum<double>() )<< "  " << colors.scalemax()<<"\t";
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
        //double x;
        //std::cin >> x;
        for( unsigned i=0; i<p.itstp; i++)
        {
            try{ karniadakis( asela, rolkar, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;
            }
            step++;
            std::cout << "(m_tot-m_0)/m_0: "<< (asela.mass()-mass0)/mass_blob0<<"\t";
            E1 = asela.energy();
            diff = (E1 - E0)/p.dt; //
            double diss = asela.energy_diffusion( );
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<" d E/dt = " << diff <<" Lambda =" << diss << "\n";
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
