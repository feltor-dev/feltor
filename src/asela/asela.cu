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
#include "dg/backend/average.cuh"
#include "file/read_input.h"
#include "solovev/geometry.h"

#include "asela.cuh"
#include "asela/parameters.h"

#define TORLIM //for toroidal limiter setup

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/


int main( int argc, char* argv[])
{
     ////////////////////////Parameter initialisation//////////////////////////
    std::vector<double> v,v2,v3;
    std::stringstream title;
    if( argc == 1)
    {
        try{
            v = file::read_input("input.txt");
            v3 = file::read_input( "geometry_params.txt"); 
        }catch( toefl::Message& m){
            m.display();
            return -1;
        }
    }
    else if( argc == 3)
    {
        try{
            v = file::read_input(argv[1]);
            v3 = file::read_input( argv[2]); 
        }catch( toefl::Message& m){
            m.display();
            return -1;
        }
    }
    else
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] \n";
        return -1;
    }
    const eule::Parameters p( v);
    p.display( std::cout);
    const solovev::GeomParameters gp(v3);
    gp.display( std::cout);
    v2 = file::read_input( "window_params.txt");
    GLFWwindow* w = draw::glfwInitAndCreateWindow( (p.Nz+1)/v2[2]*v2[3], v2[1]*v2[4], "");
    draw::RenderHostData render(v2[1], (p.Nz+1)/v2[2]);



    //////////////////////////////////////////////////////////////////////////
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grid
     dg::Grid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, p.bc, p.bc, dg::PER, dg::cylindrical);  
    //create RHS 
    std::cout << "Constructing asela...\n";
    eule::Asela<dg::DDS, dg::DMatrix, dg::DVec, dg::DVec > asela( grid, p,gp); //initialize before rolkar!
    std::cout << "Constructing Rolkar...\n";
    eule::Rolkar<dg::DMatrix, dg::DVec, dg::DVec > rolkar( grid, p,gp);
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //initial perturbation
    //dg::Gaussian3d init0(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
    dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
//     dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
//     solovev::ZonalFlow init0(p, gp);

    
    //background profile
    solovev::Nprofile prof(p, gp); //initial background profile
    std::vector<dg::DVec> y0(4, dg::evaluate( prof, grid)), y1(y0); 
    
    //field aligning
    //dg::CONSTANT gaussianZ( 1.);
    dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
    y1[1] = asela.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 3); //rounds =2 ->2*2-1

    //no field aligning
//     y1[1] = dg::evaluate( init0, grid);
    
    dg::blas1::axpby( 1., y1[1], 1., y0[1]); //initialize ni
    dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-1)); //initialize ni-1
    dg::blas1::pointwiseDot(rolkar.damping(),y0[1], y0[1]); //damp with gaussprofdamp
    asela.initializene( y0[1], y0[0]);    
    dg::blas1::axpby( 0., y0[2], 0., y0[2]); //set we = 0
    dg::blas1::axpby( 0., y0[3], 0., y0[3]); //set wi = 0

    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    karniadakis.init( asela, rolkar, y0, p.dt);
    karniadakis( asela, rolkar, y0); //now energies and potential are at time 0

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual),avisual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);
    dg::ToroidalAverage<dg::HVec> toravg(grid);

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
        hvisual = karniadakis.last()[0];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title << std::setprecision(2) << std::scientific;
        title <<"ne-1 / " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw ions
        hvisual =  karniadakis.last()[1];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title << std::setprecision(2) << std::scientific;
        title <<"ni-1 / " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw Potential
        hvisual = asela.potential()[0];
        dg::blas2::gemv( equi, hvisual, visual);
        //transform to Vor
        //dvisual=asela.potential()[0];
        //dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
        //hvisual = y1[1];
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0.,thrust::maximum<double>()  );
//         colors.scalemin() =  (float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        colors.scalemin() = -colors.scalemax();
        title <<"Phi / " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        //draw U_e
        hvisual = asela.uparallel()[0]; //=U_parallel_e
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title <<"Ue / " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw U_i
        hvisual =asela.uparallel()[1];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title <<"Ui / "<< colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw a parallel
        hvisual = asela.aparallel();
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(),visual.end(), 0., thrust::maximum<double>()  );
        colors.scalemin() = - colors.scalemax();
        title <<"A / "<<(float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() )<< "  " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
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
            asela.energies( y0); //update energetics
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
