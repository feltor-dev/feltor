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
#include "file/read_input.h"
#include "solovev/geometry.h"
#include "dg/runge_kutta.h"

#include "heat.cuh"
#include "parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the Feltor - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/


int main( int argc, char* argv[])
{
    //Parameter initialisation
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
//     GLFWwindow* w = draw::glfwInitAndCreateWindow( (p.Nz+1)/v2[2]*v2[3], v2[1]*v2[4], "");
//     draw::RenderHostData render(v2[1], (p.Nz+1)/v2[2]);
    //draw only average
    GLFWwindow* w = draw::glfwInitAndCreateWindow( (1)/v2[2]*v2[3], v2[1]*v2[4], "");
    draw::RenderHostData render(v2[1], (1)/v2[2]); 
    //////////////////////////////////////////////////////////////////////////
    
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grid
     dg::Grid3d<double > grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, p.bc, p.bc, dg::PER, dg::cylindrical);  
    //create RHS 
    
    std::cout << "initialize feltor" << std::endl;
    eule::Feltor<dg::DMatrix, dg::DVec, dg::DVec > feltor( grid, p,gp); //initialize before rolkar!
    std::cout << "initialize rolkar" << std::endl;
    eule::Rolkar<dg::DMatrix, dg::DVec, dg::DVec > rolkar( grid, p,gp);

    ////////////////////////////////The initial field////////////////////////////////
 //initial perturbation
    std::cout << "initialize delta T" << std::endl;
    dg::Gaussian3d init0(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma_z, p.amp);
//     dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
//     dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
//     solovev::ZonalFlow init0(p, gp);
//     dg::CONSTANT init0( 0.);

    
    //background profile
    std::cout << "T background" << std::endl;
    solovev::Nprofile prof(p, gp); //initial background profile
    std::vector<dg::DVec> y0(1, dg::evaluate( prof, grid)), y1(y0); 
    //field aligning
//     std::cout << "T aligning" << std::endl;  
//     dg::CONSTANT gaussianZ( 1.);
//     dg::GaussianZ gaussianZ( M_PI, p.sigma_z*M_PI, 1);
//     y1[0] = feltor.dz().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 3); //rounds =2 ->2*2-1

    //no field aligning
    std::cout << "No T aligning" << std::endl;  
    
    y1[0] = dg::evaluate( init0, grid);
    
    dg::blas1::axpby( 1., y1[0], 1., y0[0]); //initialize ni
//     dg::blas1::transform(y0[0], y0[0], dg::PLUS<>(-1)); //initialize ni-1

//     dg::blas1::pointwiseDot(rolkar.damping(),y0[0], y0[0]); //damp with gaussprofdamp
    std::cout << "Done!\n";

    //////////////////////////////////////////////////////////////////////////////////
    //RK solver
    dg::RK<4, std::vector<dg::DVec> >  rk( y0);
    feltor.energies( y0);//now energies and potential are at time 0

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual),avisual(hvisual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);

    //create timer
    dg::Timer t;
    double time = 0;
    unsigned step = 0;
    
    const double mass0 = feltor.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = feltor.energy(), energy0 = E0, E1 = 0, diff = 0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    dg::DVec T0 = dg::evaluate( dg::one, grid);  
    dg::DVec T1 = dg::evaluate( dg::one, grid);  

    dg::blas1::axpby( 1., y0[0], 0., T0); //initialize ni
    dg::DVec w3d =  dg::create::weights(grid);
    double normT0 = dg::blas2::dot(  w3d, T0);
    while ( !glfwWindowShouldClose( w ))
    {
        hvisual = y0[0];
        dg::blas1::transform(hvisual,hvisual , dg::PLUS<>(-1)); //npe = N+1
        dg::blas2::gemv( equi, hvisual, visual);
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        for( unsigned k=0; k<p.Nz;k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*size, visual.begin()+(k+1)*size);
            dg::blas1::axpby(1.0,part,1.0,avisual);
//             render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::scal(avisual,1./p.Nz);
        colors.scalemax() = (float)thrust::reduce( avisual.begin(), avisual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();        
//                 colors.scalemin() =  (float)thrust::reduce( avisual.begin(), avisual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
        title <<"T-1 / " << colors.scalemax()<<"\t";
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
            feltor.energies( y0); //update energetics
            std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass_blob0<<"\t";
            E1 = feltor.energy();
            diff = (E1 - E0)/p.dt; //
            double diss = feltor.energy_diffusion( );
            dg::blas1::axpby( 1., y0[0], -1.,T0, T1);
            double err = sqrt(dg::blas2::dot( w3d, T1)/normT0);
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<" d E/dt = " << diff <<" Lambda =" << diss << " err =" << err << "\n";
            E0 = E1;
            try{
                rk( feltor, y0, y1, p.dt);
                y0.swap( y1);}
              catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;}


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
