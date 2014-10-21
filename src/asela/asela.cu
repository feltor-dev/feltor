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

#include "asela.cuh"
#include "feltor/parameters.h"

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
    double Rmin=gp.R_0-p.boxscale*gp.a;
    double Zmin=-p.boxscale*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscale*gp.a; 
    double Zmax=p.boxscale*gp.a*gp.elongation;
    //Make grid
     dg::Grid3d<double > grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, dg::DIR, dg::DIR, dg::PER, dg::cylindrical);  
    //create RHS 
    std::cout << "Constructing Feltor...\n";
    eule::Feltor<dg::DMatrix, dg::DVec, dg::DVec > feltor( grid, p,gp); //initialize before rolkar!
    std::cout << "Constructing Rolkar...\n";
    eule::Rolkar<dg::DMatrix, dg::DVec, dg::DVec > rolkar( grid, p,gp);
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //initial perturbation
    //dg::Gaussian3d init0(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
//     dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
    dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
//     solovev::ZonalFlow init0(p, gp);

    
    //background profile
    solovev::Nprofile prof(p, gp); //initial background profile
    std::vector<dg::DVec> y0(4, dg::evaluate( prof, grid)), y1(y0); 
    
    //field aligning
    //dg::CONSTANT gaussianZ( 1.);
    dg::GaussianZ gaussianZ( M_PI, p.sigma_z*M_PI, 1);
    y1[1] = feltor.dz().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 3); //rounds =2 ->2*2-1
    y1[2] = dg::evaluate( gaussianZ, grid);
    dg::blas1::pointwiseDot( y1[1], y1[2], y1[1]);
    //no field aligning
//     y1[1] = dg::evaluate( init0, grid);
    
    dg::blas1::axpby( 1., y1[1], 1., y0[1]); //initialize ni
    dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-1)); //initialize ni-1
    dg::blas1::pointwiseDot(rolkar.damping(),y0[1], y0[1]); //damp with gaussprofdamp
    feltor.initializene( y0[1], y0[0]);    
    dg::blas1::axpby( 0., y0[2], 0., y0[2]); //set Ue = 0
    dg::blas1::axpby( 0., y0[3], 0., y0[3]); //set Ui = 0

    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    karniadakis.init( feltor, rolkar, y0, p.dt);

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
    
   

    while ( !glfwWindowShouldClose( w ))
    {
        //plot electrons
        hvisual = karniadakis.last()[0];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title << std::setprecision(2) << std::scientific;
        title <<"ne-1 / " << colors.scalemax()<<"\t";
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            dg::blas1::axpby(1.0,part,1.0,avisual);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::scal(avisual,1./p.Nz);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw ions
        hvisual =  karniadakis.last()[1];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title << std::setprecision(2) << std::scientific;
        title <<"ni-1 / " << colors.scalemax()<<"\t";
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            dg::blas1::axpby(1.0,part,1.0,avisual);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::scal(avisual,1./p.Nz);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw Potential
        hvisual = feltor.potential()[0];
        dg::blas2::gemv( equi, hvisual, visual);
        //transform to Vor
        //dvisual=feltor.potential()[0];
        //dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
        //hvisual = y1[1];
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0.,thrust::maximum<double>()  );
//         colors.scalemin() =  (float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        colors.scalemin() = -colors.scalemax();
        title <<"Phi / " << colors.scalemax()<<"\t";
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            dg::blas1::axpby(1.0,part,1.0,avisual);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::scal(avisual,1./p.Nz);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw U_e
        hvisual = feltor.uparallel()[0]; //=U_parallel_e
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title <<"Ue / " << colors.scalemax()<<"\t";
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            dg::blas1::axpby(1.0,part,1.0,avisual);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::scal(avisual,1./p.Nz);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw U_i
        hvisual =feltor.uparallel()[1];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title <<"Ui / "<< colors.scalemax()<<"\t";
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            dg::blas1::axpby(1.0,part,1.0,avisual);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::scal(avisual,1./p.Nz);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw a parallel
        hvisual = feltor.aparallel();
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(),visual.end(), 0., thrust::maximum<double>()  );
        colors.scalemin() = - colors.scalemax();
        title <<"A / "<<(float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() )<< "  " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            dg::blas1::axpby(1.0,part,1.0,avisual);

            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::scal(avisual,1./p.Nz);
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
            step++;
            std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass_blob0<<"\t";
            E1 = feltor.energy();
            diff = (E1 - E0)/p.dt; //
            double diss = feltor.energy_diffusion( );
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<" d E/dt = " << diff <<" Lambda =" << diss << "\n";
            E0 = E1;

            try{ karniadakis( feltor, rolkar, y0);}
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
