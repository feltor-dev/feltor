#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG

#include "draw/host_window.h"
//#include "draw/device_window.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/backend/sparseblockmat.cuh"
#include "dg/backend/timer.cuh"
#include "dg/backend/average.cuh"
#include "dg/backend/typedefs.cuh"

#include "feltor.cuh"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the Explicit - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/

int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Reader reader;
    Json::Value js, gs;
    if( argc == 1)
    {
        std::ifstream is("input.json");
        std::ifstream ks("geometry_params.json");
        reader.parse(is,js,false);
        reader.parse(ks,gs,false);
    }
    else if( argc == 3)
    {
        std::ifstream is(argv[1]);
        std::ifstream ks(argv[2]);
        reader.parse(is,js,false);
        reader.parse(ks,gs,false);
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] \n";
        return -1;
    }
    const feltor::Parameters p( js);
    const dg::geo::solovev::GeomParameters gp(gs);
    p.display( std::cout);
    gp.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    std::stringstream title;
    std::ifstream is( "window_params.js");
    reader.parse( is, js, false);
    is.close();
    unsigned red = js.get("reduction", 1).asUInt();
    GLFWwindow* w = draw::glfwInitAndCreateWindow( (p.Nz/red+1)*js["width"].asDouble(), js["rows"].asDouble()*js["height"].asDouble(), "");
    draw::RenderHostData render(js["rows"].asDouble(), p.Nz/red + 1);
    /////////////////////////////////////////////////////////////////////////
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grid
    dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, p.bc, p.bc, dg::PER);  

    //create RHS 
    std::cout << "Constructing Explicit...\n";
    feltor::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> feltor( grid, mag, p, gp); //initialize before rolkar!
    std::cout << "Constructing Implicit...\n";
    feltor::Implicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> rolkar( grid, p, gp, feltor.ds(), feltor.dsDIR());
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //background profile
    dg::geo::Nprofile<Psip> prof(p.bgprofamp, p.nprofileamp, gp, Psip(gp)); //initial background profile
    std::vector<dg::DVec> y0(4, dg::evaluate( prof, grid)), y1(y0); 
    //perturbation 
    dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1); //modulation along fieldline
    if( p.mode == 0 || p.mode == 1)
    {
        dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
        if( p.mode == 0)
            y1[1] = feltor.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 3); //rounds =3 ->2*3-1
        if( p.mode == 1)
            y1[1] = feltor.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1); //rounds =1 ->2*1-1
    }
    if( p.mode == 2)
    {
        dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
        y1[1] = feltor.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1); 
    }
    if( p.mode == 3)
    {
        dg::geo::ZonalFlow<Psip> init0(p.amp, p.k_psi, gp, Psip(gp));
        y1[1] = feltor.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1); 
    }
    dg::blas1::axpby( 1., y1[1], 1., y0[1]); //sum up background and perturbation
    dg::blas1::plus(y0[1], -1); //initialize ni-1
    if( p.mode == 2 || p.mode == 3)
    {
        dg::DVec damping = dg::evaluate( dg::geo::GaussianProfXDamping<Psip>(Psip(gp), gp), grid);
        dg::blas1::pointwiseDot(damping,y0[1], y0[1]); //damp with gaussprofdamp
    }
    std::cout << "intiialize ne" << std::endl;
    if( p.initcond == 0) feltor.initializene( y0[1], y0[0]);
    if( p.initcond == 1) dg::blas1::axpby( 1., y0[1], 0.,y0[0], y0[0]); //set n_e = N_i
    std::cout << "Done!\n";

    dg::blas1::axpby( 0., y0[2], 0., y0[2]); //set Ue = 0
    dg::blas1::axpby( 0., y0[3], 0., y0[3]); //set Ui = 0

    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    std::cout << "intiialize karniadakis" << std::endl;
    karniadakis.init( feltor, rolkar, y0, p.dt);
    std::cout << "Done!\n";
//     feltor.energies( y0);//now energies and potential are at time 0

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual),avisual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);
    dg::ToroidalAverage<dg::HVec> toravg(grid);
    //create timer
    dg::Timer t;
    double time = 0;
    unsigned step = 0;
    
    const double mass0 = feltor.mass();
    double E0 = feltor.energy(), energy0 = E0, E1 = 0., dEdt = 0.;
    
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
     //probe
    const dg::HVec Xprobe(1,gp.R_0+p.boxscaleRp*gp.a);
    const dg::HVec Zprobe(1,0.);
    const dg::HVec Phiprobe(1,M_PI);
    dg::IDMatrix probeinterp;
    dg::blas2::transfer( dg::create::interpolation( Xprobe, Zprobe, Phiprobe, grid, dg::NEU), probeinterp);
    dg::DVec probevalue(1,0.);   
    while ( !glfwWindowShouldClose( w ))
    {

        dg::blas1::transfer( y0[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();        
        //colors.scalemin() = 1.0;
        //colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );

        title << std::setprecision(2) << std::scientific;
        //title <<"ne / "<<(double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() )<<"  " << colors.scalemax()<<"\t";
        title <<"ne-1 / " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/red;k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*red*size, visual.begin()+(k*red+1)*size);   
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw ions
        //thrust::transform( y1[1].begin(), y1[1].end(), dvisual.begin(), dg::PLUS<double>(-0.));//ne-1
        dg::blas1::transfer( y0[1], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        //colors.scalemin() = 1.0;        
        colors.scalemin() = -colors.scalemax();        
        //colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );

        title << std::setprecision(2) << std::scientific;
        //title <<"ni / "<<(double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() )<<"  " << colors.scalemax()<<"\t";
        title <<"ni-1 / " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/red;k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*red*size, visual.begin()+(k*red+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        //draw potential
        //transform to Vor
        dvisual=feltor.potential()[0];
        dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
        dg::blas1::transfer( y1[1], hvisual);
//         hvisual = feltor.potential()[0];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(),visual.end(), 0.,thrust::maximum<double>()  );
//         colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        colors.scalemin() = -colors.scalemax();
        //title <<"Phi / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        title <<"Omega / "<< colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/red;k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*red*size, visual.begin()+(k*red+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        //draw U_e
        dg::blas1::transfer( y0[2], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), 0.,thrust::maximum<double>()  );
        //colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        colors.scalemin() = -colors.scalemax();
        //title <<"Ue / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        title <<"Ue / " << colors.scalemax()<<"\t";
                for( unsigned k=0; k<p.Nz/red;k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*red*size, visual.begin()+(k*red+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);      
        
        //draw U_i
        dg::blas1::transfer( y0[3], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>()  );
        //colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        colors.scalemin() = -colors.scalemax();
        //title <<"Ui / "<<colors.scalemin()<< "  " << colors.scalemax()<<"\t";
        title <<"Ui / " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/red;k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*red*size, visual.begin()+(k*red+1)*size);
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
//             feltor.energies( y0); //update energetics
            //Compute probe values
            dg::blas2::gemv(probeinterp,y0[0],probevalue);
            std::cout << " Ne_p - 1  = " << probevalue[0] <<"\t";
            dg::blas2::gemv(probeinterp,feltor.potential()[0],probevalue);
            std::cout << " Phi_p = " << probevalue[0] <<"\t";
            std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass0<<"\t";
            E1 = feltor.energy();
            dEdt = (E1 - E0)/p.dt; //
            double diss = feltor.energy_diffusion( );
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout << "Accuracy: "<< 2.*fabs((dEdt-diss)/(dEdt+diss))<<" d E/dt = " << dEdt <<" Lambda =" << diss << "\n";
            
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
