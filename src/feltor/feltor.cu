#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "feltor.cuh"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the Explicit - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/

int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js, gs;
    if( argc == 1)
    {
        std::ifstream is("input.json");
        std::ifstream ks("geometry_params.json");
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
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] \n";
        return -1;
    }
    const feltor::Parameters p( js);
    const dg::geo::solovev::Parameters gp(gs);
    p.display( std::cout);
    gp.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    std::stringstream title;
    std::ifstream is( "window_params.js");
    is >> js;
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
    feltor::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> explicitPart( grid, p, gp); //initialize before imp!
    std::cout << "Constructing Implicit...\n";
    feltor::Implicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> implicitPart( grid, p, gp, explicitPart.ds(), explicitPart.dsDIR());
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    std::array<std::array<dg::DVec,2>,2> y0;
    dg::DVec helper(dg::evaluate(dg::zero,grid));
    //perturbation
    dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1); //modulation along fieldline
    if( p.initni == "blob" || p.initni == "straight blob")
    {
        dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
        if( p.initni == "blob")
            helper = explicitPart.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 3); //rounds =3 ->2*3-1
        if( p.initni == "straight blob")
            helper = explicitPart.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1); //rounds =1 ->2*1-1
    }
    else if( p.initni == "turbulence")
    {
        dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
        helper = explicitPart.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1);
    }
    else if( p.initni == "zonal")
    {
        dg::geo::ZonalFlow init0(p.amp, p.k_psi, gp, dg::geo::solovev::Psip(gp));
        helper = explicitPart.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1);
    }
    else
        std::cerr <<"WARNING: Unknown initial condition for Ni!\n";
    dg::geo::Nprofile prof(p.bgprofamp, p.nprofileamp, gp, dg::geo::solovev::Psip(gp)); //initial background profile
    y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::evaluate( prof, grid);
    dg::blas1::axpby( 1., helper, 1., y0[0][1]); //sum up background and perturbation
    dg::blas1::plus(y0[0][1], -1); //initialize ni-1
    if( p.initni == "turbulence" || p.initni == "zonal") //Cut initialization outside separatrix
    {
        dg::DVec damping = dg::evaluate( dg::geo::GaussianProfXDamping(dg::geo::solovev::Psip(gp), gp), grid);
        dg::blas1::pointwiseDot(damping,y0[0][1], y0[0][1]);
    }
    std::cout << "intiialize ne" << std::endl;
    if( p.initphi == "zero")  explicitPart.initializene( y0[0][1], y0[0][0]);
    else if( p.initphi == "balance") dg::blas1::copy( y0[0][1], y0[0][0]); //set n_e = N_i
    else
        std::cerr <<"WARNING: Unknown initial condition for phi!\n";
    std::cout << "Done!\n";

    dg::blas1::copy( 0., y0[1][0]); //set Ue = 0
    dg::blas1::copy( 0., y0[1][0]); //set Ui = 0

    dg::Karniadakis< std::array<std::array<dg::DVec,2>,2> > karniadakis( y0, y0[0][0].size(), p.eps_time);
    std::cout << "intiialize karniadakis" << std::endl;
    karniadakis.init( explicitPart, implicitPart, 0., y0, p.dt);
    std::cout << "Done!\n";

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual),avisual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);
    //create timer
    dg::Timer t;
    double time = 0;
    unsigned step = 0;

    const double mass0 = explicitPart.mass();
    double E0 = explicitPart.energy(), energy0 = E0, E1 = 0., dEdt = 0.;

    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
     //probe
    const dg::HVec Xprobe(1,gp.R_0+p.boxscaleRp*gp.a);
    const dg::HVec Zprobe(1,0.);
    const dg::HVec Phiprobe(1,M_PI);
    dg::IDMatrix probeinterp;
    dg::blas2::transfer( dg::create::interpolation( Xprobe, Zprobe, Phiprobe, grid, dg::NEU), probeinterp);
    dg::DVec probevalue(1,0.);
    dg::Average<dg::HVec> toroidal_average( grid, dg::coo3d::z);
    while ( !glfwWindowShouldClose( w ))
    {

        dg::blas1::transfer( y0[0][0], hvisual);
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
        toroidal_average(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw ions
        //thrust::transform( helper.begin(), helper.end(), dvisual.begin(), dg::PLUS<double>(-0.));//ne-1
        dg::blas1::transfer( y0[0][1], hvisual);
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
        toroidal_average(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw potential
        //transform to Vor
        dvisual=explicitPart.potential()[0];
        dg::blas2::gemv( implicitPart.laplacianM(), dvisual, helper);
        dg::blas1::transfer( helper, hvisual);
//         hvisual = explicitPart.potential()[0];
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
        toroidal_average(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw U_e
        dg::blas1::transfer( y0[1][0], hvisual);
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
        toroidal_average(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //draw U_i
        dg::blas1::transfer( y0[1][1], hvisual);
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
        toroidal_average(visual,avisual);
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
            try{ karniadakis.step( explicitPart, implicitPart, time, y0);}
            catch( dg::Fail& fail) {
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;
            }
            step++;
//             explicitPart.energies( y0); //update energetics
            //Compute probe values
            dg::blas2::gemv(probeinterp,y0[0][0],probevalue);
            std::cout << " Ne_p - 1  = " << probevalue[0] <<"\t";
            dg::blas2::gemv(probeinterp,explicitPart.potential()[0],probevalue);
            std::cout << " Phi_p = " << probevalue[0] <<"\t";
            std::cout << "(m_tot-m_0)/m_0: "<< (explicitPart.mass()-mass0)/mass0<<"\t";
            E1 = explicitPart.energy();
            dEdt = (E1 - E0)/p.dt; //
            double diss = explicitPart.energy_diffusion( );
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout << "Accuracy: "<< 2.*fabs((dEdt-diss)/(dEdt+diss))<<" d E/dt = " << dEdt <<" Lambda =" << diss << "\n";

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
