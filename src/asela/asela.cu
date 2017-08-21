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
#include "geometries/geometries.h"

#include "asela.cuh"
#include "parameters.h"


/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/
typedef dg::FieldAligned< dg::CylindricalGrid3d, dg::IDMatrix, dg::DVec> DFA;

using namespace dg::geo::solovev;

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
    const eule::Parameters p( js);
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
    draw::RenderHostData render(js["rows"].asDouble(), p.Nz/js["reduction"].asUInt() + 1);
    //////////////////////////////////////////////////////////////////////////
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grid

    dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, p.bc, p.bc, dg::PER);      //create RHS 
    std::cout << "Constructing Asela...\n";
    eule::Asela<dg::CylindricalGrid3d, dg::DS<DFA, dg::DMatrix, dg::DVec>, dg::DMatrix, dg::DVec> asela( grid, p,gp); //initialize before rolkar!
    std::cout << "Constructing Rolkar...\n";
    eule::Rolkar<dg::CylindricalGrid3d, dg::DS<DFA, dg::DMatrix, dg::DVec>, dg::DMatrix, dg::DVec> rolkar(  grid, p, gp, asela.ds(), asela.dsDIR());
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
            y1[1] = asela.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 3); //rounds =3 ->2*3-1
        if( p.mode == 1)
            y1[1] = asela.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1); //rounds =1 ->2*1-1
    }
    if( p.mode == 2)
    {
        dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
        y1[1] = asela.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1); 
    }
    if( p.mode == 3)
    {
        dg::geo::ZonalFlow<Psip> init0(p.amp, p.k_psi, gp, Psip(gp));
        y1[1] = asela.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1); 
    }
    dg::blas1::axpby( 1., y1[1], 1., y0[1]); //sum up background and perturbation
    dg::blas1::plus(y0[1], -1); //initialize ni-1
    if( p.mode == 2 || p.mode == 3)
    {
        dg::DVec damping = dg::evaluate( dg::geo::GaussianProfXDamping<Psip>(Psip(gp), gp), grid);
        dg::blas1::pointwiseDot(damping,y0[1], y0[1]); //damp with gaussprofdamp
    }
    std::cout << "intiialize ne" << std::endl;
    if( p.initcond == 0) asela.initializene( y0[1], y0[0]);
    if( p.initcond == 1) dg::blas1::axpby( 1., y0[1], 0.,y0[0], y0[0]); //set n_e = N_i
    std::cout << "Done!\n";

    dg::blas1::axpby( 0., y0[2], 0., y0[2]); //set Ue = 0
    dg::blas1::axpby( 0., y0[3], 0., y0[3]); //set Ui = 0

    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    std::cout << "intiialize karniadakis" << std::endl;
    karniadakis.init( asela, rolkar, y0, p.dt);
    std::cout << "Done!\n";
//     asela.energies( y0);//now energies and potential are at time 0

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
        dg::blas1::transfer( y0[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title << std::setprecision(2) << std::scientific;
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
        dg::blas1::transfer( y0[1], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title << std::setprecision(2) << std::scientific;
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
        dg::blas1::transfer( asela.uparallel()[0], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
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
        dg::blas1::transfer( asela.uparallel()[1], hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();   
        title <<"Ui / "<< colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/red;k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*red*size, visual.begin()+(k*red+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        dg::blas1::axpby(0.0,avisual,0.0,avisual);
        toravg(visual,avisual);
        render.renderQuad( avisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //draw a parallel
        dg::blas1::transfer(asela.aparallel(), hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(),visual.end(), 0., thrust::maximum<double>()  );
        colors.scalemin() = - colors.scalemax();
        title <<"A / "<<(float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() )<< "  " << colors.scalemax()<<"\t";
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
