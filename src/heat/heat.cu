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
#include "dg/runge_kutta.h"
#include "dg/multistep.h"
#include "dg/elliptic.h"
#include "dg/cg.h"

#include "geometries/geometries.h"
#include "heat/parameters.h"


#include "heat.cuh"

typedef dg::FieldAligned< dg::CylindricalGrid3d, dg::IDMatrix, dg::DVec> DFA;
using namespace dg::geo::solovev;

int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    std::stringstream title;
    Json::Reader reader;
    Json::Value js, gs;
    if( argc == 1)
    {
        std::ifstream is("input.json");
        std::ifstream ks("geometry_params.js");
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
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] \n";
        return -1;
    }
    const eule::Parameters p( js); p.display( std::cout);
    const GeomParameters gp(gs); gp.display( std::cout);
    /////////glfw initialisation ////////////////////////////////////////////
    std::ifstream is( "window_params.js");
    reader.parse( is, js, false);
    is.close();
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["width"].asDouble(), js["height"].asDouble(), "");
    draw::RenderHostData render(js["rows"].asDouble(), js["cols"].asDouble());
    /////////////////////////////////////////////////////////////////////////
    
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;

     dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, p.bc, p.bc, dg::PER);  

//     dg::DVec w3d_ = dg::create::volume( grid);
//     dg::DVec v3d_ = dg::create::inv_volume( grid);
//     dg::DVec x = dg::evaluate( dg::zero, grid);
//     set up the parallel diffusion

    
//     dg::GeneralEllipticSym<dg::DMatrix, dg::DVec, dg::DVec> ellipticsym( grid, dg::not_normed, dg::forward);
//     dg::DVec bfield = dg::evaluate( solovev::bR( gp.R_0, gp.I_0),grid);
// 
//     ellipticsym.set_x( bfield);
//     bfield = dg::evaluate( solovev::bZ( gp.R_0, gp.I_0),grid);
//     ellipticsym.set_y( bfield);
//     bfield = dg::evaluate( solovev::bPhi( gp.R_0, gp.I_0),grid);
//     ellipticsym.set_z( bfield);
// //     
//     
//     double eps =1e-12;   
//     dg::Invert< dg::DVec> invert( x, w3d_.size(), eps );  
//     std::cout << "MAX # iterations = " << w3d_.size() << std::endl;
//     const dg::DVec rhs = dg::evaluate( solovev::DeriNeuT2( gp.R_0, gp.I_0), grid);
//     std::cout << " # of iterations "<< invert( ellipticsym, x, rhs ) << std::endl; //is dsTds
//     dg::DVec solution = dg::evaluate( solovev::FuncNeu(gp.R_0, gp.I_0),grid);
//     double normf = dg::blas2::dot( w3d_, solution);
//     std::cout << "Norm analytic Solution  "<<sqrt( normf)<<"\n";
//     double errinvT =dg::blas2::dot( w3d_, x);
//     std::cout << "Norm numerical Solution "<<sqrt( errinvT)<<"\n";
//     dg::blas1::axpby( 1., solution, +1.,x);
//     errinvT =dg::blas2::dot( w3d_, x);
//     std::cout << "Relative Difference is  "<< sqrt( errinvT/normf )<<"\n";

    
        
//     std::cout << "MAX # iterations = " << w3d_.size() << std::endl;
//     
//     std::cout << " # of iterations "<< invert( ellipticsym, x, rhs ) << std::endl; //is dsTds
//     
//     std::cout << "Norm analytic Solution  "<<sqrt( normf)<<"\n";
//     errinvT =dg::blas2::dot( w3d_, x);
//     std::cout << "Norm numerical Solution "<<sqrt( errinvT)<<"\n";
//     dg::blas1::axpby( 1., solution, +1.,x);
//     errinvT =dg::blas2::dot( w3d_, x);
//     std::cout << "Relative Difference is  "<< sqrt( errinvT/normf )<<"\n";
// 
   

    
    //create RHS     
    std::cout << "initialize feltor" << std::endl;
    eule::Feltor<dg::DS<DFA, dg::DMatrix, dg::DVec>, dg::DMatrix, dg::DVec > feltor( grid, p,gp); //initialize before rolkar!
    std::cout << "initialize rolkar" << std::endl;
    eule::Rolkar<dg::CylindricalGrid3d , dg::DS<DFA, dg::DMatrix, dg::DVec>, dg::DMatrix, dg::DVec > rolkar( grid, p,gp);

    ////////////////////////////////The initial field////////////////////////////////
 //initial perturbation
//     std::cout << "initialize delta T" << std::endl;
    dg::Gaussian3d init0(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma_z, p.amp);
//     dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
//     dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
//     solovev::ZonalFlow init0(p, gp);

//     dg::CONSTANT init0( 0.);

    
    //background profile
    std::cout << "T background" << std::endl;
    dg::geo::Nprofile<Psip> prof(p.bgprofamp, p.nprofileamp, gp, Psip(gp)); //initial background profile
    std::vector<dg::DVec> y0(1, dg::evaluate( prof, grid)), y1(y0); 
    
//     //field aligning
    std::cout << "T aligning" << std::endl;  
//     dg::CONSTANT gaussianZ( 1.);
    dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
    y1[0] = feltor.ds().fieldaligned().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 3); //rounds =2 ->2*2-1 //3 rounds for blob

    //no field aligning
//     std::cout << "No T aligning" << std::endl;      
//     y1[0] = dg::evaluate( init0, grid);
//        dg::blas1::pointwiseDot(rolkar.damping(),y1[0], y1[0]); //damp with gaussprofdamp
 
    dg::blas1::axpby( 1., y1[0], 1., y0[0]); //initialize ni
    if (p.bc ==dg::DIR)    {
    dg::blas1::transform(y0[0], y0[0], dg::PLUS<>(-1)); //initialize ni-1
    }

//     dg::blas1::pointwiseDot(rolkar.damping(),y0[0], y0[0]); //damp with gaussprofdamp
    std::cout << "Done!\n";

    //////////////////////////////////////////////////////////////////////////////////
    //RK solver
//     dg::RK<4, std::vector<dg::DVec> >  rk( y0);
    //SIRK solver
    dg::SIRK<std::vector<dg::DVec> > sirk(y0, grid.size(),p.eps_time);
//     dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(),1e-13);
//     karniadakis.init( feltor, rolkar, y0, p.dt);

     feltor.energies( y0);//now energies and potential are at time 0
   
    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual),avisual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
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
    dg::DVec w3d =  dg::create::volume(grid);
    double normT0 = dg::blas2::dot(  w3d, T0);
    while ( !glfwWindowShouldClose( w ))
    {
        dg::blas1::transfer( y0[0], hvisual);
        if (p.bc ==dg::NEU)    {
        dg::blas1::transform(hvisual,hvisual , dg::PLUS<>(-1)); //npe = N+1
        }
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
        colors.scalemax() = (double)thrust::reduce( avisual.begin(), avisual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() = -colors.scalemax();        
//                 colors.scalemin() =  (double)thrust::reduce( avisual.begin(), avisual.end(), colors.scalemax()  ,thrust::minimum<double>() );
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
//                 rk( feltor, y0, y1, p.dt);
                 sirk(feltor,rolkar,y0,y1,p.dt);
//                 karniadakis( feltor, rolkar, y0);

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
