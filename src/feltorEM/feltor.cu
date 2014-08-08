#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG

#include "draw/host_window.h"
//#include "draw/device_window.cuh"
#include "dg/backend/timer.cuh"
#include "dg/backend/xspacelib.cuh"
#include "file/read_input.h"

#include "feltor.cuh"
#include "bessel.h"
#include "parameters.h"
#include "geometry.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/


int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::vector<double> v, v2,v3;
    std::stringstream title;
    if( argc == 1)
    {
        try{
        v = file::read_input("input.txt");
        }catch( toefl::Message& m){m.display();}
    }
    else if( argc == 2)
    {
        v = file::read_input( argv[1]);
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }

    /////////////////////////////////////////////////////////////////////////
    const Parameters p( v);
    p.display( std::cout);
    v2 = file::read_input( "window_params.txt");
    GLFWwindow* w = draw::glfwInitAndCreateWindow( p.Nz/v2[2]*v2[3], v2[1]*v2[4], "");
    draw::RenderHostData render(v2[1], p.Nz/v2[2]);

    //////////////////////////////////////////////////////////////////////////
//     dg::Grid3d<double > grid( p.R_0-p.a*(1.05), p.R_0 + p.a*(1.05),  -p.a*(1.05), p.a*(1.05), 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, dg::DIR, dg::DIR, dg::PER);
    
    try{ v3 = file::read_input( "geometry_params.txt"); }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
//             std::cout << v3[i] << " ";
//             std::cout << std::endl;
        return -1;}

    const solovev::GeomParameters gp(v3);
    gp.display( std::cout);
    double Rmin=gp.R_0-(gp.boxscale)*gp.a;
    double Zmin=-(gp.boxscale)*gp.a*gp.elongation;
    double Rmax=gp.R_0+(gp.boxscale)*gp.a; 
    double Zmax=(gp.boxscale)*gp.a*gp.elongation;
    //Make grid
     dg::Grid3d<double > grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, dg::DIR, dg::DIR, dg::PER);  
     
    //create RHS 
    eule::Feltor< dg::DVec > feltor( grid, p,gp); //initialize before rolkar!
    eule::Rolkar< dg::DVec > rolkar( grid, p,gp);


      dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,15.,p.amp);
//       solovev::ZonalFlow init0(gp,p.amp);
    
//     solovev::Gradient grad(gp); //background gradient
    solovev::Nprofile grad(gp); //initial profile

    std::vector<dg::DVec> y0(4, dg::evaluate( grad, grid)), y1(y0); 
    //damp the bath on psi boundaries 
    dg::blas1::pointwiseDot(rolkar.dampin(),(dg::DVec)dg::evaluate(init0, grid), y1[0]); //is damping on bath

    
    dg::blas1::axpby( 1., y1[0], 1., y0[0]);
    dg::blas1::axpby( 1., y1[0], 1., y0[1]);
    dg::blas1::axpby( 0., y0[2], 0., y0[2]); //set Ue = 0
    dg::blas1::axpby( 0., y0[3], 0., y0[3]); //set Ui = 0
//     dg::blas1::pointwiseDot(rolkar.dampout(),y0[1],y0[1]); //is damping on bath
    feltor.log( y0, y0, 2); //transform to logarithmic values (ne and ni)
    
    dg::Karniadakis< std::vector<dg::DVec> > ab( y0, y0[0].size(), p.eps_time);
    ab.init( feltor, rolkar, y0, p.dt);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
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
        //transform field to an equidistant grid
        feltor.exp( y0, y1, 2); //calculate real densities from logdensities

        //plot electrons
        thrust::transform( y1[0].begin(), y1[0].end(), dvisual.begin(), dg::PLUS<double>(0.));//ne-1
        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() =  (float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
        title <<"ne / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }

        //draw ions
        thrust::transform( y1[1].begin(), y1[1].end(), dvisual.begin(), dg::PLUS<double>(0.));//ne-1
        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>() );
        colors.scalemin() =  (float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
        title <<"ni / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }

        //transform phi
        dg::blas2::gemv( rolkar.laplacianM(), feltor.potential()[0], y1[1]);
        hvisual = y1[1];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0.,thrust::maximum<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title <<"phi / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }

        //draw U_e
        hvisual = feltor.uparallel()[0]; //=U_parallel_e
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0.,thrust::maximum<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title <<"Ue / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }

        //draw U_i
        hvisual =feltor.uparallel()[1];
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title <<"Ui / "<<colors.scalemin()<< "  " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }

        //draw a parallel
        hvisual =feltor.aparallel();
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scalemax() = (float)thrust::reduce( visual.begin(), visual.end(), 0., thrust::maximum<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title <<"A / "<<colors.scalemin()<< "  " << colors.scalemax()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        
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
            E0 = E1;
            E1 = feltor.energy();
            diff = (E1 - E0)/p.dt;
            double diss = feltor.energy_diffusion( );
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";

            try{ ab( feltor, rolkar, y0);}
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
