#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>

#include "draw/host_window.h"

#include "feltor.cuh"
#include "implicit.h"

using HVec = dg::HVec;
using DVec = dg::DVec;
using DMatrix = dg::DMatrix;
using IDMatrix = dg::IDMatrix;
using IHMatrix = dg::IHMatrix;
using Geometry = dg::CylindricalGrid3d;
#define MPI_OUT

#include "init.h"
#include "feltordiag.h"

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
        std::cerr << "ERROR: Too many arguments!\nUsage: "
                  << argv[0]<<" [inputfile] [geomfile] \n";
        return -1;
    }
    const feltor::Parameters p( js);
    const dg::geo::solovev::Parameters gp(gs);
    p.display( std::cout);
    gp.display( std::cout);
    /////////////////////////////////////////////////////////////////////////
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grid
    dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.symmetric ? 1 : p.Nz, p.bcxN, p.bcyN, dg::PER);
    dg::DVec vol3d = dg::create::volume( grid);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    mag = dg::geo::createModifiedSolovevField(gp, (1.-p.rho_damping)*mag.psip()(mag.R0(),0.), p.alpha_mag);

    //create RHS
    std::cout << "Constructing Explicit...\n";
    feltor::Explicit<Geometry, IDMatrix, DMatrix, DVec> feltor( grid, p, mag);
    std::cout << "Constructing Implicit...\n";
    feltor::Implicit<Geometry, IDMatrix, DMatrix, DVec> im( grid, p, mag);
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    std::array<std::array<DVec,2>,2> y0;
    feltor::Initialize init( grid, p, mag);
    if( argc == 4)
        y0 = init.init_from_parameters(feltor);
    if( argc == 5)
        y0 = init.init_from_file(argv[4]);
    feltor.set_source( init.profile(), p.omega_source, init.source_damping());
    std::cout << "Initialize Timestepper" << std::endl;

    ////////////////////////create timer and timestepper
    //
    dg::Timer t;
    double time = 0.;
    unsigned step = 0;
    dg::Karniadakis< std::array<std::array<dg::DVec,2>,2 >,
        feltor::FeltorSpecialSolver<
            Geometry, IDMatrix, DMatrix, DVec>
        > karniadakis( grid, p, mag);
    karniadakis.init( feltor, im, time, y0, p.dt);
    std::cout << "Done!" << std::endl;

    std::map<std::string, const dg::DVec* > v4d;
    v4d["ne-1 / "] = &y0[0][0],               v4d["ni-1 / "] = &y0[0][1];
    v4d["Ue / "]   = &feltor.fields()[1][0],  v4d["Ui / "]   = &feltor.fields()[1][1];
    v4d["Ome / "] = &feltor.potential()[0]; v4d["Apar / "] = &feltor.induction();
    double dEdt = 0, accuracy = 0;
    double E0 = 0.;
    /////////////////////////set up transfer for glfw
    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual), avisual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);

    /////////glfw initialisation ////////////////////////////////////////////
    //
    std::stringstream title;
    std::ifstream is( "window_params.js");
    is >> js;
    is.close();
    unsigned red = js.get("reduction", 1).asUInt();
    double rows = js["rows"].asDouble(), cols = p.Nz/red+1,
           width = js["width"].asDouble(), height = js["height"].asDouble();
    if ( p.symmetric ) cols = rows, rows = 1;
    GLFWwindow* w = draw::glfwInitAndCreateWindow( cols*width, rows*height, "");
    draw::RenderHostData render(rows, cols);

    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    dg::Average<dg::HVec> toroidal_average( grid, dg::coo3d::z);
    title << std::setprecision(2) << std::scientific;
    //unsigned failed_counter = 0;
    std::vector<std::string> energies = { "nelnne", "nilnni", "aperp2", "ue2","neue2","niui2"};
    std::vector<std::string> energy_diff = { "resistivity", "leeperp", "leiperp", "leeparallel", "leiparallel"};
    while ( !glfwWindowShouldClose( w ))
    {
        title << std::fixed;
        title << "t = "<<time<<"   ";
        for( auto pair : v4d)
        {
            if(pair.first == "Ome / ")
            {
                //dg::blas2::gemv( laplacianM, *pair.second, dvisual);
                dg::assign( feltor.lapMperpP(0), hvisual);
                dg::assign( *pair.second, hvisual);
            }
            else if(pair.first == "ne-1 / " || pair.first == "ni-1 / ")
            {
                dg::assign( *pair.second, hvisual);
                dg::blas1::axpby( 1., hvisual, -1., init.profile(), hvisual);
            }
            else
                dg::assign( *pair.second, hvisual);
            dg::blas2::gemv( equi, hvisual, visual);
            colors.scalemax() = (double)thrust::reduce(
                visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
            colors.scalemin() = -colors.scalemax();
            title <<pair.first << colors.scalemax()<<"   ";
            if ( p.symmetric )
                render.renderQuad( hvisual, grid.n()*grid.Nx(),
                                            grid.n()*grid.Ny(), colors);
            else
            {
                for( unsigned k=0; k<p.Nz/red;k++)
                {
                    unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
                    dg::HVec part( visual.begin() +  k*red   *size,
                                   visual.begin() + (k*red+1)*size);
                    render.renderQuad( part, grid.n()*grid.Nx(),
                                             grid.n()*grid.Ny(), colors);
                }
                dg::blas1::scal(avisual,0.);
                toroidal_average(visual,avisual);
                render.renderQuad( avisual, grid.n()*grid.Nx(),
                                            grid.n()*grid.Ny(), colors);
            }
        }
        glfwSetWindowTitle(w,title.str().c_str());
        title.str("");
        glfwPollEvents();
        glfwSwapBuffers( w);

        //step
        t.tic();
        for( unsigned i=0; i<p.itstp; i++)
        {
            double previous_time = time;
            for( unsigned k=0; k<p.inner_loop; k++)
            {
                try{
                    karniadakis.step( feltor, im, time, y0);
                }
                catch( dg::Fail& fail) {
                    std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                    std::cerr << "Does Simulation respect CFL condition?\n";
                    glfwSetWindowShouldClose( w, GL_TRUE);
                    break;
                }
                step++;
            }
            double deltat = time - previous_time;
            double energy = 0, ediff = 0.;
            for( auto record& : diagnostics2d_list)
            {
                if( std::find( energies.begin(), energies.end(), record.name) != energies.end())
                {
                    record.function( result, var);
                    energy += dg::blas1::dot( result, feltor.vol3d());
                }
                if( std::find( energy_diff.begin(), energy_diff.end(), record.name) != energy_diff.end())
                {
                    record.function( result, var);
                    ediff += dg::blas1::dot( result, feltor.vol3d());
                }

            }
            dEdt = (energy - E0)/deltat;
            E0 = energy;
            accuracy  = 2.*fabs( (dEdt - ediff)/( dEdt + ediff));

            std::cout << "Time "<<time<<"\n";
            std::cout <<" d E/dt = " << dEdt
              <<" Lambda = " << ediff
              <<" -> Accuracy: " << accuracy << "\n";
            //----------------Test if induction equation holds
            if( p.beta != 0)
            {
                dg::blas1::pointwiseDot(
                    feltor.density(0), feltor.velocity(0), dvisual);
                dg::blas1::pointwiseDot( p.beta,
                    feltor.density(1), feltor.velocity(1), -p.beta, dvisual);
                double norm  = dg::blas2::dot( dvisual, vol3d, dvisual);
                dg::blas1::axpby( -1., feltor.lapMperpA(), 1., dvisual);
                double error = dg::blas2::dot( dvisual, vol3d, dvisual);
                std::cout << " Rel. Error Induction "<<sqrt(error/norm) <<"\n";
            }

        }
        t.toc();
        std::cout << "\n\t Step "<<step << " at time  "<<time;
        std::cout << "\n\t Average time for one step: "<<t.diff()/(double)p.itstp/(double)p.inner_loop<<"\n\n";
        //std::cout << "\n\t Total # of failed steps:   "<<failed_counter<<"\n\n";
    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////
    return 0;

}
