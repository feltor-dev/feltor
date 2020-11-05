#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>

#include "draw/host_window.h"
#include "dg/file/json_utilities.h"

#include "feltor.h"
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
    std::string inputfile, geomfile;
    if( argc == 1)
        inputfile = "input.json", geomfile= "geometry_params.json";
    else if( argc == 3)
        inputfile = argv[1], geomfile= argv[2];
    else
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "
                  << argv[0]<<" [inputfile] [geomfile] \n";
        return -1;
    }
    try{
        file::file2Json( inputfile, js, file::comments::are_forbidden, file::error::is_throw);
        feltor::Parameters(js, file::error::is_throw);
    }catch(std::runtime_error& e)
    {

        std::cerr << "ERROR in input file "<<inputfile<<std::endl;
        std::cerr <<e.what()<<std::endl;
        return -1;
    }
    try{
        file::file2Json( geomfile, gs, file::comments::are_discarded, file::error::is_throw);
    }catch(std::runtime_error& e)
    {

        std::cerr << "ERROR in geometry file "<<geomfile<<std::endl;
        std::cerr <<e.what()<<std::endl;
        return -1;
    }

    const feltor::Parameters p(js);
    p.display( std::cout);
    std::cout << gs.toStyledString() << std::endl;
    dg::geo::TokamakMagneticField mag, mod_mag;
    dg::geo::CylindricalFunctor wall, transition, sheath, direction;
    try{
        mag = dg::geo::createMagneticField(gs, file::error::is_throw);
        mod_mag = dg::geo::createModifiedField(gs, js, file::error::is_throw, wall, transition);
    }catch(std::runtime_error& e)
    {
        std::cerr << "ERROR in geometry file "<<geomfile<<std::endl;
        std::cerr <<e.what()<<std::endl;
        return -1;
    }
    /////////////////////////////////////////////////////////////////////////
    //Make grid
    double Rmin=mag.R0()-p.boxscaleRm*mag.params().a();
    double Zmin=-p.boxscaleZm*mag.params().a()*mag.params().elongation();
    double Rmax=mag.R0()+p.boxscaleRp*mag.params().a();
    double Zmax=p.boxscaleZp*mag.params().a()*mag.params().elongation();
    dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.symmetric ? 1 : p.Nz, p.bcxN, p.bcyN, dg::PER);
    try{
        dg::geo::createSheathRegion( js, file::error::is_throw, mag, wall,
                Rmin, Rmax, Zmin, Zmax, sheath, direction);
    }catch(std::runtime_error& e)
    {
        std::cerr << "ERROR in geometry file "<<geomfile<<std::endl;
        std::cerr <<e.what()<<std::endl;
        return -1;
    }

    if( p.periodify)
        mag = dg::geo::periodify( mag, Rmin, Rmax, Zmin, Zmax, dg::NEU, dg::NEU);

    //create RHS
    //std::cout << "Constructing RHS...\n";
    feltor::Explicit<Geometry, IDMatrix, DMatrix, DVec> feltor( grid, p, mag);
    std::cout << "Constructing Explicit...\n";
    //feltor::Explicit<Geometry, IDMatrix, DMatrix, DVec> feltor( grid, p, mag, false);
    std::cout << "Constructing Implicit...\n";
    feltor::Implicit<Geometry, IDMatrix, DMatrix, DVec> implicit( grid, p, mag);
    std::cout << "Done!\n";

    DVec result = dg::evaluate( dg::zero, grid);
    /// Construct feltor::Variables object for diagnostics
    std::array<DVec, 3> gradPsip;
    gradPsip[0] =  dg::evaluate( mag.psipR(), grid);
    gradPsip[1] =  dg::evaluate( mag.psipZ(), grid);
    gradPsip[2] =  result; //zero
    DVec hoo = dg::pullback( dg::geo::Hoo( mag), grid);
    feltor::Variables var = {
        feltor, p,mag, gradPsip, gradPsip, hoo
    };
    /////////////////////The initial field///////////////////////////////////////////
    double time = 0.;
    std::array<std::array<DVec,2>,2> y0;
    try{
        y0 = feltor::initial_conditions.at(p.initne)( feltor, grid, p,mod_mag );
    }catch ( std::out_of_range& error){
        std::cerr << "Warning: initne parameter '"<<p.initne<<"' not recognized! Is there a spelling error? I assume you do not want to continue with the wrong initial condition so I exit! Bye Bye :)\n";
        return -1;
    }
    { //make the HVecs temporaries

    bool fixed_profile;
    HVec profile = dg::evaluate( dg::zero, grid);
    HVec source_profile;
    try{
        source_profile = feltor::source_profiles.at(p.source_type)(
            fixed_profile, profile, grid, p,  mod_mag);
    }catch ( std::out_of_range& error){
        std::cerr << "Warning: source_type parameter '"<<p.source_type<<"' not recognized! Is there a spelling error? I assume you do not want to continue with the wrong source so I exit! Bye Bye :)\n";
        return -1;
    }
    feltor.set_source( fixed_profile, dg::construct<DVec>(profile),
        p.source_rate, dg::construct<DVec>(source_profile)
    );
    }

    ////////////////////////create timer and timestepper
    //
    dg::Timer t;
    unsigned step = 0;
    dg::Karniadakis< std::array<std::array<dg::DVec,2>,2 >,
        feltor::FeltorSpecialSolver<
            Geometry, IDMatrix, DMatrix, DVec>
        > karniadakis( grid, p, mag);
    {
    HVec h_wall = dg::pullback( wall, grid);
    HVec h_sheath = dg::pullback( sheath, grid);
    HVec h_velocity = dg::pullback( direction, grid);
    feltor.set_wall_and_sheath( p.damping_rate, dg::construct<DVec>( h_wall), p.sheath_rate, dg::construct<DVec>(h_sheath), dg::construct<DVec>(h_velocity));
    implicit.set_wall_and_sheath( p.damping_rate, dg::construct<DVec>( h_wall), p.sheath_rate, dg::construct<DVec>(h_sheath));
    karniadakis.solver().set_wall_and_sheath( p.damping_rate, dg::construct<DVec>( h_wall), p.sheath_rate, dg::construct<DVec>(h_sheath));
    }

    std::cout << "Initialize Timestepper" << std::endl;
    karniadakis.init( feltor, implicit, time, y0, p.dt);
    std::cout << "Done!" << std::endl;

    std::map<std::string, const dg::DVec* > v4d;
    v4d["ne-1 / "] = &y0[0][0],  v4d["ni-1 / "] = &y0[0][1];
    v4d["Ue / "]   = &feltor.velocity(0), v4d["Ui / "]   = &feltor.velocity(1);
    v4d["Phi / "] = &feltor.potential(0); v4d["Apar / "] = &feltor.induction();
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
    file::file2Json( "window_params.json", js, file::comments::are_discarded);
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
    while ( !glfwWindowShouldClose( w ))
    {
        title << std::fixed;
        title << "t = "<<time<<"   ";
        for( auto pair : v4d)
        {
            if(pair.first == "Phi / ")
            {
                //dg::assign( feltor.lapMperpP(0), hvisual);
                dg::assign( *pair.second, hvisual);
            }
            else if(pair.first == "ne-1 / " || pair.first == "ni-1 / ")
            {
                dg::assign( *pair.second, hvisual);
                //dg::blas1::axpby( 1., hvisual, -1., profile, hvisual);
            }
            else
                dg::assign( *pair.second, hvisual);
            dg::blas2::gemv( equi, hvisual, visual);
            colors.scalemax() = dg::blas1::reduce(
                visual, 0., dg::AbsMax<double>() );
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
                    karniadakis.step( feltor, implicit, time, y0);
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
            for( auto& record : feltor::diagnostics2d_list)
            {
                if( std::find( feltor::energies.begin(), feltor::energies.end(), record.name) != feltor::energies.end())
                {
                    std::cout << record.name<<" : ";
                    record.function( result, var);
                    double norm = dg::blas1::dot( result, feltor.vol3d());
                    energy += norm;
                    std::cout << norm<<std::endl;

                }
                if( std::find( feltor::energy_diff.begin(), feltor::energy_diff.end(), record.name) != feltor::energy_diff.end())
                {
                    std::cout << record.name<<" : ";
                    record.function( result, var);
                    double norm = dg::blas1::dot( result, feltor.vol3d());
                    ediff += norm;
                    std::cout << norm<<std::endl;
                }

            }
            dEdt = (energy - E0)/deltat;
            E0 = energy;
            accuracy  = 2.*fabs( (dEdt - ediff)/( dEdt + ediff));

            std::cout << "\tTime "<<time<<"\n";
            std::cout <<"\td E/dt = " << dEdt
              <<" Lambda = " << ediff
              <<" -> Accuracy: " << accuracy << "\n";
            double max_ue = dg::blas1::reduce(
                feltor.velocity(0), 0., dg::AbsMax<double>() );
            MPI_OUT std::cout << "\tMaximum ue "<<max_ue<<"\n";
            //----------------Test if induction equation holds
            if( p.beta != 0)
            {
                dg::blas1::pointwiseDot(
                    feltor.density(0), feltor.velocity(0), dvisual);
                dg::blas1::pointwiseDot( p.beta,
                    feltor.density(1), feltor.velocity(1), -p.beta, dvisual);
                double norm  = dg::blas2::dot( dvisual, feltor.vol3d(), dvisual);
                dg::blas1::axpby( -1., feltor.lapMperpA(), 1., dvisual);
                double error = dg::blas2::dot( dvisual, feltor.vol3d(), dvisual);
                std::cout << "\tRel. Error Induction "<<sqrt(error/norm) <<"\n";
            }

        }
        t.toc();
        std::cout << "\n\t Step "<<step << " at time  "<<time;
        std::cout << "\n\t Average time for one step: "<<t.diff()/(double)p.itstp/(double)p.inner_loop<<"\n\n";
    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////
    return 0;

}
