#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <sstream>
#include <cmath>
#include <csignal>

#ifdef WITH_MPI
#include <mpi.h>
#endif //WITH_MPI

#ifndef WITHOUT_GLFW
#include "draw/host_window.h"
#endif // WITHOUT_GLFW

#include "dg/file/file.h"
#include "feltor.h"
#include "init.h"
#include "feltordiag.h"
#include "init_from_file.h"


using Vector = std::array<std::array<dg::x::DVec, 2>,2>;

int main( int argc, char* argv[])
{
#ifdef WITH_MPI
    ////////////////////////////////setup MPI///////////////////////////////
    dg::mpi_init( argc, argv);
    MPI_Comm comm;
    dg::mpi_init3d( dg::DIR, dg::DIR, dg::PER, comm, std::cin, true);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    std::signal(SIGINT, common::sigterm_handler);
    std::signal(SIGTERM, common::sigterm_handler);
#endif //WITH_MPI
    ////////////////////////Parameter initialisation//////////////////////////

    dg::file::WrappedJsonValue js( dg::file::error::is_throw);

    common::parse_input_file<feltor::Parameters>( argc, argv, js);

    common::parse_geometry_file( argv[1], js);

    const feltor::Parameters p( js);

    std::string inputfile = js.toStyledString();
    DG_RANK0 std::cout << inputfile <<  std::endl;

    // create a timer
    dg::Timer t;

    dg::geo::TokamakMagneticField mag, mod_mag, unmod_mag;
    dg::geo::CylindricalFunctor wall, transition;
    common::create_mag_wall( argv[1], js, mag, mod_mag, unmod_mag, wall, transition);
#ifdef WITH_MPI
    common::check_Nz( p.Nz, comm);
#endif //WITH_MPI

    //Make grids
    auto box = common::box( js);
    dg::x::CylindricalGrid3d grid( box.at("Rmin"), box.at("Rmax"),
            box.at("Zmin"), box.at("Zmax"), 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.symmetric ? 1 : p.Nz, p.bcxN, p.bcyN, dg::PER
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
        );

    ////////////////////////////////set up computations///////////////////////////


    DG_RANK0 std::cout << "# Constructing Feltor...\n";
    //feltor::Filter<dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DVec> filter( grid, js);
    feltor::Explicit< dg::x::CylindricalGrid3d, dg::x::IDMatrix,
        dg::x::DMatrix, dg::x::DVec> feltor( grid, p, mag, js);
    DG_RANK0 std::cout << "# Done!\n";

    feltor.set_wall( p.wall_rate, dg::construct<dg::x::DVec>( dg::pullback(
                    wall, grid)), p.nwall, p.uwall );

    dg::geo::CylindricalFunctor sheath, sheath_coordinate =
        [](double x, double y){return 0.;};
    if( p.sheath_bc != "none")
    {
        common::create_and_set_sheath( argv[1], js, mag, wall, sheath,
                sheath_coordinate, grid, feltor);
    }

    DG_RANK0 std::cout << "# Set Source \n";
    try{
        bool fixed_profile;
        dg::x::HVec ne_profile, source_profile;
        double minne = 0., minrate = 0., minalpha = 0.;
        source_profile = feltor::source_profiles( feltor,
            fixed_profile, ne_profile, grid, mag, unmod_mag, js["source"],
            minne, minrate, minalpha);
        feltor.set_source( fixed_profile,
                dg::construct<dg::x::DVec>(ne_profile), p.source_rate,
                dg::construct<dg::x::DVec>(source_profile),
                minne, minrate, minalpha);
    }catch ( std::out_of_range& error){
        DG_RANK0 std::cerr << "ERROR: in source: "<<error.what();
        DG_RANK0 std::cerr <<"Is there a spelling error? I assume you do not want to continue with the wrong source so I exit! Bye Bye :)"<<std::endl;
        dg::abort_program();
    }
    /// /////////////The initial field//////////////////////////////////////////
    double time = 0.;
    Vector y0;
    DG_RANK0 std::cout << "# Set Initial conditions ... \n";
    t.tic();
    if( argc == 4 )
    {
        try{
            y0 = feltor::init_from_file(argv[3], grid, p, time);
        }catch (std::exception& e){
            DG_RANK0 std::cerr << "ERROR occured initializing from file "<<argv[3]<<std::endl;
            DG_RANK0 std::cerr << e.what()<<std::endl;
            dg::abort_program();
        }
    }
    else
    {
        try{
            y0 = feltor::initial_conditions(feltor, grid, p, mag, unmod_mag,
                    js["init"], time, sheath_coordinate );
#ifdef WITH_NAVIER_STOKES
            std::string advection = js["advection"].get("type", "velocity-staggered").asString();
            if( advection == "log-staggered" || advection == "staggered-direct")
                dg::blas1::transform( y0[0], y0[0], dg::LN<double>());
            else if( advection == "staggered")
            {
                // MW: Not correct, density is not staggered,only
                // works when velocity is zero
                dg::blas1::pointwiseDot( y0[0], y0[1], y0[1]);
            }
#endif // WITH_NAVIER_STOKES
        }catch ( dg::Error& error){
            DG_RANK0 std::cerr << error.what();
            DG_RANK0 std::cerr << "Is there a spelling error? I assume you do not want to continue with the wrong parameter so I exit! Bye Bye :)\n";
            dg::abort_program();
        }
    }
    t.toc();
    DG_RANK0 std::cout << "# ... took  "<<t.diff()<<"s\n";


    ///////////////////////////////////////////////////////////////////////////
    double t_output = time;
    unsigned failed =0;
    bool adaptive = false;
    auto odeint = common::init_timestepper<Vector>( js, feltor, time, y0, adaptive, failed);

    /// //////////////////////////set up netcdf/////////////////////////////////////
    if( p.output == "netcdf")
    {
        if( argc != 3 && argc != 4)
        {
            DG_RANK0 std::cerr << "ERROR: Wrong number of arguments for netcdf output!\nUsage: "
                    << argv[0]<<" [input.json] [output.nc]\n OR \n"
                    << argv[0]<<" [input.json] [output.nc] [initial.nc] "<<std::endl;
            dg::abort_program();
        }
        std::string file_name = argv[2];
        dg::file::NcFile file;
        try{
            file.open( file_name, dg::file::nc_clobber);
            feltor::write_global_attributes( file, argc, argv, inputfile);
#ifdef WRITE_POL_FILE
            file_pol.open( "polarisation.nc", dg::file::nc_clobber);
            feltor::write_global_attributes( file_pol, argc, argv, inputfile);
            file_pol.defput_dim( "x", {{"axis", "X"}}, grid.abscissas(0));
            file_pol.defput_dim( "y", {{"axis", "Y"}}, grid.abscissas(1));
            file_pol.defput_dim( "z", {{"axis", "Z"}}, grid.abscissas(2));
#endif

        }catch( std::exception& e)
        {
            DG_RANK0 std::cerr << "ERROR creating file "<<file_name<<std::endl;
            DG_RANK0 std::cerr << e.what()<<std::endl;
            dg::abort_program();
        }

        // helper variables for output computations
        dg::x::CylindricalGrid3d g3d_out( grid.x0(), grid.x1(), grid.y0(), grid.y1(), 0, 2.*M_PI,
            p.n, p.Nx/p.cx, p.Ny/p.cy, p.symmetric ? 1 : p.Nz, p.bcxN, p.bcyN, dg::PER
            #ifdef WITH_MPI
            , comm
            #endif //WITH_MPI
            );

        std::array<dg::x::DVec, 3> gradPsip; //referenced by Variables
        gradPsip[0] =  dg::evaluate( mag.psipR(), grid);
        gradPsip[1] =  dg::evaluate( mag.psipZ(), grid);
        gradPsip[2] =  dg::evaluate( dg::zero, grid); //zero
        dg::x::HVec resultH = dg::evaluate( dg::zero, grid);
        dg::x::DVec resultD = dg::evaluate( dg::zero, grid);
        dg::x::HVec resultH_out = dg::evaluate( dg::zero, g3d_out);
        dg::x::DVec resultD_out = dg::evaluate( dg::zero, g3d_out);
        feltor::Variables var{
            feltor, y0, p, mag, gradPsip, gradPsip, gradPsip, gradPsip,
            0., // duration
            &failed // nfailed
        };

        // STATIC OUTPUT
        //create & output static 3d variables into file
        file.defput_dim( "x", {{"axis", "X"}, {"long_name", "R coordinate in Cylindrical R,Z,Phi coordinate system"}, {"units", "rho_s"}}, g3d_out.abscissas(0));
        file.defput_dim( "y", {{"axis", "Y"}, {"long_name", "Z coordinate in Cylindrical R,Z,Phi coordinate system"}, {"units", "rho_s"}}, g3d_out.abscissas(1));
        file.defput_dim( "z", {{"axis", "Z"}, {"long_name", "Phi coordinate in Cylindrical R,Z,Phi coordinate system"}, {"units", "rho_s"}}, g3d_out.abscissas(2));
        for( auto& record: feltor::diagnostics3d_static_list)
        {
            record.function( resultH_out, var.mag, g3d_out);
            file.defput_var( record.name, {"z", "y", "x"}, record.atts,
                    {g3d_out}, resultH_out);
        }
        //create & output static 2d variables into file
        feltor::write_static_list( file, feltor::diagnostics2d_static_list,
            var, grid, g3d_out, transition);

        if( p.calibrate)
        {
            file.close();
#ifdef WITH_MPI
            MPI_Finalize();
#endif //WITH_MPI
            return 0;
        }
        // DYNAMIC OUTPUT
        file.def_dimvar_as<double>( "time", NC_UNLIMITED, {{"axis", "T"}, {"units", "Omega_ci^-1"}});

        feltor::WriteIntegrateDiagnostics2dList diag2d( file, grid, g3d_out,
            feltor::generate_equation_list( js));
        file.defput_dim( "xr", {{"axis", "X"}}, grid.abscissas(0));
        file.defput_dim( "yr", {{"axis", "Y"}}, grid.abscissas(1));
        file.defput_dim( "zr", {{"axis", "Z"}}, grid.abscissas(2));
        for( auto& record: feltor::restart3d_list)
            file.def_var_as<double>( record.name, {"zr", "yr", "xr"}, record.atts);
        // Probes need to be the last because they define dimensions in subgroup
        dg::file::Probes probes( file, grid, dg::file::parse_probes(js));

        ///////////////////////////////////first output/////////////////////////
        DG_RANK0 std::cout << "# First output ... \n";
        //first, update feltor (to get potential etc.)
        {
            std::array<std::array<dg::x::DVec,2>,2> y1(y0);
            try{
                feltor( time, y0, y1);
            } catch( dg::Fail& fail) {
                DG_RANK0 std::cerr << "CG failed to converge in first step to "
                                  <<fail.epsilon()<<std::endl;
                file.close();
                dg::abort_program();
            }
        }

        DG_RANK0 std::cout << "# Write restart ...\n";
        for( auto& record: feltor::restart3d_list)
        {
            record.function( resultD, feltor);
            file.put_var( record.name, {grid}, resultD);
        }

        DG_RANK0 std::cout << "# Write diag1d ...\n";
        file.put_var( "time", {0}, time);
        for( auto& record: feltor::diagnostics1d_list)
        {
            file.def_var_as<double>( record.name, {"time"}, record.atts);
            file.put_var( record.name, {0}, record.function( var));
        }
        DG_RANK0 std::cout << "# Write diag2d ...\n";
        diag2d.write( time, var );
        DG_RANK0 std::cout << "# Write diag4d ...\n";
        dg::MultiMatrix<dg::x::DMatrix, dg::x::DVec> project(
            dg::create::fast_projection( grid, 1, p.cx, p.cy));
        for( auto& record : feltor::diagnostics3d_list)
        {
            record.function ( resultD, var);
            dg::apply( project, resultD, resultD_out);
            file.defput_var( record.name, {"time", "z", "y", "x"},
                    record.atts, {0, g3d_out}, resultD_out);
        }


        DG_RANK0 std::cout << "# Write static probes ...\n";
        probes.static_write( feltor::diagnostics2d_static_list, var, grid);
        DG_RANK0 std::cout << "# Write probes ...\n";
        probes.write( time, feltor::probe_list, var);

        DG_RANK0 std::cout << "# Close file ...\n";
        file.close();
        size_t start = 1;
        DG_RANK0 std::cout << "# First write successful!\n";
        ///////////////////////////////Timeloop/////////////////////////////////

        t.tic();
        bool abort = false;
        for( unsigned i=1; i<=p.maxout; i++)
        {
            dg::Timer ti;
            ti.tic();
            for( unsigned j=1; j<=p.itstp; j++)
            {
                try{
                    odeint->integrate( time, y0, t_output + j*p.deltaT, y0,
                        j<p.itstp ? dg::to::at_least :  dg::to::exact);
                }
                catch( dg::Fail& fail){ // a specific exception
                    DG_RANK0 std::cerr << "ERROR failed to converge to "<<fail.epsilon()<<"\n";
                    DG_RANK0 std::cerr << "Does simulation respect CFL condition?"<<std::endl;
                    DG_RANK0 std::cerr << "Writing last Output and exit..."<<std::endl;
                    abort = true; // exit gracefully
                }
                catch( std::exception& fail) { // more generic exception
                    DG_RANK0 std::cerr << "ERROR in timestepper\n";
                    DG_RANK0 std::cerr << fail.what()<<std::endl;
                    DG_RANK0 std::cerr << "Writing last Output and exit..."<<std::endl;
                    abort = true;
                }
                dg::Timer tti;
                tti.tic();

                probes.buffer(time, feltor::probe_list, var);
                diag2d.buffer( time, var);

                DG_RANK0 std::cout << "\tTime "<<time<<"\n";
                double max_ue = dg::blas1::reduce(
                    feltor.velocity(0), 0., dg::AbsMax<double>() );
                DG_RANK0 std::cout << "\tMaximum ue "<<max_ue<<"\n";
                if( adaptive )
                {
                    DG_RANK0 std::cout << "\tdt "<<odeint->get_dt()<<"\n";
                    DG_RANK0 std::cout << "\tfailed "<<*var.nfailed<<"\n";
                }
                tti.toc();
                DG_RANK0 std::cout << " Time for internal diagnostics "<<tti.diff()<<"s\n";
                if( abort) break;
            }
            ti.toc();
            var.duration = ti.diff();
            t_output += p.itstp*p.deltaT;
            // Does not work due to direct application of Laplace
            // The Laplacian of Aparallel looks smooth in paraview
            ////----------------Test if ampere equation holds
            //if( p.beta != 0 && !abort)
            //{
            //    feltor.compute_lapMperpA( resultD);
            //    double norm  = dg::blas2::dot( resultD, feltor.vol3d(), resultD);
            //    dg::blas1::pointwiseDot( -p.beta,
            //        feltor.density(0), feltor.velocity(0), p.beta,
            //        feltor.density(1), feltor.velocity(1), -1., resultD);
            //    double error = dg::blas2::dot( resultD, feltor.vol3d(), resultD);
            //    DG_RANK0 std::cout << "\tRel. Error Ampere "<<sqrt(error/norm) <<"\n";
            //}
            DG_RANK0 std::cout << "\n\t Step: Time "<<time <<" of " << p.Tend;
            DG_RANK0 std::cout << "\n\t Average time for one inner loop: "
                        << var.duration/(double)p.itstp<<"s";

            ti.tic();
            //////////////////////////write fields////////////////////////
            file.open( file_name, dg::file::nc_write);
            probes.flush();
            diag2d.flush( var);

            for( auto& record: feltor::restart3d_list)
            {
                record.function( resultD, feltor);
                file.put_var( record.name, {grid}, resultD);
            }

            file.put_var( "time", {start}, time);
            for( auto& record: feltor::diagnostics1d_list)
                file.put_var( record.name, {start}, record.function( var));
            for( auto& record : feltor::diagnostics3d_list)
            {
                record.function ( resultD, var);
                dg::apply( project, resultD, resultD_out);
                file.put_var( record.name, {start, g3d_out}, resultD_out);
            }

            file.close();
            start++;
            ti.toc();
            DG_RANK0 std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
            if( abort) break;
        }
        t.toc();
        unsigned hour = (unsigned)floor(t.diff()/3600);
        unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
        double second = t.diff() - hour*3600 - minute*60;
        DG_RANK0 std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
        DG_RANK0 std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
        DG_RANK0 std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/inner loop\n";
    }
#ifndef WITHOUT_GLFW
    if( p.output == "glfw")
    {
        dg::Timer t;

        std::map<std::string, const dg::x::DVec* > v4d;
        v4d["ne-1 / "] = &feltor.density(0),  v4d["ni-1 / "] = &feltor.density(1);
        v4d["Ue / "]   = &feltor.velocity(0), v4d["Ui / "]   = &feltor.velocity(1);
        v4d["Phi / "] = &feltor.potential(0); v4d["Apar / "] = &feltor.aparallel();
        /////////////////////////set up transfer for glfw
        dg::DVec dvisual( grid.size(), 0.);
        dg::HVec hvisual( grid.size(), 0.), visual(hvisual), avisual(hvisual);
        dg::IHMatrix equi = dg::create::backscatter( grid);
        draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);

        /////////glfw initialisation ////////////////////////////////////////////
        //
        std::stringstream title;
        unsigned red = js["output"]["window"].get("reduction", 1).asUInt();
        double rows = js["output"]["window"]["rows"].asDouble(), cols = p.Nz/red+1,
               width = js["output"]["window"]["width"].asDouble(), height = js["output"]["window"]["height"].asDouble();
        if ( p.symmetric ) cols = rows, rows = 1;
        GLFWwindow* w = draw::glfwInitAndCreateWindow( cols*width, rows*height, "");
        draw::RenderHostData render(rows, cols);

        std::cout << "Begin computation \n";
        std::cout << std::scientific << std::setprecision( 2);
        dg::Average<dg::IHMatrix, dg::HVec> toroidal_average( grid,
                dg::coo3d::z);
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
            for( unsigned i=1; i<=p.itstp; i++)
            {
                odeint->integrate( time, y0, t_output + i*p.deltaT, y0,
                         i<p.itstp ? dg::to::at_least :  dg::to::exact);
                std::cout << "Time "<<time<<" t_out "<<t_output<<" deltaT "<<p.deltaT<<" i "<<i<<" itstp "<<p.itstp<<"\n";

                double max_ue = dg::blas1::reduce(
                    feltor.velocity(0), 0., dg::AbsMax<double>() );
                std::cout << "\tMaximum ue "<<max_ue<<"\n";
                if( adaptive )
                {
                    std::cout << "\tdt "<<odeint->get_dt()<<"\n";
                    std::cout << "\tfailed "<<failed<<"\n";
                }
                //----------------Test if ampere equation holds
                // Does not work due to direct application of Laplace
                // The Laplacian of Aparallel looks smooth in paraview
                //if( p.beta != 0)
                //{
                //    feltor.compute_lapMperpA( dvisual);
                //    double norm  = dg::blas2::dot( dvisual, feltor.vol3d(), dvisual);
                //    dg::blas1::pointwiseDot( -p.beta,
                //        feltor.density(0), feltor.velocity(0), p.beta,
                //        feltor.density(1), feltor.velocity(1), -1., dvisual);
                //    double error = dg::blas2::dot( dvisual, feltor.vol3d(), dvisual);
                //    DG_RANK0 std::cout << "\tRel. Error Ampere "<<sqrt(error/norm) <<"\n";
                //}
            }
            t_output += p.itstp*p.deltaT;
            t.toc();
            std::cout << "\n\t Time  "<<time<<" of "<<p.Tend;
            std::cout << "\n\t Average time for one inner loop: "<<t.diff()/(double)p.itstp<<"\n\n";
        }
        glfwTerminate();
    }
#endif //WITHOUT_GLFW
#ifdef WITH_MPI
    MPI_Finalize();
#endif //WITH_MPI

    return 0;

}
