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


    ///PROBE ADDITIONS!!!
    dg::HVec R_probe, Z_probe, P_probe;
    dg::file::WrappedJsonValue js_probes( dg::file::error::is_throw);
    if( p.probes)
    {
        common::parse_probes( argv[1], js, js_probes, R_probe, Z_probe, P_probe);
    }

    unsigned num_pins=R_probe.size();
    // create interpolation matrix
    dg::x::IHMatrix probe_interpolate = dg::create::interpolation( R_probe, Z_probe, P_probe, grid);
    // Create helper storage probe variables
#ifdef WITH_MPI
    // every processor gets the probes (slightly inefficient...)
    dg::MHVec simple_probes(R_probe, grid.communicator());
#else //WITH_MPI
    dg::HVec simple_probes(num_pins);
#endif
    std::map<std::string, std::vector<dg::x::HVec>> simple_probes_intern;
    for( auto& record : feltor::probe_list)
        simple_probes_intern[record.name] = std::vector<dg::x::HVec>(p.itstp, simple_probes);
    std::vector<double> time_intern(p.itstp);




    ///////////////////////////////////////////////////////////////////////////
    std::array<dg::x::DVec, 3> gradPsip; //referenced by Variables
    gradPsip[0] =  dg::evaluate( mag.psipR(), grid);
    gradPsip[1] =  dg::evaluate( mag.psipZ(), grid);
    gradPsip[2] =  dg::evaluate( dg::zero, grid); //zero
    unsigned failed =0;
    feltor::Variables var{
        feltor, y0, p, mag, gradPsip, gradPsip, gradPsip, gradPsip,
        0., // duration
        &failed // nfailed
    };
    double t_output = time;
    bool adaptive = false;
    auto odeint = common::init_timestepper<Vector>( js, feltor, time, y0, adaptive, failed);

    unsigned maxout = js["output"].get( "maxout", 0).asUInt();
    std::string output_mode = js["timestepper"].get(
            "output-mode", "Tend").asString();
    double Tend = 0, deltaT = 0.;
    if( output_mode == "Tend")
    {
        Tend = js["timestepper"].get( "Tend", 1).asDouble();
        deltaT = Tend/(double)(maxout*p.itstp);
    }
    else if( output_mode == "deltaT")
    {
        deltaT = js["timestepper"].get( "deltaT", 1).asDouble()/(double)(p.itstp);
        Tend = deltaT*(double)(maxout*p.itstp);
    }
    else
        throw std::runtime_error( "timestepper: output-mode "+output_mode+" not recognized!\n");
    /// //////////////////////////set up netcdf/////////////////////////////////////
    if( p.output == "netcdf")
    {
        // helper variables for output computations
        unsigned cx = js["output"]["compression"].get(0u,1).asUInt();
        unsigned cy = js["output"]["compression"].get(1u,1).asUInt();
        unsigned n_out = p.n, Nx_out = p.Nx/cx, Ny_out = p.Ny/cy, Nz_out = p.Nz;
        dg::x::CylindricalGrid3d g3d_out( grid.x0(), grid.x1(), grid.y0(), grid.y1(), 0, 2.*M_PI,
            n_out, Nx_out, Ny_out, p.symmetric ? 1 : Nz_out, p.bcxN, p.bcyN, dg::PER
            #ifdef WITH_MPI
            , comm
            #endif //WITH_MPI
            );
        std::unique_ptr<typename dg::x::CylindricalGrid3d::perpendicular_grid>
            g2d_out_ptr  ( dynamic_cast<typename
                    dg::x::CylindricalGrid3d::perpendicular_grid*>(
                        g3d_out.perp_grid()));
#ifdef WITH_MPI
        unsigned local_size2d = g2d_out_ptr->local().size();
#else
        unsigned local_size2d = g2d_out_ptr->size();
#endif
        std::map<std::string, dg::Simpsons<dg::x::HVec>> time_integrals;
        dg::Average<dg::x::HVec> toroidal_average( g3d_out, dg::coo3d::z, "simple");
        dg::MultiMatrix<dg::x::HMatrix,dg::x::HVec> projectH =
            dg::create::fast_projection( grid, 1, cx, cy);
        dg::MultiMatrix<dg::x::DMatrix,dg::x::DVec> projectD =
            dg::create::fast_projection( grid, 1, cx, cy);
        dg::x::HVec transferH( dg::evaluate(dg::zero, g3d_out));
        dg::x::DVec transferD( dg::evaluate(dg::zero, g3d_out));
        dg::x::HVec transferH2d = dg::evaluate( dg::zero, *g2d_out_ptr);
        dg::x::DVec transferD2d = dg::evaluate( dg::zero, *g2d_out_ptr);
        dg::x::HVec resultH = dg::evaluate( dg::zero, grid);
        dg::x::DVec resultD = dg::evaluate( dg::zero, grid);
        if( argc != 3 && argc != 4)
        {
            DG_RANK0 std::cerr << "ERROR: Wrong number of arguments for netcdf output!\nUsage: "
                    << argv[0]<<" [input.json] [output.nc]\n OR \n"
                    << argv[0]<<" [input.json] [output.nc] [initial.nc] "<<std::endl;
            dg::abort_program();
        }
        dg::file::NC_Error_Handle err;
        std::string file_name = argv[2];
        int ncid=-1;
        try{
            DG_RANK0 err = nc_create( file_name.data(), NC_NETCDF4|NC_CLOBBER, &ncid);
#ifdef WRITE_POL_FILE
            DG_RANK0 err_pol = nc_create( "polarisation.nc", NC_NETCDF4|NC_CLOBBER, &ncid_pol);
#endif

        }catch( std::exception& e)
        {
            DG_RANK0 std::cerr << "ERROR creating file "<<file_name<<std::endl;
            DG_RANK0 std::cerr << e.what()<<std::endl;
            dg::abort_program();
        }
        /// Set global attributes
        std::map<std::string, std::string> att;
        att["title"] = "Output file of feltor/src/feltor/feltor.cu";
        att["Conventions"] = "CF-1.8";
        ///Get local time and begin file history
        auto ttt = std::time(nullptr);
        std::ostringstream oss;
        ///time string  + program-name + args
        oss << std::put_time(std::localtime(&ttt), "%F %T %Z");
        for( int i=0; i<argc; i++) oss << " "<<argv[i];
        att["history"] = oss.str();
        att["comment"] = "Find more info in feltor/src/feltor/feltor.tex";
        att["source"] = "FELTOR";
        att["git-hash"] = GIT_HASH;
        att["git-branch"] = GIT_BRANCH;
        att["compile-time"] = COMPILE_TIME;
        att["references"] = "https://github.com/feltor-dev/feltor";
        att["inputfile"] = inputfile;
        for( auto pair : att)
        {
            DG_RANK0 err = nc_put_att_text( ncid, NC_GLOBAL,
                pair.first.data(), pair.second.size(), pair.second.data());
#ifdef WRITE_POL_FILE
            DG_RANK0 err_pol = nc_put_att_text( ncid_pol, NC_GLOBAL,
                pair.first.data(), pair.second.size(), pair.second.data());
#endif
        }

        // Define dimensions (t,z,y,x)
        int dim_ids[4], restart_dim_ids[3], tvarID;
        DG_RANK0 err = dg::file::define_dimensions( ncid, &dim_ids[1], g3d_out,
                {"z", "y", "x"});
        if( !p.calibrate)
            DG_RANK0 err = dg::file::define_time( ncid, "time", dim_ids, &tvarID);
        DG_RANK0 err = dg::file::define_dimensions( ncid, restart_dim_ids, grid,
                {"zr", "yr", "xr"});
#ifdef WRITE_POL_FILE
        DG_RANK0 err_pol = dg::file::define_dimensions( ncid_pol, dim_ids_pol, grid,
                {"z", "y", "x"});
#endif
        int dim_ids3d[3] = {dim_ids[0], dim_ids[2], dim_ids[3]};
        bool write2d = true;
#ifdef WITH_MPI
        //only the globally first slice should write
        if( !(g3d_out.local().z0() - g3d_out.global().z0() < 1e-14) ) write2d = false;
#endif //WITH_MPI

        //create & output static 3d variables into file
        for ( auto& record : feltor::diagnostics3d_static_list)
        {
            int vecID;
            DG_RANK0 err = nc_def_var( ncid, record.name.data(), NC_DOUBLE, 3,
                &dim_ids[1], &vecID);
            DG_RANK0 err = nc_put_att_text( ncid, vecID,
                "long_name", record.long_name.size(), record.long_name.data());
            DG_RANK0 std::cout << "Computing "<<record.name<<"\n";
            record.function( transferH, var, g3d_out);
            //record.function( resultH, var, grid);
            //dg::blas2::symv( projectH, resultH, transferH);
            dg::file::put_var_double( ncid, vecID, g3d_out, transferH);
        }
        //create & output static 2d variables into file
        for ( auto& record : feltor::diagnostics2d_static_list)
        {
            int vecID;
            DG_RANK0 err = nc_def_var( ncid, record.name.data(), NC_DOUBLE, 2,
                &dim_ids[2], &vecID);
            DG_RANK0 err = nc_put_att_text( ncid, vecID,
                "long_name", record.long_name.size(), record.long_name.data());
            DG_RANK0 std::cout << "Computing2d "<<record.name<<"\n";
            //record.function( transferH, var, g3d_out); //ATTENTION: This does not work because feltor internal variables return full grid functions
            record.function( resultH, var, grid);
            dg::blas2::symv( projectH, resultH, transferH);
            if(write2d)dg::file::put_var_double( ncid, vecID, *g2d_out_ptr, transferH);
        }
        {
            // transition has to be done by hand
            int vecID;
            DG_RANK0 err = nc_def_var( ncid, "MagneticTransition", NC_DOUBLE, 2,
                &dim_ids[2], &vecID);
            std::string long_name = "The region where the magnetic field is modified";
            DG_RANK0 err = nc_put_att_text( ncid, vecID,
                "long_name", long_name.size(), long_name.data());
            DG_RANK0 std::cout << "Computing2d MagneticTransition\n";
            resultH = dg::pullback( transition, grid);
            dg::blas2::symv( projectH, resultH, transferH);
            if(write2d)dg::file::put_var_double( ncid, vecID, *g2d_out_ptr, transferH);
        }

        if( p.calibrate)
        {
            DG_RANK0 err = nc_close(ncid);
#ifdef WITH_MPI
            MPI_Finalize();
#endif //WITH_MPI
            return 0;
        }

        //Create field IDs
        // the vector ids
        std::map<std::string, int> id1d, id3d, id4d, restart_ids;
        for( auto& record : feltor::diagnostics3d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id4d[name] = 0;//creates a new id4d entry for all processes
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 4, dim_ids,
                &id4d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id4d.at(name), "long_name", long_name.size(),
                long_name.data());
        }
        for( auto& record : feltor::restart3d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            restart_ids[name] = 0;//creates a new entry for all processes
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3,
                    restart_dim_ids, &restart_ids.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, restart_ids.at(name),
                    "long_name", long_name.size(), long_name.data());
        }

        std::vector<std::vector<feltor::Record>> equation_list;
        bool equation_list_exists = js["output"].isMember("equations");
        if( equation_list_exists)
        {
            for( unsigned i=0; i<js["output"]["equations"].size(); i++)
            {
                std::string eqn = js["output"]["equations"][i].asString();
                if( eqn == "Basic")
                    equation_list.push_back(feltor::basicDiagnostics2d_list);
                else if( eqn == "Mass-conserv")
                    equation_list.push_back(feltor::MassConsDiagnostics2d_list);
                else if( eqn == "Energy-theorem")
                    equation_list.push_back(feltor::EnergyDiagnostics2d_list);
                else if( eqn == "Toroidal-momentum")
                    equation_list.push_back(feltor::ToroidalExBDiagnostics2d_list);
                else if( eqn == "Parallel-momentum")
                    equation_list.push_back(feltor::ParallelMomDiagnostics2d_list);
                else if( eqn == "Zonal-Flow-Energy")
                    equation_list.push_back(feltor::RSDiagnostics2d_list);
                else if( eqn == "COCE")
                    equation_list.push_back(feltor::COCEDiagnostics2d_list);
                else
                    throw std::runtime_error( "output: equations: "+eqn+" not recognized!\n");
            }
        }
        else // default diagnostics
        {
            equation_list.push_back(feltor::basicDiagnostics2d_list);
            equation_list.push_back(feltor::MassConsDiagnostics2d_list);
            equation_list.push_back(feltor::EnergyDiagnostics2d_list);
            equation_list.push_back(feltor::ToroidalExBDiagnostics2d_list);
            equation_list.push_back(feltor::ParallelMomDiagnostics2d_list);
            equation_list.push_back(feltor::RSDiagnostics2d_list);
        }

        std::string m_list;
        for( auto& m_list : equation_list)
        {
            for( auto& record : m_list)
            {
                std::string name = record.name + "_ta2d";
                std::string long_name = record.long_name + " (Toroidal average)";
                id3d[name] = 0;//creates a new id3d entry for all processes
                DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids3d,
                    &id3d.at(name));
                DG_RANK0 err = nc_put_att_text( ncid, id3d.at(name), "long_name",
                    long_name.size(), long_name.data());

                name = record.name + "_2d";
                long_name = record.long_name + " (Evaluated on phi = 0 plane)";
                id3d[name] = 0;
                DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3,
                    dim_ids3d, &id3d.at(name));
                DG_RANK0 err = nc_put_att_text( ncid, id3d.at(name), "long_name",
                    long_name.size(), long_name.data());
            }
        }

        for( auto& record : feltor::diagnostics1d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id1d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1,
                &dim_ids[0], &id1d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id1d.at(name), "long_name",
                long_name.size(), long_name.data());
        }

        //Probes:
        int probe_grp_id;
        if( p.probes)
        {
            DG_RANK0 err = nc_def_grp(ncid,"probes",&probe_grp_id);
            std::string format = js_probes["format"].toStyledString();
            DG_RANK0 err = nc_put_att_text( probe_grp_id, NC_GLOBAL,
                "format", format.size(), format.data());
        }
        int probe_dim_ids[2];
        int probe_timevarID;
        std::map<std::string, int> probe_id_field;
        if( p.probes)
        {
            dg::Grid1d g1d( 0,1,1,num_pins);
            DG_RANK0 err = dg::file::define_dimensions( probe_grp_id,
                    probe_dim_ids, &probe_timevarID, g1d, {"time", "x"});
            int R_pin_id, Z_pin_id, P_pin_id;
            DG_RANK0 err = nc_def_var(probe_grp_id, "R", NC_DOUBLE, 1, &probe_dim_ids[1], &R_pin_id);
            DG_RANK0 err = nc_def_var(probe_grp_id, "Z", NC_DOUBLE, 1, &probe_dim_ids[1], &Z_pin_id);
            DG_RANK0 err = nc_def_var(probe_grp_id, "P", NC_DOUBLE, 1, &probe_dim_ids[1], &P_pin_id);
            DG_RANK0 err = nc_put_var_double( probe_grp_id, R_pin_id, R_probe.data());
            DG_RANK0 err = nc_put_var_double( probe_grp_id, Z_pin_id, Z_probe.data());
            DG_RANK0 err = nc_put_var_double( probe_grp_id, P_pin_id, P_probe.data());
            for( auto& record : feltor::probe_list)
            {
                std::string name = record.name;
                std::string long_name = record.long_name;
                probe_id_field[name] = 0;//creates a new id4d entry for all processes
                DG_RANK0 err = nc_def_var( probe_grp_id, name.data(),
                        NC_DOUBLE, 2, probe_dim_ids,
                        &probe_id_field.at(name));
                DG_RANK0 err = nc_put_att_text( probe_grp_id,
                        probe_id_field.at(name), "long_name", long_name.size(),
                        long_name.data());
            }
            for ( auto& record : feltor::diagnostics2d_static_list)
            {
                int vecID;
                DG_RANK0 err = nc_def_var( probe_grp_id, record.name.data(), NC_DOUBLE, 1,
                    &probe_dim_ids[1], &vecID);
                DG_RANK0 err = nc_put_att_text( probe_grp_id, vecID,
                    "long_name", record.long_name.size(), record.long_name.data());
                record.function( resultH, var, grid);
                dg::blas2::symv( probe_interpolate, resultH, simple_probes);
                DG_RANK0 nc_put_var_double( ncid, vecID,
#ifdef WITH_MPI
                    simple_probes.data().data()
#else
                    simple_probes.data()
#endif
                        );
            }
        }

        ///////////////////////////////////first output/////////////////////////
        DG_RANK0 std::cout << "First output ... \n";
        //first, update feltor (to get potential etc.)
        {
            std::array<std::array<dg::x::DVec,2>,2> y1(y0);
            try{
                feltor( time, y0, y1);
            } catch( dg::Fail& fail) {
                DG_RANK0 std::cerr << "CG failed to converge in first step to "
                                  <<fail.epsilon()<<std::endl;
                DG_RANK0 err = nc_close(ncid);
                dg::abort_program();
            }
        }

        size_t start = 0, count = 1;
        DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
        for( auto& record : feltor::diagnostics3d_list)
        {
            record.function( resultD, var);
            dg::blas2::symv( projectD, resultD, transferD);
            dg::assign( transferD, transferH);
            dg::file::put_vara_double( ncid, id4d.at(record.name), start, g3d_out, transferH);
        }
        for( auto& record : feltor::restart3d_list)
        {
            record.function( resultD, var);
            dg::assign( resultD, resultH);
            dg::file::put_var_double( ncid, restart_ids.at(record.name), grid, resultH);
        }

        for( auto& m_list : equation_list)
        {
            for( auto& record : m_list)
            {
                dg::Timer tti;
                tti.tic();
                record.function( resultD, var);
                dg::blas2::symv( projectD, resultD, transferD);
                //toroidal average
                std::string name = record.name + "_ta2d";
                dg::assign( transferD, transferH);
                toroidal_average( transferH, transferH2d, false);
                //create and init Simpsons for time integrals
                if( record.integral) time_integrals[name].init( time, transferH2d);
                tti.toc();
                DG_RANK0 std::cout<< name << " Computing average took "<<tti.diff()<<"\n";
                tti.tic();
                if(write2d) dg::file::put_vara_double( ncid, id3d.at(name), start, *g2d_out_ptr, transferH2d);
                tti.toc();
                DG_RANK0 std::cout<< name << " 2d output took "<<tti.diff()<<"\n";
                tti.tic();
                // add a slice
                name = record.name + "_2d";
                feltor::slice_vector3d( transferD, transferD2d, local_size2d);
                dg::assign( transferD2d, transferH2d);
                if( record.integral) time_integrals[name].init( time, transferH2d);
                if(write2d) dg::file::put_vara_double( ncid, id3d.at(name), start, *g2d_out_ptr, transferH2d);
                tti.toc();
                DG_RANK0 std::cout<< name << " 2d output took "<<tti.diff()<<"\n";
            }
        }
        for( auto& record : feltor::diagnostics1d_list)
        {
            double result = record.function( var);
            DG_RANK0 nc_put_vara_double( ncid, id1d.at(record.name), &start, &count, &result);
        }


        /// Probes FIRST output ///
        size_t probe_start[] = {0, 0};
        size_t probe_count[] = {1, num_pins};
        time_intern[0]=time;
        if(p.probes)
        {
        DG_RANK0 err = nc_put_vara_double( probe_grp_id, probe_timevarID,
                &probe_start[0], &count, &time_intern[0]);

        for( auto& record : feltor::probe_list)
        {
            record.function( resultD, var);
            dg::assign( resultD, resultH);
            dg::blas2::symv( probe_interpolate, resultH, simple_probes);
            DG_RANK0 err = nc_put_vara_double( probe_grp_id,
                    probe_id_field.at(record.name), probe_start, probe_count,
#ifdef WITH_MPI
                    simple_probes.data().data()
#else
                    simple_probes.data()
#endif
            );
        }
        }
         /// End probes output ///

        DG_RANK0 err = nc_close(ncid);
        DG_RANK0 std::cout << "First write successful!\n";
        ///////////////////////////////Timeloop/////////////////////////////////

        t.tic();
        bool abort = false;
        for( unsigned i=1; i<=maxout; i++)
        {
            dg::Timer ti;
            ti.tic();
            for( unsigned j=1; j<=p.itstp; j++)
            {
                try{
                    odeint->integrate( time, y0, t_output + j*deltaT, y0,
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


                if(p.probes)
                {
                for( auto& record : feltor::probe_list)
                {
                    record.function( resultD, var);
                    dg::assign( resultD, resultH);
                    dg::blas2::symv( probe_interpolate, resultH, simple_probes);
                    simple_probes_intern.at(record.name)[j-1]=simple_probes;
                    time_intern[j-1]=time;
                }
                }
                for( auto& m_list : equation_list)
                {
                    for( auto& record : m_list)
                    {
                        if( record.integral)
                        {
                            record.function( resultD, var);
                            dg::blas2::symv( projectD, resultD, transferD);
                            //toroidal average and add to time integral
                            dg::assign( transferD, transferH);
                            toroidal_average( transferH, transferH2d, false);
                            time_integrals.at(record.name+"_ta2d").add( time,
                                transferH2d);
                            // 2d data of plane varphi = 0
                            feltor::slice_vector3d( transferD, transferD2d,
                                local_size2d);
                            dg::assign( transferD2d, transferH2d);
                            time_integrals.at(record.name+"_2d").add( time,
                                transferH2d);
                        }
                    }
                }

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
            t_output += p.itstp*deltaT;
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
            DG_RANK0 std::cout << "\n\t Step: Time "<<time <<" of " << Tend;
            DG_RANK0 std::cout << "\n\t Average time for one inner loop: "
                        << var.duration/(double)p.itstp<<"s";

            ti.tic();
            //////////////////////////write fields////////////////////////
            start = i;
            DG_RANK0 err = nc_open(file_name.data(), NC_WRITE, &ncid);
            DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
            for( auto& record : feltor::diagnostics3d_list)
            {
                record.function( resultD, var);
                dg::blas2::symv( projectD, resultD, transferD);
                dg::assign( transferD, transferH);
                dg::file::put_vara_double( ncid, id4d.at(record.name), start,
                        g3d_out, transferH);
            }
            for( auto& record : feltor::restart3d_list)
            {
                record.function( resultD, var);
                dg::assign( resultD, resultH);
                dg::file::put_var_double( ncid, restart_ids.at(record.name),
                        grid, resultH);
            }
            for( auto& m_list : equation_list)
            {
            for( auto& record : m_list)
            {
                if(record.integral) // we already computed the output...
                {
                    std::string name = record.name+"_ta2d";
                    transferH2d = time_integrals.at(name).get_integral();
                    time_integrals.at(name).flush();
                    if(write2d) dg::file::put_vara_double( ncid, id3d.at(name),
                            start, *g2d_out_ptr, transferH2d);

                    name = record.name+"_2d";
                    transferH2d = time_integrals.at(name).get_integral( );
                    time_integrals.at(name).flush();
                    if(write2d) dg::file::put_vara_double( ncid, id3d.at(name),
                            start, *g2d_out_ptr, transferH2d);
                }
                else // compute from scratch
                {
                    record.function( resultD, var);
                    dg::blas2::symv( projectD, resultD, transferD);
                    std::string name = record.name + "_ta2d";
                    dg::assign( transferD, transferH);
                    toroidal_average( transferH, transferH2d, false);
                    if(write2d) dg::file::put_vara_double( ncid, id3d.at(name),
                            start, *g2d_out_ptr, transferH2d);

                    // 2d data of plane varphi = 0
                    name = record.name+"_2d";
                    feltor::slice_vector3d( transferD, transferD2d, local_size2d);
                    dg::assign( transferD2d, transferH2d);
                    if(write2d) dg::file::put_vara_double( ncid, id3d.at(name),
                            start, *g2d_out_ptr, transferH2d);
                }
            }
            }

            for( auto& record : feltor::diagnostics1d_list)
            {
                double result = record.function( var);
                DG_RANK0 nc_put_vara_double( ncid, id1d.at(record.name), &start, &count, &result);
            }
            //OUTPUT OF PROBES
            if(p.probes){
            for( unsigned j=0; j<p.itstp; j++)
            {
                probe_start[0] += 1;
                DG_RANK0 err = nc_put_vara_double( probe_grp_id,
                        probe_timevarID, &probe_start[0] , &probe_count[0],
                        &time_intern[j]);
                for( auto& record : feltor::probe_list)
                {
#ifdef WITH_MPI
                    DG_RANK0 err = nc_put_vara_double( probe_grp_id,
                            probe_id_field.at(record.name), probe_start,
                            probe_count,
                            simple_probes_intern.at(record.name)[j].data().data());
#else
                    DG_RANK0 err = nc_put_vara_double( probe_grp_id,
                            probe_id_field.at(record.name), probe_start,
                            probe_count,
                            simple_probes_intern.at(record.name)[j].data());
#endif
                }
            }
            }

            DG_RANK0 err = nc_close(ncid);
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
        DG_RANK0 std::cout <<"which is         \t"<<t.diff()/p.itstp/maxout<<"s/inner loop\n";
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
        dg::Average<dg::HVec> toroidal_average( grid, dg::coo3d::z, "simple");
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
                odeint->integrate( time, y0, t_output + i*deltaT, y0,
                         i<p.itstp ? dg::to::at_least :  dg::to::exact);
                std::cout << "Time "<<time<<" t_out "<<t_output<<" deltaT "<<deltaT<<" i "<<i<<" itstp "<<p.itstp<<"\n";

                double max_ue = dg::blas1::reduce(
                    feltor.velocity(0), 0., dg::AbsMax<double>() );
                std::cout << "\tMaximum ue "<<max_ue<<"\n";
                if( adaptive )
                {
                    std::cout << "\tdt "<<odeint->get_dt()<<"\n";
                    std::cout << "\tfailed "<<*var.nfailed<<"\n";
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
            t_output += p.itstp*deltaT;
            t.toc();
            std::cout << "\n\t Time  "<<time<<" of "<<Tend;
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
