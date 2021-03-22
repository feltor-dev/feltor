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

#include "dg/file/file.h"
#include "feltor.h"
#include "implicit.h"
#include "init.h"
#include "feltordiag.h"
#include "init_from_file.h"

#ifdef WITH_MPI
//ATTENTION: in slurm should be used with --signal=SIGINT@30 (<signal>@<time in seconds>)
void sigterm_handler(int signal)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    std::cout << " pid "<<rank<<" sigterm_handler, got signal " << signal << std::endl;
    MPI_Finalize();
    exit(signal);
}
#endif //WITH_MPI

int main( int argc, char* argv[])
{
#ifdef WITH_MPI
    ////////////////////////////////setup MPI///////////////////////////////
    dg::mpi_init( argc, argv);
    MPI_Comm comm;
    dg::mpi_init3d( dg::DIR, dg::DIR, dg::PER, comm, std::cin, true);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    std::signal(SIGINT, sigterm_handler);
    std::signal(SIGTERM, sigterm_handler);
#endif //WITH_MPI
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Value js, gs;
    if( argc != 4 && argc != 5)
    {
        DG_RANK0 std::cerr << "ERROR: Wrong number of arguments!\nUsage: "
                << argv[0]<<" [input.json] [geometry.json] [output.nc]\n OR \n"
                << argv[0]<<" [input.json] [geometry.json] [output.nc] [initial.nc] "<<std::endl;
#ifdef WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
        return -1;
    }
    else
    {
        try{
            dg::file::file2Json( argv[1], js, dg::file::comments::are_discarded, dg::file::error::is_throw);
            feltor::Parameters( js, dg::file::error::is_throw);
        } catch( std::exception& e) {
            DG_RANK0 std::cerr << "ERROR in input parameter file "<<argv[1]<<std::endl;
            DG_RANK0 std::cerr << e.what()<<std::endl;
#ifdef WITH_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
            return -1;
        }
        try{
            dg::file::file2Json( argv[2], gs, dg::file::comments::are_discarded, dg::file::error::is_throw);
        } catch( std::exception& e) {
            DG_RANK0 std::cerr << "ERROR in geometry file "<<argv[2]<<std::endl;
            DG_RANK0 std::cerr << e.what()<<std::endl;
#ifdef WITH_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
            return -1;
        }
    }
    const feltor::Parameters p( js);
    DG_RANK0 std::cout << js <<  std::endl;
    std::string inputfile = js.toStyledString(), geomfile = gs.toStyledString();
    DG_RANK0 std::cout << geomfile << std::endl;
    dg::geo::TokamakMagneticField mag, mod_mag;
    dg::geo::CylindricalFunctor wall, transition, sheath, direction;
    try{
        mag = dg::geo::createMagneticField(gs, dg::file::error::is_throw);
        mod_mag = dg::geo::createModifiedField(gs, js, dg::file::error::is_throw, wall, transition);
    }catch(std::runtime_error& e)
    {
        std::cerr << "ERROR in geometry file "<<argv[2]<<std::endl;
        std::cerr <<e.what()<<std::endl;
#ifdef WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
        return -1;
    }
#ifdef WITH_MPI
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    if( dims[2] >= (int)p.Nz)
    {
        DG_RANK0 std::cerr << "ERROR: Number of processes in z "<<dims[2]<<" may not be larger or equal Nz "<<p.Nz<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        return -1;
    }
#endif //WITH_MPI
    ////////////////////////////////set up computations///////////////////////////
    //Make grids
    double Rmin=mag.R0()-p.boxscaleRm*mag.params().a();
    double Zmin=-p.boxscaleZm*mag.params().a()*mag.params().elongation();
    double Rmax=mag.R0()+p.boxscaleRp*mag.params().a();
    double Zmax=p.boxscaleZp*mag.params().a()*mag.params().elongation();
    dg::x::CylindricalGrid3d grid( Rmin, Rmax, Zmin, Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.symmetric ? 1 : p.Nz, p.bcxN, p.bcyN, dg::PER
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
        );
    dg::x::CylindricalGrid3d g3d_out( Rmin, Rmax, Zmin, Zmax, 0, 2.*M_PI,
        p.n_out, p.Nx_out, p.Ny_out, p.symmetric ? 1 : p.Nz_out, p.bcxN, p.bcyN, dg::PER
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
        );
    std::unique_ptr<typename dg::x::CylindricalGrid3d::perpendicular_grid> g2d_out_ptr  ( dynamic_cast<typename dg::x::CylindricalGrid3d::perpendicular_grid*>( g3d_out.perp_grid()));
#ifdef WITH_MPI
    unsigned local_size2d = g2d_out_ptr->local().size();
#else
    unsigned local_size2d = g2d_out_ptr->size();
#endif

    try{
        dg::geo::createSheathRegion( js, dg::file::error::is_throw, mag, wall,
                Rmin, Rmax, Zmin, Zmax, sheath, direction);
    }catch(std::runtime_error& e)
    {
        DG_RANK0 std::cerr << "ERROR in geometry file "<<geomfile<<std::endl;
        DG_RANK0 std::cerr <<e.what()<<std::endl;
        return -1;
    }
    if( p.periodify)
        mag = dg::geo::periodify( mag, Rmin, Rmax, Zmin, Zmax, dg::NEU, dg::NEU);

    //create RHS
    DG_RANK0 std::cout << "Constructing Explicit...\n";
    feltor::Explicit< dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec> feltor( grid, p, mag);
    //DG_RANK0 std::cout << "Constructing Implicit...\n";
    //feltor::Implicit< dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec> implicit( grid, p, mag);
    DG_RANK0 std::cout << "Done!\n";

    // helper variables for output computations
    std::map<std::string, dg::Simpsons<dg::x::HVec>> time_integrals;
    dg::Average<dg::x::HVec> toroidal_average( g3d_out, dg::coo3d::z, "simple");
    dg::MultiMatrix<dg::x::HMatrix,dg::x::HVec> projectH = dg::create::fast_projection( grid, 1, p.cx, p.cy, dg::normed);
    dg::MultiMatrix<dg::x::DMatrix,dg::x::DVec> projectD = dg::create::fast_projection( grid, 1, p.cx, p.cy, dg::normed);
    dg::x::HVec transferH( dg::evaluate(dg::zero, g3d_out));
    dg::x::DVec transferD( dg::evaluate(dg::zero, g3d_out));
    dg::x::HVec transferH2d = dg::evaluate( dg::zero, *g2d_out_ptr);
    dg::x::DVec transferD2d = dg::evaluate( dg::zero, *g2d_out_ptr);
    dg::x::HVec resultH = dg::evaluate( dg::zero, grid);
    dg::x::DVec resultD = dg::evaluate( dg::zero, grid);

    std::array<dg::x::DVec, 3> gradPsip;
    gradPsip[0] =  dg::evaluate( mag.psipR(), grid);
    gradPsip[1] =  dg::evaluate( mag.psipZ(), grid);
    gradPsip[2] =  resultD; //zero
    dg::x::DVec hoo = dg::pullback( dg::geo::Hoo( mag), grid);
    feltor::Variables var = {
        feltor, p, mag, gradPsip, gradPsip, hoo
    };
    // the vector ids
    std::map<std::string, int> id3d, id4d, restart_ids;

    double dEdt = 0, accuracy = 0;
    double E0 = 0.;

    /// //////////////////The initial field///////////////////////////////////////////
    double time = 0.;
    std::array<std::array<dg::x::DVec,2>,2> y0;
    if( argc == 4)
    {
        try{
            y0 = feltor::initial_conditions.at(p.initne)( feltor, grid, p,mag );
        }catch ( std::out_of_range& error){
            DG_RANK0 std::cerr << "Warning: initne parameter '"<<p.initne<<"' not recognized! Is there a spelling error? I assume you do not want to continue with the wrong initial condition so I exit! Bye Bye :)" << std::endl;
#ifdef WITH_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
            return -1;
        }
    }
    if( argc == 5)
    {
        try{
            y0 = feltor::init_from_file(argv[4], grid, p,time);
        }catch (std::exception& e){
            DG_RANK0 std::cerr << "ERROR occured initializing from file "<<argv[4]<<std::endl;
            DG_RANK0 std::cerr << e.what()<<std::endl;
#ifdef WITH_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
            return -1;
        }
    }

    try{
        bool fixed_profile;
        dg::x::HVec profile, source_profile;
        source_profile = feltor::source_profiles.at(p.source_type)(
            fixed_profile, profile, grid, p, mag);
        feltor.set_source( fixed_profile, dg::construct<dg::x::DVec>(profile),
            p.source_rate, dg::construct<dg::x::DVec>(source_profile)
        );
    }catch ( std::out_of_range& error){
        std::cerr << "Warning: source_type parameter '"<<p.source_type<<"' not recognized! Is there a spelling error? I assume you do not want to continue with the wrong source so I exit! Bye Bye :)"<<std::endl;
#ifdef WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
        return -1;
    }

    /// //////////////////////////set up netcdf/////////////////////////////////////
    dg::file::NC_Error_Handle err;
    std::string file_name = argv[3];
    int ncid=-1;
    try{
        DG_RANK0 err = nc_create( file_name.data(), NC_NETCDF4|NC_CLOBBER, &ncid);
    }catch( std::exception& e)
    {
        std::cerr << "ERROR creating file "<<file_name<<std::endl;
        std::cerr << e.what()<<std::endl;
#ifdef WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
       return -1;
    }
    /// Set global attributes
    std::map<std::string, std::string> att;
    att["title"] = "Output file of feltor/src/feltor/feltor_hpc.cu";
    att["Conventions"] = "CF-1.7";
    ///Get local time and begin file history
    auto ttt = std::time(nullptr);
    auto tm = *std::localtime(&ttt);

    std::ostringstream oss;
    ///time string  + program-name + args
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    for( int i=0; i<argc; i++) oss << " "<<argv[i];
    att["history"] = oss.str();
    att["comment"] = "Find more info in feltor/src/feltor/feltor.tex";
    att["source"] = "FELTOR";
    att["references"] = "https://github.com/feltor-dev/feltor";
    att["inputfile"] = inputfile;
    att["geomfile"] = geomfile;
    for( auto pair : att)
        DG_RANK0 err = nc_put_att_text( ncid, NC_GLOBAL,
            pair.first.data(), pair.second.size(), pair.second.data());

    // Define dimensions (t,z,y,x)
    int dim_ids[4], restart_dim_ids[3], tvarID;
    DG_RANK0 err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, g3d_out, {"time", "z", "y", "x"});
    DG_RANK0 err = dg::file::define_dimensions( ncid, restart_dim_ids, grid, {"zr", "yr", "xr"});
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
        DG_RANK0 err = nc_enddef( ncid);
        DG_RANK0 std::cout << "Computing "<<record.name<<"\n";
        record.function( transferH, var, g3d_out);
        //record.function( resultH, var, grid);
        //dg::blas2::symv( projectH, resultH, transferH);
        dg::file::put_var_double( ncid, vecID, g3d_out, transferH);
        DG_RANK0 err = nc_redef(ncid);
    }
    //create & output static 2d variables into file
    for ( auto& record : feltor::diagnostics2d_static_list)
    {
        int vecID;
        DG_RANK0 err = nc_def_var( ncid, record.name.data(), NC_DOUBLE, 2,
            &dim_ids[2], &vecID);
        DG_RANK0 err = nc_put_att_text( ncid, vecID,
            "long_name", record.long_name.size(), record.long_name.data());
        DG_RANK0 err = nc_enddef( ncid);
        DG_RANK0 std::cout << "Computing2d "<<record.name<<"\n";
        //record.function( transferH, var, g3d_out); //ATTENTION: This does not work because feltor internal varialbes return full grid functions
        record.function( resultH, var, grid);
        dg::blas2::symv( projectH, resultH, transferH);
        if(write2d)dg::file::put_var_double( ncid, vecID, *g2d_out_ptr, transferH);
        DG_RANK0 err = nc_redef(ncid);
    }

    //Create field IDs
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
        DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, restart_dim_ids,
            &restart_ids.at(name));
        DG_RANK0 err = nc_put_att_text( ncid, restart_ids.at(name), "long_name", long_name.size(),
            long_name.data());
    }
    for( auto& record : feltor::diagnostics2d_list)
    {
        std::string name = record.name + "_ta2d";
        std::string long_name = record.long_name + " (Toroidal average)";
        id3d[name] = 0;//creates a new id3d entry for all processes
        DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids3d,
            &id3d.at(name));
        DG_RANK0 err = nc_put_att_text( ncid, id3d.at(name), "long_name", long_name.size(),
            long_name.data());

        name = record.name + "_2d";
        long_name = record.long_name + " (Evaluated on phi = 0 plane)";
        id3d[name] = 0;
        DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids3d,
            &id3d.at(name));
        DG_RANK0 err = nc_put_att_text( ncid, id3d.at(name), "long_name", long_name.size(),
            long_name.data());
    }
    DG_RANK0 err = nc_enddef(ncid);
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
#ifdef WITH_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
            return -1;
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
    for( auto& record : feltor::diagnostics2d_list)
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

        // and a slice
        name = record.name + "_2d";
        feltor::slice_vector3d( transferD, transferD2d, local_size2d);
        dg::assign( transferD2d, transferH2d);
        if( record.integral) time_integrals[name].init( time, transferH2d);
        if(write2d) dg::file::put_vara_double( ncid, id3d.at(name), start, *g2d_out_ptr, transferH2d);
        tti.toc();
        DG_RANK0 std::cout<< name << " 2d output took "<<tti.diff()<<"\n";
    }
    DG_RANK0 err = nc_close(ncid);
    DG_RANK0 std::cout << "First write successful!\n";
    ///////////////////////////////////////Timeloop/////////////////////////////////
    //dg::Karniadakis< std::array<std::array<dg::x::DVec,2>,2 >,
    //    feltor::FeltorSpecialSolver<
    //        dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec>
    //    > karniadakis( grid, p, mag);
    dg::ExplicitMultistep< std::array<std::array<dg::x::DVec,2>,2 > > mp( "TVB-3-3", y0);
    {
        dg::x::HVec h_wall = dg::pullback( wall, grid);
        dg::x::HVec h_sheath = dg::pullback( sheath, grid);
        dg::x::HVec h_velocity = dg::pullback( direction, grid);
    feltor.set_wall_and_sheath( p.wall_rate, dg::construct<dg::x::DVec>( h_wall), p.sheath_rate, dg::construct<dg::x::DVec>(h_sheath), dg::construct<dg::x::DVec>(h_velocity));
    //implicit.set_wall_and_sheath( p.wall_rate, dg::construct<dg::x::DVec>( h_wall), p.sheath_rate, dg::construct<dg::x::DVec>(h_sheath));
    //karniadakis.solver().set_wall_and_sheath( p.wall_rate, dg::construct<dg::x::DVec>( h_wall), p.sheath_rate, dg::construct<dg::x::DVec>(h_sheath));
    }

    DG_RANK0 std::cout << "Initialize Timestepper" << std::endl;
    //karniadakis.init( feltor, implicit, time, y0, p.dt);
    mp.init( feltor, time, y0, p.dt);
    dg::Timer t;
    t.tic();
    unsigned step = 0;
    for( unsigned i=1; i<=p.maxout; i++)
    {

        dg::Timer ti;
        ti.tic();
        for( unsigned j=0; j<p.itstp; j++)
        {
            double previous_time = time;
            for( unsigned k=0; k<p.inner_loop; k++)
            {
                try{
                    //karniadakis.step( feltor, implicit, time, y0);
                    mp.step( feltor, time, y0);
                }
                catch( dg::Fail& fail){
                    DG_RANK0 std::cerr << "ERROR failed to converge to "<<fail.epsilon()<<"\n";
                    DG_RANK0 std::cerr << "Does simulation respect CFL condition?"<<std::endl;
#ifdef WITH_MPI
                    MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
                    return -1;
                }
                catch( std::exception& fail) {
                    DG_RANK0 std::cerr << "ERROR in timestepper\n";
                    DG_RANK0 std::cerr << fail.what()<<std::endl;
#ifdef WITH_MPI
                    MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
                    return -1;
                }
                step++;
            }
            dg::Timer tti;
            tti.tic();
            double deltat = time - previous_time;
            double energy = 0, ediff = 0.;
            for( auto& record : feltor::diagnostics2d_list)
            {
                if( std::find( feltor::energies.begin(), feltor::energies.end(), record.name) != feltor::energies.end())
                {
                    record.function( resultD, var);
                    energy += dg::blas1::dot( resultD, feltor.vol3d());
                }
                if( record.integral)
                {
                    record.function( resultD, var);
                    dg::blas2::symv( projectD, resultD, transferD);
                    //toroidal average and add to time integral
                    dg::assign( transferD, transferH);
                    toroidal_average( transferH, transferH2d, false);
                    time_integrals.at(record.name+"_ta2d").add( time, transferH2d);

                    // 2d data of plane varphi = 0
                    feltor::slice_vector3d( transferD, transferD2d, local_size2d);
                    dg::assign( transferD2d, transferH2d);
                    time_integrals.at(record.name+"_2d").add( time, transferH2d);
                    if( std::find( feltor::energy_diff.begin(), feltor::energy_diff.end(), record.name) != feltor::energy_diff.end())
                        ediff += dg::blas1::dot( resultD, feltor.vol3d());
                }

            }

            dEdt = (energy - E0)/deltat;
            E0 = energy;
            accuracy  = 2.*fabs( (dEdt - ediff)/( dEdt + ediff));

            DG_RANK0 std::cout << "\tTime "<<time<<"\n";
            DG_RANK0 std::cout <<"\td E/dt = " << dEdt
                      <<" Lambda = " << ediff
                      <<" -> Accuracy: " << accuracy << "\n";
            double max_ue = dg::blas1::reduce(
                feltor.velocity(0), 0., dg::AbsMax<double>() );
            DG_RANK0 std::cout << "\tMaximum ue "<<max_ue<<"\n";
            //----------------Test if induction equation holds
            if( p.beta != 0)
            {
                dg::blas1::pointwiseDot(
                    feltor.density(0), feltor.velocity(0), resultD);
                dg::blas1::pointwiseDot( p.beta,
                    feltor.density(1), feltor.velocity(1), -p.beta, resultD);
                double norm  = dg::blas2::dot( resultD, feltor.vol3d(), resultD);
                dg::blas1::axpby( -1., feltor.lapMperpA(), 1., resultD);
                double error = dg::blas2::dot( resultD, feltor.vol3d(), resultD);
                DG_RANK0 std::cout << "\tRel. Error Induction "<<sqrt(error/norm) <<"\n";
            }
            tti.toc();
            DG_RANK0 std::cout << " Time for internal diagnostics "<<tti.diff()<<"s\n";
        }
        ti.toc();
        DG_RANK0 std::cout << "\n\t Step "<<step <<" of "
                    << p.inner_loop*p.itstp*p.maxout << " at time "<<time;
        DG_RANK0 std::cout << "\n\t Average time for one step: "
                    << ti.diff()/(double)p.itstp/(double)p.inner_loop<<"s";
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
            dg::file::put_vara_double( ncid, id4d.at(record.name), start, g3d_out, transferH);
        }
        for( auto& record : feltor::restart3d_list)
        {
            record.function( resultD, var);
            dg::assign( resultD, resultH);
            dg::file::put_var_double( ncid, restart_ids.at(record.name), grid, resultH);
        }
        for( auto& record : feltor::diagnostics2d_list)
        {
            if(record.integral) // we already computed the output...
            {
                std::string name = record.name+"_ta2d";
                transferH2d = time_integrals.at(name).get_integral();
                time_integrals.at(name).flush();
                if(write2d) dg::file::put_vara_double( ncid, id3d.at(name), start, *g2d_out_ptr, transferH2d);

                name = record.name+"_2d";
                transferH2d = time_integrals.at(name).get_integral( );
                time_integrals.at(name).flush( );
                if(write2d) dg::file::put_vara_double( ncid, id3d.at(name), start, *g2d_out_ptr, transferH2d);
            }
            else // compute from scratch
            {
                record.function( resultD, var);
                dg::blas2::symv( projectD, resultD, transferD);

                std::string name = record.name+"_ta2d";
                dg::assign( transferD, transferH);
                toroidal_average( transferH, transferH2d, false);
                if(write2d) dg::file::put_vara_double( ncid, id3d.at(name), start, *g2d_out_ptr, transferH2d);

                // 2d data of plane varphi = 0
                name = record.name+"_2d";
                feltor::slice_vector3d( transferD, transferD2d, local_size2d);
                dg::assign( transferD2d, transferH2d);
                if(write2d) dg::file::put_vara_double( ncid, id3d.at(name), start, *g2d_out_ptr, transferH2d);
            }
        }
        DG_RANK0 err = nc_close(ncid);
        ti.toc();
        DG_RANK0 std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
    }
    t.toc();
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    DG_RANK0 std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    DG_RANK0 std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    DG_RANK0 std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout/p.inner_loop<<"s/step\n";
#ifdef WITH_MPI
    MPI_Finalize();
#endif //WITH_MPI

    return 0;

}
