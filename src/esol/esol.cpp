#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#ifdef WITH_MPI
#include <mpi.h>
#endif //WITH_MPI

#ifndef WITHOUT_GLFW
#include "draw/host_window.h"
#endif // WITHOUT_GLFW
#include "dg/file/file.h"

#include "esol.h"
#include "init.h"
#include "init_from_file.h"
#include "diag.h"


int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    std::stringstream title;

    dg::file::WrappedJsonValue ws = dg::file::file2Json( argc == 1 ? "input/default.json" : argv[1]);

    const esol::Parameters p( ws);
    
#ifdef WITH_MPI
    ////////////////////////////////setup MPI///////////////////////////////
    dg::mpi_init( &argc, &argv);
    MPI_Comm comm;
    dg::mpi_init2d( p.bc_x, p.bc_y, comm, std::cin, true);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI

    DG_RANK0 std::cout << ws.toStyledString() <<std::endl;

    ///////MAKE GRID///////////////////////////////////////////////
    dg::x::CartesianGrid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
    );
    ///////MAKE MODEL///////////////////////////////////////////////
    DG_RANK0 std::cout << "Constructing Esol...\n";
    esol::Esol<dg::x::CartesianGrid2d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec> esol( grid, p);
    DG_RANK0 std::cout << "Done!\n";

    //////////////////create initial fields///////////////////////////////////////
    double time = 0.;
    std::array<dg::x::DVec,2> y0;
    if( argc == 4 )
    {
        try{
            y0 = esol::init_from_file(argv[3], grid, p, time);
        }catch (std::exception& error){
            DG_RANK0 std::cerr << "ERROR occured initializing from file "<<argv[3]<<std::endl;
            DG_RANK0 std::cerr << error.what()<<std::endl;
#ifdef WITH_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
            return -1;
        }
    }
    else {
        try{
            y0 = esol::initial_conditions(esol, grid, p, ws );
        }catch ( std::exception& error){
            DG_RANK0 std::cerr << "Error in input file\n ";
            DG_RANK0 std::cerr << error.what();
#ifdef WITH_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
            return -1;
        }
    }
    
    DG_RANK0 std::cout << "Initialize time stepper..." << std::endl;
    dg::ExplicitMultistep<std::array<dg::x::DVec, 2>> multistep;
    dg::Adaptive< dg::ERKStep< std::array<dg::x::DVec,2>>> adapt;
    double rtol = 0., atol = 0., dt = 0.;
    unsigned step = 0;
    double dt_out = p.dt*p.itstp;
    double t_out = time + dt_out;
    if( p.timestepper == "multistep")
    {
        std::string tableau = ws[ "timestepper"]["tableau"].asString("TVB-3-3");
        multistep.construct( tableau, y0);
        dt = p.dt;
        multistep.init( esol, time, y0, dt);
    }
    else if (p.timestepper == "adaptive")
    {
        std::string tableau = ws[ "timestepper"]["tableau"].asString( "Bogacki-Shampine-4-2-3");
        adapt.construct( tableau, y0);
        rtol = ws[ "timestepper"][ "rtol"].asDouble( 1e-7);
        atol = ws[ "timestepper"][ "atol"].asDouble( 1e-10);
        dt = 1e-6; //that should be a small enough initial guess
    }
    else
    {
        DG_RANK0 std::cerr<<"Error: Unrecognized timestepper: '"<<p.timestepper<<"'! Exit now!";
#ifdef WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
        return -1;
    }
    DG_RANK0 std::cout << "Done!\n";

    /// ////////////Init diagnostics ////////////////////
    esol::Variables var = {esol, p, y0, 0.0};
    dg::Timer t;
    t.tic();
    {
        std::array<dg::x::DVec,2>y1 = y0;
        esol( 0., y0, y1);
    }
    t.toc();
    var.duration = t.diff();
    t.tic();


    DG_RANK0 std::cout << "Begin computation \n";
    DG_RANK0 std::cout << std::scientific << std::setprecision( 2);
#ifndef WITHOUT_GLFW
    if( "glfw" == p.output)
    {
        /////////glfw initialisation ////////////////////////////////////////////
        dg::file::WrappedJsonValue js = dg::file::file2Json( "window_params.json");
        GLFWwindow* w = draw::glfwInitAndCreateWindow( js["width"].asDouble(), js["height"].asDouble(), "");
        draw::RenderHostData render(js["rows"].asDouble(), js["cols"].asDouble());
        //create visualisation vectors
        dg::DVec visual( grid.size()), temp(visual);
        dg::HVec hvisual( grid.size());
        //transform vector to an equidistant grid
        std::stringstream title;
        draw::ColorMapRedBlueExtMinMax colors( -1.0, 1.0);
        dg::IDMatrix equidistant = dg::create::backscatter( grid );
        // the things to plot:
        std::map<std::string, const dg::DVec* > v2d;
        v2d["ne / "] = &esol.density(0);
        v2d["Ni / "] = &esol.density(1);
        v2d["Phi / "] = &esol.potential(0);
        v2d["Vor / "] = &esol.potential(0);

        while ( !glfwWindowShouldClose( w ))
        {
            for( auto pair : v2d)
            {
                if( pair.first == "Vor / " )
                {
                    esol.compute_vorticity( 1., *pair.second, 0., temp);
                    dg::blas2::gemv( equidistant, temp, visual);
                }
                else
                    dg::blas2::gemv( equidistant, *pair.second, visual);
                dg::assign( visual, hvisual);
                colors.scalemax() = dg::blas1::reduce(
                    hvisual, 0., dg::AbsMax<double>() );
                colors.scalemin() = -colors.scalemax();
                title << std::setprecision(2) << std::scientific;
                title <<pair.first << colors.scalemax()<<"   ";
                render.renderQuad( hvisual, grid.n()*grid.Nx(),
                        grid.n()*grid.Ny(), colors);
            }
            title << std::fixed;
            title << " &&   time = "<<time;
            glfwSetWindowTitle(w,title.str().c_str());
            title.str("");
            glfwPollEvents();
            glfwSwapBuffers( w);

            //step
            dg::Timer ti;
            ti.tic();
            while( time < t_out )
            {
                if( time+dt > t_out)
                    dt = t_out-time;
                try{
                    if( p.timestepper == "adaptive")
                        adapt.step( esol, time, y0, time, y0, dt, dg::pid_control, dg::l2norm, rtol, atol);
                    if( p.timestepper == "multistep")
                        multistep.step( esol, time, y0);
                }
                catch( dg::Fail& fail) {
                    DG_RANK0 std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                    DG_RANK0 std::cerr << "Does Simulation respect CFL condition?\n";
                    glfwSetWindowShouldClose( w, GL_TRUE);
                    break;
                }
                step++;
            }
            t_out += dt_out;
            ti.toc();
            DG_RANK0 std::cout << "\n\t Step "<<step;
            DG_RANK0 std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n";
        }
        glfwTerminate();
    }
#endif //WITHOT_GLFW    
    if( "netcdf" == p.output)
    {
        std::string inputfile = ws.toStyledString(); //save input without comments, which is important if netcdf file is later read by another parser
        std::string outputfile;
        if( argc==1 || argc == 2 )
            outputfile = "esol.nc";
        else
            outputfile = argv[2];
        /// //////////////////////set up netcdf/////////////////////////////////////
        dg::file::NC_Error_Handle err;
        int ncid=-1;
        try{
            DG_RANK0 err = nc_create( outputfile.c_str(),NC_NETCDF4|NC_CLOBBER, &ncid);
        }catch( std::exception& e)
        {
            std::cerr << "ERROR creating file "<<outputfile<<std::endl;
            std::cerr << e.what()<<std::endl;
           return -1;
        }
        /// Set global attributes
        std::map<std::string, std::string> att;
        att["title"] = "Output file of feltor/src/esol/esol.cu";
        att["Conventions"] = "CF-1.7";
        att["history"] = dg::file::timestamp( argc, argv);
        att["comment"] = "Find more info in feltor/src/esol/esol.tex";
        att["source"] = "FELTOR";
        att["references"] = "https://github.com/feltor-dev/feltor";
        att["inputfile"] = inputfile;
        for( auto pair : att)
            DG_RANK0 err = nc_put_att_text( ncid, NC_GLOBAL,
                pair.first.data(), pair.second.size(), pair.second.data());

        int dim_ids[3], restart_dim_ids[2], tvarID;
        std::map<std::string, int> id1d, id3d, restart_ids;
        dg::x::CartesianGrid2d grid_out(  0, p.lx, 0, p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y
            #ifdef WITH_MPI
            , comm
            #endif //WITH_MPI
            );
        dg::x::IHMatrix projection = dg::create::interpolation( grid_out, grid);
        DG_RANK0 err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid_out,
                {"time", "y", "x"});
        DG_RANK0 err = dg::file::define_dimensions( ncid, restart_dim_ids, grid,
                {"yr", "xr"});
        //Create field IDs
        for( auto& record : esol::diagnostics2d_list)
        {
            DG_RANK0 std::cout << record.name << std::endl;
            std::string name = record.name;
            std::string long_name = record.long_name;
            id3d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
                    &id3d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id3d.at(name), "long_name",
                    long_name.size(), long_name.data());
        }
        for( auto& record : esol::diagnostics2d_list)
        {
            std::string name = record.name + "_1d";
            std::string long_name = record.long_name + " (Volume integrated)";
            id1d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1, &dim_ids[0],
                &id1d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id1d.at(name), "long_name",
                    long_name.size(), long_name.data());
        }
        for( auto& record : esol::restart2d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            restart_ids[name] = 0;//creates a new entry for all processes
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 2, restart_dim_ids,
                &restart_ids.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, restart_ids.at(name), "long_name", long_name.size(),
                long_name.data());
        }
        for( auto& record : esol::diagnostics1d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id1d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1, &dim_ids[0],
                &id1d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id1d.at(name), "long_name", long_name.size(),
                long_name.data());
        }
        dg::x::DVec volume = dg::create::volume( grid);
        dg::x::DVec resultD = volume;
        dg::x::HVec resultH = dg::evaluate( dg::zero, grid);
        dg::x::HVec transferH = dg::evaluate( dg::zero, grid_out);
        for( auto& record : esol::diagnostics2d_static_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            int staticID = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 2, &dim_ids[1],
                &staticID);
            DG_RANK0 err = nc_put_att_text( ncid, staticID, "long_name", long_name.size(),
                long_name.data());
            DG_RANK0 err = nc_enddef(ncid);
            record.function( resultD, var);
            dg::assign( resultD, resultH);
            dg::blas2::gemv( projection, resultH, transferH);
            dg::file::put_var_double( ncid, staticID, grid_out, transferH);
            DG_RANK0 err = nc_redef(ncid);
        }
        DG_RANK0 err = nc_enddef(ncid);
        size_t start = {0};
        size_t count = {1};
        ///////////////////////////////////first output/////////////////////////
        for( auto& record : esol::diagnostics2d_list)
        {
            record.function( resultD, var);
            double result = dg::blas1::dot( volume, resultD);
            dg::assign( resultD, resultH);
            dg::blas2::gemv( projection, resultH, transferH);
            dg::file::put_vara_double( ncid, id3d.at(record.name), start,
                    grid_out, transferH);
            DG_RANK0 err = nc_put_vara_double( ncid, id1d.at(record.name+"_1d"),
                    &start, &count, &result);
        }
        for( auto& record : esol::diagnostics1d_list)
        {
            double result = record.function( var);
            DG_RANK0 err = nc_put_vara_double( ncid, id1d.at(record.name), &start, &count, &result);
        }
        DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
        DG_RANK0 err = nc_close( ncid);
        ///////////////////////////////////timeloop/////////////////////////
        for( unsigned i=1; i<=p.maxout; i++)
        {
            dg::Timer ti;
            ti.tic();
            while( time < t_out )
            {
                if( time+dt > t_out)
                    dt = t_out-time;
                try{
                    if( p.timestepper == "adaptive")
                        adapt.step( esol, time, y0, time, y0, dt, dg::pid_control, dg::l2norm, rtol, atol);
                    if( p.timestepper == "multistep")
                        multistep.step( esol, time, y0);
                }
                catch( dg::Fail& fail) {
                    DG_RANK0 std::cerr << "ERROR failed to converge to "<<fail.epsilon()<<"\n";
                    DG_RANK0 std::cerr << "Does simulation respect CFL condition?"<<std::endl;
#ifdef WITH_MPI
                    MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
                    return -1;
                }
            }
            t_out += dt_out;
            ti.toc();
            var.duration = ti.diff() / (double) p.itstp;
            step+=p.itstp;
            DG_RANK0 std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time << " with current timestep "<<dt;
            DG_RANK0 std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
            //output all fields
            ti.tic();
            start = i;
            DG_RANK0 err = nc_open(outputfile.data(), NC_WRITE, &ncid);
            DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
            for( auto& record : esol::diagnostics2d_list)
            {
                record.function( resultD, var);
                double result = dg::blas1::dot( volume, resultD);
                dg::assign( resultD, resultH);
                dg::blas2::gemv( projection, resultH, transferH);
                dg::file::put_vara_double( ncid, id3d.at(record.name), start, grid_out, transferH);
                DG_RANK0 err = nc_put_vara_double( ncid, id1d.at(record.name+"_1d"), &start, &count, &result);
            }
            for( auto& record : esol::restart2d_list)
            {
                record.function( resultD, var);
                dg::assign( resultD, resultH);
                dg::file::put_var_double( ncid, restart_ids.at(record.name), grid, resultH);
            }
            for( auto& record : esol::diagnostics1d_list)
            {
                double result = record.function( var);
                DG_RANK0 err = nc_put_vara_double( ncid, id1d.at(record.name), &start, &count, &result);
            }
            DG_RANK0 err = nc_close( ncid);
            ti.toc();
            DG_RANK0 std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
        }
    }
    if( !("netcdf" == p.output) && !("glfw" == p.output))
    {
        DG_RANK0 std::cerr <<"Error: Wrong value for output type "<<p.output<<" Must be glfw or netcdf! Exit now!";
#ifdef WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
        return -1;
    }
    t.toc();
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    DG_RANK0 std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    DG_RANK0 std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    DG_RANK0 std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";
#ifdef WITH_MPI
    MPI_Finalize();
#endif //WITH_MPI
    return 0;

}
