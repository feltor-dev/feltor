#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>

#ifdef ASELA_MPI
#include <mpi.h>
#endif //ASELA_MPI

#ifndef WITHOUT_GLFW
#include "draw/host_window.h"
#endif // WITHOUT_GLFW

#include "dg/file/json_utilities.h"

#include "reconnection.h"
#include "init.h"
#include "diag.h"

#ifdef MPI_VERSION
#define DG_RANK0 if(rank==0)
#else
#define DG_RANK0
#endif


int main( int argc, char* argv[])
{
#ifdef ASELA_MPI
    ////////////////////////////////setup MPI///////////////////////////////
#ifdef _OPENMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    assert( provided >= MPI_THREAD_FUNNELED && "Threaded MPI lib required!\n");
#else
    MPI_Init(&argc, &argv);
#endif
    int periods[2] = {false, true}; //non-, periodic
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices==0){
        std::cerr << "No CUDA capable devices found"<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        return -1;
    }
    int device = rank % num_devices; //assume # of gpus/node is fixed
    std::cout << "# Rank "<<rank<<" computes with device "<<device<<" !"<<std::endl;
    cudaSetDevice( device);
#endif//THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int np[2];
    if(rank==0)
    {
        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads( );
#endif //omp
        std::cin>> np[0] >> np[1];
        std::cout << "# Computing with "
                  << np[0]<<" x "<<np[1]<<" processes x "
                  << num_threads<<" threads = "
                  <<size*num_threads<<" total"<<std::endl;
;
        if( size != np[0]*np[1])
        {
            std::cerr << "ERROR: Process partition needs to match total number of processes!"<<std::endl;
#ifdef ASELA_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#endif //ASELA_MPI
            return -1;
        }
    }
    MPI_Bcast( np, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);
#endif //ASELA_MPI

    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js;
    enum dg::file::error mode = dg::file::error::is_throw;
    if( argc == 1)
        dg::file::file2Json( "input/default.json", js, dg::file::comments::are_discarded);
    else
        dg::file::file2Json( argv[1], js);
    DG_RANK0 std::cout << js <<std::endl;

    const asela::Parameters p( js);

    //////////////////////////////////////////////////////////////////////////
    //Make grid

    dg::x::CartesianGrid2d grid( -p.lxhalf, p.lxhalf, -p.lyhalf, p.lyhalf , p.n, p.Nx, p.Ny, dg::DIR, dg::PER
        #ifdef ASELA_MPI
        , comm
        #endif //ASELA_MPI
        );
    DG_RANK0 std::cout << "Constructing Explicit...\n";
    asela::Asela<dg::x::CartesianGrid2d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec> asela( grid, p);
    DG_RANK0 std::cout << "Done!\n";

    /// //////////////////The initial field///////////////////////////////////////////
    double time = 0.;
    std::array<std::array<DVec,2>,2> y0;
    if( argc == 4)
    {
        try{
            y0 = feltor::initial_conditions.at(p.initne)( asela, grid, p );
        }catch ( std::out_of_range& error){
            MPI_OUT std::cerr << "Warning: initne parameter '"<<p.initne<<"' not recognized! Is there a spelling error? I assume you do not want to continue with the wrong initial condition so I exit! Bye Bye :)" << std::endl;
#ifdef ASELA_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#endif //ASELA_MPI
            return -1;
        }
    }
    DG_RANK0 std::cout << "Initialize time stepper..." << std::endl;
    std::string tableau = dg::file::get( mode, js, "timestepper", "tableau", "ImEx-BDF-3-3").asString();
    dg::ExplicitMultistep< std::array<std::array<dg::x::DVec,2>,2>> multistep( tableau, y0);
    unsigned step = 0;
    multistep.init( asela, time, y0, p.dt);
    DG_RANK0 std::cout << "Done!\n";


    /// ////////////Init diagnostics ////////////////////
    asela::Variables var = {asela, p, y0[0]};
    dg::Timer t;
    t.tic();

    DG_RANK0 std::cout << "Begin computation \n";
    DG_RANK0 std::cout << std::scientific << std::setprecision( 2);
    unsigned maxout = dg::file::get( mode, js, "output", "maxout", 100).asUInt();
    unsigned itstp = dg::file::get( mode, js, "output", "itstp", 5).asUInt();
    std::string output = dg::file::get( mode, js, "output", "type", "glfw").asString();
#ifndef WITHOUT_GLFW
    if( "glfw" == output)
    {
        /////////glfw initialisation ////////////////////////////////////////////
        dg::file::file2Json( "window_params.json", js, dg::file::comments::are_discarded);
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
        v2d["ne-1 / "] = &y0[0][0],  v4d["ni-1 / "] = &y0[0][1];
        v2d["Ue / "]   = &asela.velocity(0), v4d["Ui / "]   = &asela.velocity(1);
        v2d["Phi / "] = &asela.potential(0); v4d["Apar / "] = &asela.aparallel(0);
        v2d["Vor / "] = &asela.potential(0); v4d["j / "]    = &asela.aparallel(0);

        while ( !glfwWindowShouldClose( w ))
        {
            for( auto pair : v2d)
            {
                if( pair.first == "Vor / " || pair.first == "j / ")
                {
                    dg::blas2::symv( asela.laplacianM(), *pair.second, temp);
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
            for( unsigned i=0; i<itstp; i++)
            {
                try{ multistep.step( asela, time, y0);}
                catch( dg::Fail& fail) {
                    std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                    std::cerr << "Does Simulation respect CFL condition?\n";
                    glfwSetWindowShouldClose( w, GL_TRUE);
                    break;
                }
                step++;
            }
            ti.toc();
            std::cout << "\n\t Step "<<step;
            std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n";
        }
        glfwTerminate();
    }
#endif //WITHOUT_GLFW
    if( "netcdf" == output)
    {
        std::string inputfile = js.toStyledString(); //save input without comments, which is important if netcdf file is later read by another parser
        std::string outputfile;
        if( argc == 1 || argc == 2)
            outputfile = "reconnection.nc";
        else
            outputfile = argv[2];
        /// //////////////////////set up netcdf/////////////////////////////////////
        dg::file::NC_Error_Handle err;
        int ncid=-1;
        try{
            err = nc_create( outputfile.c_str(),NC_NETCDF4|NC_CLOBBER, &ncid);
        }catch( std::exception& e)
        {
            std::cerr << "ERROR creating file "<<outputfile<<std::endl;
            std::cerr << e.what()<<std::endl;
           return -1;
        }
        /// Set global attributes
        std::map<std::string, std::string> att;
        att["title"] = "Output file of feltor/src/reco2D/reconnection.cu";
        att["Conventions"] = "CF-1.7";
        ///Get local time and begin file history
        auto ttt = std::time(nullptr);
        auto tm = *std::localtime(&ttt);

        std::ostringstream oss;
        ///time string  + program-name + args
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        for( int i=0; i<argc; i++) oss << " "<<argv[i];
        att["history"] = oss.str();
        att["comment"] = "Find more info in feltor/src/reco2D/reconnection.tex";
        att["source"] = "FELTOR";
        att["references"] = "https://github.com/feltor-dev/feltor";
        att["inputfile"] = inputfile;
        for( auto pair : att)
            DG_RANK0 err = nc_put_att_text( ncid, NC_GLOBAL,
                pair.first.data(), pair.second.size(), pair.second.data());

        int dim_ids[3], tvarID;
        std::map<std::string, int> id1d, id3d;
        dg::x::CartesianGrid2d grid_out( -p.lxhalf, p.lxhalf, -p.lyhalf, p.lyhalf , p.n_out, p.Nx_out, p.Ny_out, dg::DIR, dg::PER
            #ifdef ASELA_MPI
            , comm
            #endif //ASELA_MPI
            );
        dg::x::IHMatrix projection = dg::create::interpolation( grid_out, grid);
        err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid_out,
                {"time", "y", "x"});

        //Create field IDs
        for( auto& record : asela::diagnostics2d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id3d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
                    &id3d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id3d.at(name), "long_name",
                    long_name.size(), long_name.data());
        }
        for( auto& record : shu::diagnostics2d_list)
        {
            std::string name = record.name + "_1d";
            std::string long_name = record.long_name + " (Volume integrated)";
            id1d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1, &dim_ids[0],
                &id1d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id1d.at(name), "long_name",
                    long_name.size(), long_name.data());
        }
        err = nc_enddef(ncid);
        size_t start = {0};
        size_t count = {1};
        ///////////////////////////////////first output/////////////////////////
        dg::x::DVec volume = dg::create::volume( grid);
        dg::x::DVec resultD = volume;
        dg::x::HVec resultH = dg::evaluate( dg::zero, grid);
        dg::x::HVec transferH = dg::evaluate( dg::zero, grid_out);
        for( auto& record : asela::diagnostics2d_list)
        {
            record.function( resultD, var);
            double result = dg::blas1::dot( volume, resultD);
            dg::assign( resultD, resultH);
            dg::blas2::gemv( projection, resultH, transferH);
            dg::file::put_vara_double( ncid, id3d.at(record.name), start,
                    grid_out, transferH);
            DG_RANK0 err = nc_put_vara_double( ncid, id1d.at(record.name),
                    &start, &count, &result);
        }
        DG_RANK0 err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        ///////////////////////////////////timeloop/////////////////////////
        unsigned step=0;
        for( unsigned i=1; i<=maxout; i++)
        {
            dg::Timer ti;
            ti.tic()
            for( unsigned j=0; j<itstp; j++)
            {
                try{ multistep.step( asela, time, y0);}
                catch( dg::Fail& fail) {
                    DG_RANK0 std::cerr << "ERROR failed to converge to "<<fail.epsilon()<<"\n";
                    DG_RANK0 std::cerr << "Does simulation respect CFL condition?"<<std::endl;
#ifdef ASELA_MPI
                    MPI_Abort(MPI_COMM_WORLD, -1);
#endif //ASELA_MPI
                    return -1;
                }
            }
            ti.toc();
            step+=itstp;
            DG_RANK0 std::cout << "\n\t Step "<<step <<" of "<<itstp*maxout <<" at time "<<time;
            DG_RANK0 std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)itstp<<"s\n\n"<<std::flush;
            //output all fields
            ti.tic();
            start = i;
            DG_RANK0 err = nc_open(file_name.data(), NC_WRITE, &ncid);
            DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
            for( auto& record : asela::diagnostics2d_list)
            {
                record.function( resultD, var);
                double result = dg::blas1::dot( volume, resultD);
                dg::assign( resultD, resultH);
                dg::blas2::gemv( projection, resultH, transferH);
                dg::file::put_vara_double( ncid, id3d.at(record.name), start, grid_out, transferH);
                DG_RANK0 err = nc_put_vara_double( ncid, id1d.at(record.name), &start, &count, &result);
            }
            DG_RANK0 err = nc_close( ncid);
            ti.toc();
            DG_RANK0 std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
        }
        DG_RANK0 err = nc_close(ncid);
    }
    if( !("netcdf" == output) && !("glfw" == output))
    {
        DG_RANK0 std::cerr <<"Error: Wrong value for output type "<<output<<" Must be glfw or netcdf! Exit now!";
#ifdef ASELA_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif //ASELA_MPI
        return -1;
    }
    ////////////////////////////////////////////////////////////////////
    t.toc();
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    DG_RANK0 std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    DG_RANK0 std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    DG_RANK0 std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";
#ifdef ASELA_MPI
    MPI_Finalize();
#endif //ASELA_MPI
    return 0;

}
