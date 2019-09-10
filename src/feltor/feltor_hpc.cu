#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <sstream>
#include <cmath>
#include <csignal>

#ifdef FELTOR_MPI
#include <mpi.h>
#endif //FELTOR_MPI

#include "dg/file/nc_utilities.h"
#include "feltor.h"
#include "implicit.h"

#ifdef FELTOR_MPI
using HVec = dg::MHVec;
using DVec = dg::MDVec;
using HMatrix = dg::MHMatrix;
using DMatrix = dg::MDMatrix;
using IDMatrix = dg::MIDMatrix;
using IHMatrix = dg::MIHMatrix;
using Geometry = dg::CylindricalMPIGrid3d;
#define MPI_OUT if(rank==0)
#else //FELTOR_MPI
using HVec = dg::HVec;
using DVec = dg::DVec;
using HMatrix = dg::HMatrix;
using DMatrix = dg::DMatrix;
using IDMatrix = dg::IDMatrix;
using IHMatrix = dg::IHMatrix;
using Geometry = dg::CylindricalGrid3d;
#define MPI_OUT
#endif //FELTOR_MPI

#include "init.h"
#include "feltordiag.h"

#ifdef FELTOR_MPI
//ATTENTION: in slurm should be used with --signal=SIGINT@30 (<signal>@<time in seconds>)
void sigterm_handler(int signal)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    std::cout << " pid "<<rank<<" sigterm_handler, got signal " << signal << std::endl;
    MPI_Finalize();
    exit(signal);
}
#endif //FELTOR_MPI

int main( int argc, char* argv[])
{
#ifdef FELTOR_MPI
    ////////////////////////////////setup MPI///////////////////////////////
#ifdef _OPENMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    assert( provided >= MPI_THREAD_FUNNELED && "Threaded MPI lib required!\n");
#else
    MPI_Init(&argc, &argv);
#endif
    int periods[3] = {false, false, true}; //non-, non-, periodic
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices==0){
        std::cerr << "No CUDA capable devices found"<<std::endl;
        return -1;
    }
    int device = rank % num_devices; //assume # of gpus/node is fixed
    std::cout << "# Rank "<<rank<<" computes with device "<<device<<" !"<<std::endl;
    cudaSetDevice( device);
#endif//THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int np[3];
    if(rank==0)
    {
        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads( );
#endif //omp
        std::cin>> np[0] >> np[1] >>np[2];
        std::cout << "# Computing with "
                  << np[0]<<" x "<<np[1]<<" x "<<np[2] << " processes x "
                  << num_threads<<" threads = "
                  <<size*num_threads<<" total"<<std::endl;
;
        assert( size == np[0]*np[1]*np[2] &&
        "Partition needs to match total number of processes!");
    }
    MPI_Bcast( np, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 3, np, periods, true, &comm);
    ////////////////////////////// Install signal handler ///////////////////
    std::signal(SIGINT, sigterm_handler);
    std::signal(SIGTERM, sigterm_handler);
#endif //FELTOR_MPI
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Value js, gs;
    Json::CharReaderBuilder parser;
    parser["collectComments"] = false;
    std::string errs;
    if( argc != 4 && argc != 5)
    {
        MPI_OUT std::cerr << "ERROR: Wrong number of arguments!\nUsage: "
                << argv[0]<<" [input.json] [geometry.json] [output.nc]\n OR \n"
                << argv[0]<<" [input.json] [geometry.json] [output.nc] [initial.nc] \n";
        return -1;
    }
    else
    {
        std::ifstream is(argv[1]);
        std::ifstream ks(argv[2]);
        parseFromStream( parser, is, &js, &errs); //read input without comments
        parseFromStream( parser, ks, &gs, &errs); //read input without comments
    }
    const feltor::Parameters p( js);
    const dg::geo::solovev::Parameters gp(gs);
    MPI_OUT p.display( std::cout);
    MPI_OUT gp.display( std::cout);
    std::string input = js.toStyledString(), geom = gs.toStyledString();
    ////////////////////////////////set up computations///////////////////////////
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grids
    Geometry grid( Rmin, Rmax, Zmin, Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.symmetric ? 1 : p.Nz, p.bcxN, p.bcyN, dg::PER
        #ifdef FELTOR_MPI
        , comm
        #endif //FELTOR_MPI
        );
    Geometry g3d_out( Rmin, Rmax, Zmin, Zmax, 0, 2.*M_PI,
        p.n_out, p.Nx_out, p.Ny_out, p.symmetric ? 1 : p.Nz_out, p.bcxN, p.bcyN, dg::PER
        #ifdef FELTOR_MPI
        , comm
        #endif //FELTOR_MPI
        );
    std::unique_ptr<typename Geometry::perpendicular_grid> g2d_out_ptr  ( dynamic_cast<typename Geometry::perpendicular_grid*>( g3d_out.perp_grid()));
#ifdef FELTOR_MPI
    unsigned local_size2d = g2d_out_ptr->local().size();
#else
    unsigned local_size2d = g2d_out_ptr->size();
#endif

    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    mag = dg::geo::createModifiedSolovevField(gp, (1.-p.rho_damping)*mag.psip()(mag.R0(),0.), p.alpha_mag);

    //create RHS
    MPI_OUT std::cout << "Constructing Explicit...\n";
    feltor::Explicit< Geometry, IDMatrix, DMatrix, DVec> feltor( grid, p, mag);
    MPI_OUT std::cout << "Constructing Implicit...\n";
    feltor::Implicit< Geometry, IDMatrix, DMatrix, DVec> im( grid, p, mag);
    MPI_OUT std::cout << "Done!\n";

    // helper variables for output computations
    std::map<std::string, dg::Simpsons<HVec>> time_integrals;
    dg::Average<HVec> toroidal_average( g3d_out, dg::coo3d::z);
    dg::MultiMatrix<HMatrix,HVec> projectH = dg::create::fast_projection( grid, p.cx, p.cy, dg::normed);
    dg::MultiMatrix<DMatrix,DVec> projectD = dg::create::fast_projection( grid, p.cx, p.cy, dg::normed);
    HVec transferH( dg::evaluate(dg::zero, g3d_out));
    DVec transferD( dg::evaluate(dg::zero, g3d_out));
    HVec transferH2d = dg::evaluate( dg::zero, *g2d_out_ptr);
    DVec transferD2d = dg::evaluate( dg::zero, *g2d_out_ptr);
    HVec resultH = dg::evaluate( dg::zero, grid);
    DVec resultD = dg::evaluate( dg::zero, grid);

    std::array<DVec, 3> gradPsip;
    gradPsip[0] =  dg::evaluate( mag.psipR(), grid);
    gradPsip[1] =  dg::evaluate( mag.psipZ(), grid);
    gradPsip[2] =  resultD; //zero
    feltor::Variables var = {
        feltor, p, gradPsip, gradPsip
    };
    // the vector ids
    std::map<std::string, int> id3d, id4d;

    double dEdt = 0, accuracy = 0;
    double E0 = 0.;

    /// //////////////////The initial field///////////////////////////////////////////
    double time = 0.;
    std::array<std::array<DVec,2>,2> y0;
    feltor::Initialize init( p, gp, mag);
    if( argc == 4)
        y0 = init.init_from_parameters(feltor, grid);
    if( argc == 5)
        y0 = init.init_from_file(argv[4], grid, time);
    feltor.set_source( init.profile(grid), p.omega_source, init.source_damping(grid));

    /// //////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    std::string file_name = argv[3];
    int ncid=-1;
    MPI_OUT err = nc_create( file_name.data(), NC_NETCDF4|NC_CLOBBER, &ncid);
    /// Set global attributes
    std::map<std::string, std::string> att;
    att["title"] = "Output file of feltor/src/feltor_hpc.cu";
    att["Conventions"] = "CF-1.7";
    ///Get local time and begin file history
    auto ttt = std::time(nullptr);
    auto tm = *std::localtime(&ttt);

    std::ostringstream oss;
    ///time string  + program-name + args
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    for( int i=0; i<argc; i++) oss << " "<<argv[i];
    att["history"] = oss.str();
    att["comment"] = "Find more info in feltor/src/feltor.tex";
    att["source"] = "FELTOR";
    att["references"] = "https://github.com/feltor-dev/feltor";
    att["inputfile"] = input;
    att["geomfile"] = geom;
    for( auto pair : att)
        MPI_OUT err = nc_put_att_text( ncid, NC_GLOBAL,
            pair.first.data(), pair.second.size(), pair.second.data());

    // Define dimensions (t,z,y,x)
    int dim_ids[4], tvarID;
#ifdef FELTOR_MPI
    MPI_OUT err = file::define_dimensions( ncid, dim_ids, &tvarID, g3d_out.global());
#else //FELTOR_MPI
    err = file::define_dimensions( ncid, dim_ids, &tvarID, g3d_out);
#endif //FELTOR_MPI
    int dim_ids3d[3] = {dim_ids[0], dim_ids[2], dim_ids[3]};

    //create & output static 3d variables into file
    for ( auto& record : feltor::diagnostics3d_static_list)
    {
        int vecID;
        MPI_OUT err = nc_def_var( ncid, record.name.data(), NC_DOUBLE, 3,
            &dim_ids[1], &vecID);
        MPI_OUT err = nc_put_att_text( ncid, vecID,
            "long_name", record.long_name.size(), record.long_name.data());
        MPI_OUT err = nc_enddef( ncid);
        MPI_OUT std::cout << "Computing "<<record.name<<"\n";
        record.function( resultH, var, grid, gp, mag);
        dg::blas2::symv( projectH, resultH, transferH);
        file::write_static3d( ncid, vecID, transferH, g3d_out);
        MPI_OUT err = nc_redef(ncid);
    }

    //Create field IDs
    for( auto& record : feltor::diagnostics3d_list)
    {
        std::string name = record.name;
        std::string long_name = record.long_name;
        id4d[name] = 0;//creates a new id4d entry for all processes
        MPI_OUT err = nc_def_var( ncid, name.data(), NC_DOUBLE, 4, dim_ids,
            &id4d.at(name));
        MPI_OUT err = nc_put_att_text( ncid, id4d.at(name), "long_name", long_name.size(),
            long_name.data());
    }
    for( auto& record : feltor::diagnostics2d_list)
    {
        std::string name = record.name + "_ta2d";
        std::string long_name = record.long_name + " (Toroidal average)";
        id3d[name] = 0;//creates a new id3d entry for all processes
        MPI_OUT err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids3d,
            &id3d.at(name));
        MPI_OUT err = nc_put_att_text( ncid, id3d.at(name), "long_name", long_name.size(),
            long_name.data());

        name = record.name + "_2d";
        long_name = record.long_name + " (Evaluated on phi = 0 plane)";
        id3d[name] = 0;
        MPI_OUT err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids3d,
            &id3d.at(name));
        MPI_OUT err = nc_put_att_text( ncid, id3d.at(name), "long_name", long_name.size(),
            long_name.data());
    }
    MPI_OUT err = nc_enddef(ncid);
    ///////////////////////////////////first output/////////////////////////
    MPI_OUT std::cout << "First output ... \n";
    //first, update feltor (to get potential etc.)
    {
        std::array<std::array<DVec,2>,2> y1(y0);
        try{
            feltor( time, y0, y1);
        } catch( dg::Fail& fail) {
            MPI_OUT std::cerr << "CG failed to converge in first step to "
                              <<fail.epsilon()<<"\n";
            MPI_OUT err = nc_close(ncid);
            return -1;
        }
    }

    size_t start = 0, count = 1;
    MPI_OUT err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
    for( auto& record : feltor::diagnostics3d_list)
    {
        record.function( resultD, var);
        dg::blas2::symv( projectD, resultD, transferD);
        dg::assign( transferD, transferH);
        file::write_dynamic3d( ncid, id4d.at(record.name), start, transferH, g3d_out);
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
        MPI_OUT std::cout<< name << " Computing average took "<<tti.diff()<<"\n";
        tti.tic();
#ifdef FELTOR_MPI
        //only the globally first slice should write
        if( g3d_out.local().z0() - g3d_out.global().z0() < 1e-14)
#endif //FELTOR_MPI
            file::write_dynamic2d( ncid, id3d.at(name), start, transferH2d, *g2d_out_ptr);
        tti.toc();
        MPI_OUT std::cout<< name << " 2d output took "<<tti.diff()<<"\n";
        tti.tic();

        // and a slice
        name = record.name + "_2d";
        feltor::slice_vector3d( transferD, transferD2d, local_size2d);
        dg::assign( transferD2d, transferH2d);
        if( record.integral) time_integrals[name].init( time, transferH2d);
#ifdef FELTOR_MPI
        //only the globally first slice should write
        if( g3d_out.local().z0() - g3d_out.global().z0() < 1e-14)
#endif //FELTOR_MPI
            file::write_dynamic2d( ncid, id3d.at(name), start, transferH2d, *g2d_out_ptr);
        tti.toc();
        MPI_OUT std::cout<< name << " 2d output took "<<tti.diff()<<"\n";
    }
    MPI_OUT err = nc_close(ncid);
    MPI_OUT std::cout << "First write successful!\n";
    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Karniadakis< std::array<std::array<DVec,2>,2 >,
        feltor::FeltorSpecialSolver<
            Geometry, IDMatrix, DMatrix, DVec>
        > karniadakis( grid, p, mag);
    karniadakis.init( feltor, im, time, y0, p.dt);
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
                    karniadakis.step( feltor, im, time, y0);
                }
                catch( dg::Fail& fail) {
                    MPI_OUT std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                    MPI_OUT std::cerr << "Does Simulation respect CFL condition?\n";
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
                if( std::find( feltor::energy_diff.begin(), feltor::energy_diff.end(), record.name) != feltor::energy_diff.end())
                {
                    record.function( resultD, var);
                    ediff += dg::blas1::dot( resultD, feltor.vol3d());
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
                }

            }

            dEdt = (energy - E0)/deltat;
            E0 = energy;
            accuracy  = 2.*fabs( (dEdt - ediff)/( dEdt + ediff));

            MPI_OUT std::cout << "\tTime "<<time<<"\n";
            MPI_OUT std::cout <<"\td E/dt = " << dEdt
                      <<" Lambda = " << ediff
                      <<" -> Accuracy: " << accuracy << "\n";
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
                MPI_OUT std::cout << "\tRel. Error Induction "<<sqrt(error/norm) <<"\n";
            }
            tti.toc();
            MPI_OUT std::cout << " Time for internal diagnostics "<<tti.diff()<<"s\n";
        }
        ti.toc();
        MPI_OUT std::cout << "\n\t Step "<<step <<" of "
                    << p.inner_loop*p.itstp*p.maxout << " at time "<<time;
        MPI_OUT std::cout << "\n\t Average time for one step: "
                    << ti.diff()/(double)p.itstp/(double)p.inner_loop<<"s";
        ti.tic();
        //////////////////////////write fields////////////////////////
        start = i;
        MPI_OUT err = nc_open(file_name.data(), NC_WRITE, &ncid);
        MPI_OUT err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
        for( auto& record : feltor::diagnostics3d_list)
        {
            record.function( resultD, var);
            dg::blas2::symv( projectD, resultD, transferD);
            dg::assign( transferD, transferH);
            file::write_dynamic3d( ncid, id4d.at(record.name), start, transferH, g3d_out);
        }
        for( auto& record : feltor::diagnostics2d_list)
        {
            if(record.integral) // we already computed the output...
            {
                std::string name = record.name+"_ta2d";
                transferH2d = time_integrals.at(name).get_integral();
                time_integrals.at(name).flush();
#ifdef FELTOR_MPI
                //only the globally first slice should write
                if( g3d_out.local().z0() - g3d_out.global().z0() < 1e-14)
#endif //FELTOR_MPI
                    file::write_dynamic2d( ncid, id3d.at(name), start, transferH2d, *g2d_out_ptr);

                name = record.name+"_2d";
                transferH2d = time_integrals.at(name).get_integral( );
                time_integrals.at(name).flush( );
#ifdef FELTOR_MPI
                //only the globally first slice should write
                if( g3d_out.local().z0() - g3d_out.global().z0() < 1e-14)
#endif //FELTOR_MPI
                    file::write_dynamic2d( ncid, id3d.at(name), start, transferH2d, *g2d_out_ptr);
            }
            else //manage the time integrators
            {
                record.function( resultD, var);
                dg::blas2::symv( projectD, resultD, transferD);

                std::string name = record.name+"_ta2d";
                dg::assign( transferD, transferH);
                toroidal_average( transferH, transferH2d, false);
#ifdef FELTOR_MPI
                //only the globally first slice should write
                if( g3d_out.local().z0() - g3d_out.global().z0() < 1e-14)
#endif //FELTOR_MPI
                    file::write_dynamic2d( ncid, id3d.at(name), start, transferH2d, *g2d_out_ptr);

                // 2d data of plane varphi = 0
                name = record.name+"_2d";
                feltor::slice_vector3d( transferD, transferD2d, local_size2d);
                dg::assign( transferD2d, transferH2d);
#ifdef FELTOR_MPI
                //only the globally first slice should write
                if( g3d_out.local().z0() - g3d_out.global().z0() < 1e-14)
#endif //FELTOR_MPI
                    file::write_dynamic2d( ncid, id3d.at(name), start, transferH2d, *g2d_out_ptr);
            }
        }
        MPI_OUT err = nc_close(ncid);
        ti.toc();
        MPI_OUT std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
    }
    t.toc();
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    MPI_OUT std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    MPI_OUT std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    MPI_OUT std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout/p.inner_loop<<"s/step\n";
#ifdef FELTOR_MPI
    MPI_Finalize();
#endif //FELTOR_MPI

    return 0;

}
