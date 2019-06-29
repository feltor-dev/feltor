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
#include "feltor.cuh"
#include "implicit.h"

#ifdef FELTOR_MPI
using HVec = dg::MHVec;
using DVec = dg::MDVec;
using DMatrix = dg::MDMatrix;
using IDMatrix = dg::MIDMatrix;
using IHMatrix = dg::MIHMatrix;
using Geometry = dg::CylindricalMPIGrid3d;
#define MPI_OUT if(rank==0)
#else //FELTOR_MPI
using HVec = dg::HVec;
using DVec = dg::DVec;
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

    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    mag = dg::geo::createModifiedSolovevField(gp, (1.-p.rho_damping)*mag.psip()(mag.R0(),0.), p.alpha_mag);

    //create RHS
    MPI_OUT std::cout << "Constructing Explicit...\n";
    feltor::Explicit< Geometry, IDMatrix, DMatrix, DVec> feltor( grid, p, mag);
    MPI_OUT std::cout << "Constructing Implicit...\n";
    feltor::Implicit< Geometry, IDMatrix, DMatrix, DVec> im( grid, p, mag);
    MPI_OUT std::cout << "Done!\n";

    // helper variables for various stuff
    std::array<DVec, 3> gradPsip;
    gradPsip[0] =  dg::evaluate( mag.psipR(), grid);
    gradPsip[1] =  dg::evaluate( mag.psipZ(), grid);
    gradPsip[2] =  result; //zero
    feltor::Variables var = {
        feltor, p, gradPsip, gradPsip
    };
    std::map<std::string, dg::Simpsons<DVec>> time_integrals;
    dg::Average<DVec> toroidal_average( g3d_out, dg::coo3d::z);
    dg::MultiMatrix<IHMatrix,HVec> projectH = dg::create::fast_projection( grid, p.cx, p.cy, dg::normed);
    dg::MultiMatrix<IDMatrix,DVec> projectD = dg::create::fast_projection( grid, p.cx, p.cy, dg::normed);
    HVec transferH( dg::evaluate(dg::zero, g3d_out));
    DVec transferD( dg::evaluate(dg::zero, g3d_out));
    DVec transferD2d = dg::evaluate( dg::zero, g2d_out);
    HVec transferH2d = dg::evaluate( dg::zero, g2d_out);
    /// Construct feltor::Variables object for diagnostics
    DVec result = dg::evaluate( dg::zero, grid);
    HVec resultH( dg::evaluate( dg::zero, grid));

    double dEdt = 0, accuracy = 0;
    double E0 = 0.;

    //!///////////////////The initial field///////////////////////////////////////////
    double time = 0;
    std::array<std::array<DVec,2>,2> y0;
    feltor::Initialize init( grid, p, mag);
    if( argc == 4)
        y0 = init.init_from_parameters(feltor);
    if( argc == 5)
        y0 = init.init_from_file(argv[4]);
    feltor.set_source( init.profile(), p.omega_source, init.source_damping());

    /// //////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    std::string file_name = argv[3];
    int ncid=-1;
    MPI_OUT err = nc_create( file_name.data(), NC_NETCDF4|NC_CLOBBER, &ncid);
    feltor::ManageOutput output(g3d_out);
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

    //create & output static 3d variables into file
    for ( auto record& diagnostics3d_static_list)
    {
        int vecID;
        MPI_OUT err = nc_def_var( ncid, record.name.data(), NC_DOUBLE, 3,
            &dim_ids[1], &vecID);
        MPI_OUT err = nc_put_att_text( ncid, vecID,
            "long_name", record.long_name.size(), record.long_name.data());
        MPI_OUT err = nc_enddef( ncid);
        record.function( resultH, var, grid, gp, mag);
        dg::blas2::symv( projectH, resultH, transferH);
        output.output_static3d( ncid, vecID, transferH);
        MPI_OUT err = nc_redef(ncid);
    }

    //Create field IDs
    std::map<std::string, int> id3d, id4d;
    for( auto record& : feltor::diagnostics3d_list)
    {
        std::string name = record.name;
        std::string long_name = record.long_name;
        MPI_OUT err = nc_def_var( ncid, name.data(), NC_DOUBLE, 4, dim_ids,
            &id4d[name]);//creates a new id4d entry
        MPI_OUT err = nc_put_att_text( ncid, id4d[name], "long_name", long_name.size(),
            long_name.data());
    }
    for( auto record& : feltor::diagnostics2d_list)
    {
        std::string name = record.name + "_ta2d";
        std::string long_name = record.long_name + " (Toroidal average)";
        if( record.integral){
            name += "_tt";
            long_name+= " (Time average)";
        }
        MPI_OUT err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
            &id3d[name]);//creates a new id3d entry
        MPI_OUT err = nc_put_att_text( ncid, id3d[name], "long_name", long_name.size(),
            long_name.data());

        name = record.name + "_2d";
        long_name = record.long_name + " (Evaluated on phi = pi plane)";
        if( record.integral){
            name += "_tt";
            long_name+= " (Time average)";
        }
        err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
            &id3d[name]);
        err = nc_put_att_text( ncid, id3d[name], "long_name", long_name.size(),
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
    for( auto record& : diagnostics3d_list)
    {
        record.function( result, var);
        dg::blas2::symv( projectD, result, transferD);
        dg::assign( transferD, transferH);
        output.output_dynamic3d( ncid, id4d.at(record.name), start, transferH);
    }
    for( auto record& : diagnostics2d_list)
    {
        record.function( result, var);
        dg::blas2::symv( projectD, result, transferD);
        //toroidal average
        toroidal_average( transferD, transfer2dD);
        dg::assign( transfer2dD, transfer2dH);

        // 2d data of plane varphi = 0
        dg::HVec t2d_mp(result.data().begin(),
            result.data().begin() + g2d_out.size() );
        //compute toroidal average and extract 2d field
        output.output_dynamic2d( ncid, id3d.at(name), start, transferH2d);
    }
    MPI_OUT err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
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
            Timer tti;
            tti.tic();
            double deltat = time - previous_time;
            double energy = 0, ediff = 0.;
            for( auto record& : diagnostics2d_list)
            {
                if( std::find( feltor::energies.begin(), feltor::energies.end(), record.name) != feltor::energies.end())
                {
                    record.function( result, var);
                    energy += dg::blas1::dot( result, feltor.vol3d());
                }
                if( std::find( feltor::energy_diff.begin(), feltor::energy_diff.end(), record.name) != feltor::energy_diff.end())
                {
                    record.function( result, var);
                    ediff += dg::blas1::dot( result, feltor.vol3d());
                }
                if( record.integral)
                {
                    record.function( result, var);
                    dg::blas2::symv( projectD, result, transferD);
                    //toroidal average and add to time integral
                    toroidal_average( transferD, transferD2d);
                    time_integrals.at(record.name+"_ta2d_tt").add( time, transferD2d);

                    // 2d data of plane varphi = 0
                    dg::HVec t2d_mp(result.data().begin(),
                        result.data().begin() + g2d_out.size() );
                    time_integrals.at(record.name+"_2d_tt").add( time, transfer2dD);
                }

            }

            dEdt = (energy - E0)/deltat;
            E0 = energy;
            accuracy  = 2.*fabs( (dEdt - ediff)/( dEdt + ediff));

            MPI_OUT std::cout << "Time "<<time<<"\n";
            MPI_OUT std::cout <<" d E/dt = " << dEdt
                      <<" Lambda = " << ediff
                      <<" -> Accuracy: " << accuracy << "\n";
            //----------------Test if induction equation holds
            if( p.beta != 0)
            {
                dg::blas1::pointwiseDot(
                    feltor.density(0), feltor.velocity(0), temp);
                dg::blas1::pointwiseDot( p.beta,
                    feltor.density(1), feltor.velocity(1), -p.beta, temp);
                double norm  = dg::blas2::dot( temp, feltor.vol3d(), temp);
                dg::blas1::axpby( -1., feltor.lapMperpA(), 1., temp);
                double error = dg::blas2::dot( temp, feltor.vol3d(), temp);
                MPI_OUT std::cout << " Rel. Error Induction "<<sqrt(error/norm) <<"\n";
            }
            tti.tic();
            std::cout << " Time for internal diagnostics "<<tti.diff()<<"s\n";
        }
        ti.toc();
        MPI_OUT std::cout << "\n\t Step "<<step <<" of "
                    << p.inner_loop*p.itstp*p.maxout << " at time "<<time;
        MPI_OUT std::cout << "\n\t Average time for one step: "
                    << ti.diff()/(double)p.itstp/(double)p.inner_loop<<"s";
        ti.tic();
        //////////////////////////write fields////////////////////////
        start4d[0] = i;
        MPI_OUT err = nc_open(file_name.data(), NC_WRITE, &ncid);
        MPI_OUT err = nc_put_vara_double( ncid, tvarID, start4d, count4d, &time);
        for( auto record& : diagnostics3d_list)
        {
            record.function( result, var);
            dg::blas2::symv( projectD, result, transferD);
            dg::assign( transferD, transferH);
            output.output_dynamic3d( ncid, id4d.at(record.name), start, transferH);
        }
        for( auto record& : diagnostics2d_list)
        {
            if(!record.integral)
            {
                record.function( result, var);
                dg::blas2::symv( projectD, result, transferD);
                //toroidal average
                toroidal_average( transferD, transfer2dD);
                dg::assign( transfer2dD, transfer2dH);

                // 2d data of plane varphi = 0
                dg::HVec t2d_mp(result.data().begin(),
                    result.data().begin() + g2d_out.size() );
            }
            else //manage the time integrators
            {
                std::string name = record.name+"_ta2d_tt";
                transfer2dD = time_integrals.at(name).get_integral( );
                std::array<double,2> bb = time_integrals.at(name).get_boundaries( );
                dg::scal( transfer2dD, 1./(bb[1]-bb[0]));
                time_integrals.at(name).flush( );
                dg::assign( transferD2d, transferH2d);
                output.output_dynamic2d( ncid, id3d.at(name), start, transferH2d);

                name = record.name+"_2d_tt";
                transfer2dD = time_integrals.at(name).get_integral( );
                std::array<double,2> bb = time_integrals.at(name).get_boundaries( );
                dg::scal( transferD2d, 1./(bb[1]-bb[0]));
                time_integrals.at(name).flush( );
                dg::assign( transferD2d, transferH2d);
                output.output_dynamic2d( ncid, id3d.at(name), start, transferH2d);
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
