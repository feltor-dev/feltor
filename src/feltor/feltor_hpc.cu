#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <sstream>
#include <cmath>

#ifdef FELTOR_MPI
#include <mpi.h>
#include "netcdf_par.h"
#endif //FELTOR_MPI

#include "dg/file/nc_utilities.h"
#include "feltor.cuh"

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
#endif //FELTOR_MPI
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Value js, gs;
    Json::CharReaderBuilder parser;
    parser["collectComments"] = false;
    std::string errs;
    if( argc != 4)
    {
        MPI_OUT std::cerr << "ERROR: Wrong number of arguments!\nUsage: "
                << argv[0]<<" [inputfile] [geomfile] [outputfile]\n";
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
    Geometry grid_out( Rmin, Rmax, Zmin, Zmax, 0, 2.*M_PI,
        p.n_out, p.Nx_out, p.Ny_out, p.symmetric ? 1 : p.Nz_out, p.bcxN, p.bcyN, dg::PER
        #ifdef FELTOR_MPI
        , comm
        #endif //FELTOR_MPI
        );
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);

    //create RHS
    MPI_OUT std::cout << "Constructing Explicit...\n";
    feltor::Explicit< Geometry, IDMatrix, DMatrix, DVec> feltor( grid, p, mag);
    MPI_OUT std::cout << "Constructing Implicit...\n";
    feltor::Implicit< Geometry, IDMatrix, DMatrix, DVec> im( grid, p, mag);
    MPI_OUT std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //First the profile and the source (on the host since we want to output those)
    HVec profile = dg::pullback( dg::geo::Compose<dg::LinearX>( mag.psip(),
        p.nprofamp/mag.psip()(mag.R0(), 0.), 0.), grid);
    HVec xpoint_damping = dg::evaluate( dg::one, grid);
    if( gp.hasXpoint() )
        xpoint_damping = dg::pullback(
            dg::geo::ZCutter(-1.1*gp.elongation*gp.a), grid);
    HVec source_damping = dg::pullback(dg::geo::TanhDamping(
        //first change coordinate from psi to (psi_0 - psip)/psi_0
        dg::geo::Compose<dg::LinearX>( mag.psip(), -1./mag.psip()(mag.R0(), 0.),1.),
        //then shift tanh
        p.rho_source-3.*p.alpha, p.alpha, -1.), grid);
    HVec damping_damping = dg::pullback(dg::geo::TanhDamping(
        //first change coordinate from psi to (psi_0 - psip)/psi_0
        dg::geo::Compose<dg::LinearX>( mag.psip(), -1./mag.psip()(mag.R0(), 0.),1.),
        //then shift tanh
        p.rho_damping, p.alpha, 1.), grid);
    dg::blas1::pointwiseDot( xpoint_damping, source_damping, source_damping);

    HVec profile_damping = dg::pullback( dg::geo::TanhDamping(
        mag.psip(), -3.*p.alpha, p.alpha, -1), grid);
    dg::blas1::pointwiseDot( xpoint_damping, profile_damping, profile_damping);
    dg::blas1::pointwiseDot( profile_damping, profile, profile);

    feltor.set_source( profile, p.omega_source, source_damping,
        p.omega_damping, damping_damping);
    im.set_damping( p.omega_damping, damping_damping);

    //Now perturbation
    HVec ntilde = dg::evaluate(dg::zero,grid);
    if( p.initne == "blob" || p.initne == "straight blob")
    {
        dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
        dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
        if( p.symmetric)
            ntilde = dg::pullback( init0, grid);
        else if( p.initne == "blob")//rounds =3 ->2*3-1
        {
            dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                fieldaligned( mag, grid, p.bcxN, p.bcyN,
                dg::geo::NoLimiter(), p.rk4eps, 5, 5);
            //evaluate should always be used with mx,my > 1
            ntilde = fieldaligned.evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 3);
        }
        else if( p.initne == "straight blob")//rounds =1 ->2*1-1
        {
            dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                fieldaligned( mag, grid, p.bcxN, p.bcyN,
                dg::geo::NoLimiter(), p.rk4eps, 5, 5);
            //evaluate should always be used with mx,my > 1
            ntilde = fieldaligned.evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1);
        }
    }
    else if( p.initne == "turbulence")
    {
        dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
        dg::BathRZ init0(16,16,Rmin,Zmin, 30.,5.,p.amp);
        if( p.symmetric)
            ntilde = dg::pullback( init0, grid);
        else
        {
            dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                fieldaligned( mag, grid, p.bcxN, p.bcyN,
                dg::geo::NoLimiter(), p.rk4eps, 5, 5);
            //evaluate should always be used with mx,my > 1
            ntilde = fieldaligned.evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 1);
        }
        dg::blas1::pointwiseDot( profile_damping, ntilde, ntilde);
    }
    else if( p.initne == "zonal")
    {
        dg::geo::ZonalFlow init0(mag.psip(), p.amp, 0., p.k_psi);
        ntilde = dg::pullback( init0, grid);
        dg::blas1::pointwiseDot( profile_damping, ntilde, ntilde);
    }
    else
        MPI_OUT std::cerr <<"WARNING: Unknown initial condition!\n";
    std::array<std::array<DVec,2>,2> y0;
    y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(profile);
    dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0]);
    MPI_OUT std::cout << "initialize ni" << std::endl;
    if( p.initphi == "zero")
        feltor.initializeni( y0[0][0], y0[0][1]);
    else if( p.initphi == "balance")
        dg::blas1::copy( y0[0][0], y0[0][1]); //set N_i = n_e
    else
        MPI_OUT std::cerr <<"WARNING: Unknown initial condition for phi!\n";

    dg::blas1::copy( 0., y0[1][0]); //set we = 0
    dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
    ////////////map quantities to output/////////////////
    //since we map pointers we don't need to update those later
    std::map<std::string, const DVec* > v4d;
    v4d["electrons"] = &feltor.fields()[0][0], v4d["ions"] = &feltor.fields()[0][1];
    v4d["Ue"] = &feltor.fields()[1][0],        v4d["Ui"] = &feltor.fields()[1][1];
    v4d["potential"] = &feltor.potential()[0];
    v4d["induction"] = &feltor.induction();
    const feltor::Quantities& q = feltor.quantities();
    double dEdt = 0, accuracy = 0, dMdt = 0, accuracyM  = 0;
    std::map<std::string, const double*> v0d{
        {"energy", &q.energy}, {"ediff", &q.ediff},
        {"mass", &q.mass}, {"diff", &q.diff}, {"Apar", &q.Apar},
        {"Se", &q.S[0]}, {"Si", &q.S[1]}, {"Uperp", &q.Tperp},
        {"Upare", &q.Tpar[0]}, {"Upari", &q.Tpar[1]},
        {"dEdt", &dEdt}, {"accuracy", &accuracy},
        {"aligned", &q.aligned}
    };
    /////////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
#ifdef FELTOR_MPI
    MPI_Info info = MPI_INFO_NULL;
    err = nc_create_par( argv[3], NC_NETCDF4|NC_MPIIO|NC_CLOBBER, comm, info, &ncid);
#else //FELTOR_MPI
    err = nc_create( argv[3],NC_NETCDF4|NC_CLOBBER, &ncid);
#endif //FELTOR_MPI
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    err = nc_put_att_text( ncid, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids[4], tvarID;
#ifdef FELTOR_MPI
    err = file::define_dimensions( ncid, dim_ids, &tvarID, grid_out.global());
    err = nc_var_par_access( ncid, tvarID, NC_COLLECTIVE);
#else //FELTOR_MPI
    err = file::define_dimensions( ncid, dim_ids, &tvarID, grid_out);
#endif //FELTOR_MPI
    {   //output 3d variables into file
        dg::geo::BFieldR fieldR(mag);
        dg::geo::BFieldZ fieldZ(mag);
        dg::geo::BFieldP fieldP(mag);

        HVec vecR = dg::pullback( fieldR, grid);
        HVec vecZ = dg::pullback( fieldZ, grid);
        HVec vecP = dg::pullback( fieldP, grid);
        HVec psip = dg::pullback( mag.psip(), grid);
        std::map<std::string, const HVec*> v3d{
            {"BR", &vecR}, {"BZ", &vecZ}, {"BP", &vecP},
            {"Psip", &psip}, {"Nprof", &profile },
            {"Source", &source_damping }, {"Damping", &damping_damping}
        };
        IHMatrix project = dg::create::projection( grid_out, grid);
        HVec transferH( dg::evaluate(dg::zero, grid_out));
        for( auto pair : v3d)
        {
            int vecID;
            err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 3,
                &dim_ids[1], &vecID);
            #ifdef FELTOR_MPI
            err = nc_var_par_access( ncid, vecID, NC_COLLECTIVE);
            #endif //FELTOR_MPI
            err = nc_enddef( ncid);
            dg::blas2::symv( project, *pair.second, transferH);
            err = nc_put_var_double( ncid, vecID,
            #ifdef FELTOR_MPI
                transferH.data().data()
            #else //FELTOR_MPI
                transferH.data()
            #endif //FELTOR_MPI
            );
            err = nc_redef(ncid);
        }
    }

    //field IDs
    int EtimeID, EtimevarID;
    err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
#ifdef FELTOR_MPI
    err = nc_var_par_access( ncid, EtimevarID, NC_COLLECTIVE);
#endif //FELTOR_MPI
    std::map<std::string, int> id0d, id4d;
    for( auto pair : v0d)
    {
        err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 1,
            &EtimeID, &id0d[pair.first]);
#ifdef FELTOR_MPI
        err = nc_var_par_access( ncid, id0d[pair.first], NC_COLLECTIVE);
#endif //FELTOR_MPI
    }
    for( auto pair : v4d)
    {
        err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 4,
            dim_ids, &id4d[pair.first]);
#ifdef FELTOR_MPI
        err = nc_var_par_access( ncid, id4d[pair.first], NC_COLLECTIVE);
#endif //FELTOR_MPI
    }
    err = nc_enddef(ncid);
    ///////////////////////////////////first output/////////////////////////
    double time = 0, dt = p.dt;
    MPI_OUT std::cout << "First output ... \n";
    //first, update quantities in feltor
    {
        std::array<std::array<DVec,2>,2> y1(y0);
        try{
            feltor( time, y0, y1);
        } catch( dg::Fail& fail) {
            MPI_OUT std::cerr << "CG failed to converge in first step to "
                              <<fail.epsilon()<<"\n";
            err = nc_close(ncid);
            return -1;
        }
        feltor.update_quantities();
    }
    MPI_OUT q.display(std::cout);
    double energy0 = q.energy, mass0 = q.mass, E0 = energy0, M0 = mass0;
#ifdef FELTOR_MPI
    int dims[3],  coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    size_t count[4] = {1, grid_out.local().Nz(),
        grid_out.n()*(grid_out.local().Ny()),
        grid_out.n()*(grid_out.local().Nx())};
    size_t start[4] = {0, coords[2]*count[1],
                          coords[1]*count[2],
                          coords[0]*count[3]};
#else //FELTOR_MPI
    size_t start[4] = {0, 0, 0, 0};
    size_t count[4] = {1, grid_out.Nz(), grid_out.n()*grid_out.Ny(),
        grid_out.n()*grid_out.Nx()};
#endif //FELTOR_MPI
    DVec transferD( dg::evaluate(dg::zero, grid_out));
    HVec transferH( dg::evaluate(dg::zero, grid_out));
    IDMatrix project = dg::create::projection( grid_out, grid);
    for( auto pair : v4d)
    {
        dg::blas2::symv( project, *pair.second, transferD);
        dg::assign( transferD, transferH);
        err = nc_put_vara_double( ncid, id4d.at(pair.first), start, count,
            #ifdef FELTOR_MPI
            transferH.data().data()
            #else //FELTOR_MPI
            transferH.data()
            #endif //FELTOR_MPI
        );
    }
    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    err = nc_put_vara_double( ncid, EtimevarID, start, count, &time);

    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    for( auto pair : v0d)
        err = nc_put_vara_double( ncid, id0d.at(pair.first),
            Estart, Ecount, pair.second);
#ifndef FELTOR_MPI
    err = nc_close(ncid);
#endif //FELTOR_MPI
    MPI_OUT std::cout << "First write successful!\n";
    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Adaptive< dg::ARKStep<std::array<std::array<DVec,2>,2>> > adaptive(
        "ARK-4-2-3", y0, grid.size(), p.eps_time);
    dg::Timer t;
    t.tic();
    unsigned step = 0, failed_counter = 0;
    MPI_OUT q.display(std::cout);
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
                    do
                    {
                        adaptive.step( feltor, im, time, y0, time, y0, dt,
                            dg::pid_control, dg::l2norm, p.rtol, 1e-10);
                        if( adaptive.failed())
                        {
                            MPI_OUT std::cout << "FAILED STEP! REPEAT!\n";
                            failed_counter++;
                        }
                    }while ( adaptive.failed());
                }
                catch( dg::Fail& fail) {
                    MPI_OUT std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                    MPI_OUT std::cerr << "Does Simulation respect CFL condition?\n";
                    #ifdef FELTOR_MPI
                    err = nc_close(ncid);
                    #endif //FELTOR_MPI
                    return -1;
                }
                step++;
            }
            double deltat = time - previous_time;

            feltor.update_quantities();
            MPI_OUT std::cout << "Timestep "<<deltat<<"\n";
            dEdt = (*v0d["energy"] - E0)/deltat, dMdt = (*v0d["mass"] - M0)/deltat;
            E0 = *v0d["energy"], M0 = *v0d["mass"];
            accuracy  = 2.*fabs( (dEdt - *v0d["ediff"])/( dEdt + *v0d["ediff"]));
            accuracyM = 2.*fabs( (dMdt - *v0d["diff"])/( dMdt + *v0d["diff"]));
            #ifndef FELTOR_MPI
            err = nc_open(argv[3], NC_WRITE, &ncid);
            #endif //FELTOR_MPI
            Estart[0] = step;
            err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
            for( auto pair : v0d)
                err = nc_put_vara_double( ncid, id0d.at(pair.first),
                    Estart, Ecount, pair.second);

            MPI_OUT q.display(std::cout);
            MPI_OUT std::cout << "(m_tot-m_0)/m_0: "<< (*v0d["mass"]-mass0)/mass0<<"\t";
            MPI_OUT std::cout << "(E_tot-E_0)/E_0: "<< (*v0d["energy"]-energy0)/energy0<<"\t";
            MPI_OUT std::cout <<" d E/dt = " << dEdt
                      <<" Lambda = " << *v0d["ediff"]
                      <<" -> Accuracy: " << accuracy << "\n";
            MPI_OUT std::cout <<" d M/dt = " << dMdt
                      <<" Lambda = " << *v0d["diff"]
                      <<" -> Accuracy: " << accuracyM << "\n";
            #ifndef FELTOR_MPI
            err = nc_close(ncid);
            #endif //FELTOR_MPI

        }
        ti.toc();
        MPI_OUT std::cout << "\n\t Step "<<step <<" of "
                    << p.inner_loop*p.itstp*p.maxout << " at time "<<time;
        MPI_OUT std::cout << "\n\t Average time for one step: "
                    << ti.diff()/(double)p.itstp/(double)p.inner_loop<<"s";
        MPI_OUT std::cout << "\n\t Total number of failed steps: "
                    << failed_counter;
        ti.tic();
        //////////////////////////write fields////////////////////////
        start[0] = i;
        #ifndef FELTOR_MPI
        err = nc_open(argv[3], NC_WRITE, &ncid);
        #endif //FELTOR_MPI
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        for( auto pair : v4d)
        {
            dg::blas2::symv( project, *pair.second, transferD);
            dg::assign( transferD, transferH);
            err = nc_put_vara_double( ncid, id4d.at(pair.first), start, count,
                #ifdef FELTOR_MPI
                transferH.data().data()
                #else //FELTOR_MPI
                transferH.data()
                #endif //FELTOR_MPI
            );
        }
        #ifndef FELTOR_MPI
        err = nc_close(ncid);
        #endif //FELTOR_MPI
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
    err = nc_close(ncid);
    MPI_Finalize();
#endif //FELTOR_MPI

    return 0;

}
