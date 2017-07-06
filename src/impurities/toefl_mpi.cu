#include <iostream>
#include <iomanip>
#include <vector>

#include <mpi.h> //activate mpi
#include "netcdf_par.h"

#include "toeflI.cuh"
#include "parameters.h"

#include "file/nc_utilities.h"

#include "dg/backend/timer.cuh"

/*
   - reads parameters from input.txt or any other given file,
   - integrates the ToeflR - functor and
   - writes outputs to a given outputfile using netcdf4.
*/

int main( int argc, char* argv[])
{   ////////////////////////////////setup MPI///////////////////////////////
    int provided;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if( provided != MPI_THREAD_FUNNELED)
    {   std::cerr << "wrong mpi-thread environment provided!\n";
        return -1;
    }
    int periods[2] = {false, true}; //non-, periodic
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices==0)
    {   std::cerr << "No CUDA capable devices found"<<std::endl;
        return -1;
    }
    int device = rank % num_devices; //assume # of gpus/node is fixed
    cudaSetDevice( device);
#endif//cuda
    int np[2];
    if(rank==0)
    {   std::cin>> np[0] >> np[1];
        std::cout << "Computing with "<<np[0]<<" x "<<np[1]<<" = "<<size<<std::endl;
        assert( size == np[0]*np[1]);
    }
    MPI_Bcast( np, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);
    ////////////////////////Parameter initialisation//////////////////////////
    std::string input;
    if( argc != 3)
    {   if(rank==0)std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n";
        return -1;
    }
    else
    {   input = file::read_file( argv[1]);
    }
    Json::Reader reader;
    Json::Value js;
    reader.parse( input, js, false); //read input without comments
    input = js.toStyledString(); //save input without comments, which is important if netcdf file is later read by another parser
    const imp::Parameters p( js);
    if(rank==0)p.display( std::cout);
    ////////////////////////////////set up computations///////////////////////////
    dg::CartesianMPIGrid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y, comm);
    dg::CartesianMPIGrid2d grid_out( 0, p.lx, 0,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y, comm);
    //create RHS
    dg::Diffusion< dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec > diffusion( grid, p);
    dg::ToeflI< dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec > toeflI( grid, p);
    /////////////////////The initial field///////////////////////////////////////////
    dg::Gaussian gaussian( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp); //gaussian width is in absolute values
    std::vector<dg::MDVec> y0(3, dg::evaluate(dg::zero, grid));
    dg::Helmholtz<dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec> & gamma = toeflI.gamma();
    if( p.mode == 1)
    {   if( p.vorticity == 0)
        {   gamma.alpha() = -0.5*p.tau[1];
            y0[0] = dg::evaluate( gaussian, grid);
            dg::blas2::symv( gamma, y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
            dg::MDVec v2d=dg::create::inv_weights(grid);
            dg::blas2::symv( v2d, y0[1], y0[1]);
            dg::blas1::scal( y0[1], 1./p.a[1]); //n_i ~1./a_i n_e
            y0[2] = dg::evaluate( dg::zero, grid);
        }
        else
        {   y0[1] = y0[0] = dg::evaluate( gaussian, grid);
            dg::blas1::scal( y0[1], 1/p.a[1]);
            y0[2] = dg::evaluate( dg::zero, grid);
        }
    }
    if( p.mode == 2)
    {   //init wall in y0[2]
        dg::GaussianX wall( p.wall_pos*p.lx, p.wall_sigma, p.wall_amp);
        dg::MDVec wallv = dg::evaluate( wall, grid);
        gamma.alpha() = -0.5*p.tau[2]*p.mu[2];
        dg::blas2::symv( gamma, wallv, y0[2]);
        dg::MDVec v2d=dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[2], y0[2]);
        if( p.a[2] != 0.)
            dg::blas1::scal( y0[2], 1./p.a[2]); //n_z ~1./a_z

        //init blob in y0[1]
        gamma.alpha() = -0.5*p.tau[1];
        y0[0] = dg::evaluate( gaussian, grid);
        dg::blas2::symv( gamma, y0[0], y0[1]);
        dg::blas1::pointwiseDot( v2d, y0[1], y0[1]);
        if( p.a[2] == 1)
        {   std::cerr << "No blob with trace ions possible!\n";
            return -1;
        }
        dg::blas1::scal( y0[1], 1./p.a[1]); //n_i ~1./a_i n_e

        //sum up
        if( p.a[2] != 0)
            dg::blas1::axpby( 1., wallv, 1., y0[0]); //add wall to blob in n_e
    }
    if( p.mode == 3)
    {   gamma.alpha() = -0.5*p.tau[2]*p.mu[2];
        y0[0] = dg::evaluate( gaussian, grid);
        dg::blas2::symv( gamma, y0[0], y0[2]);
        dg::MDVec v2d=dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[2], y0[2]);
        if( p.a[2] == 0)
        {   std::cerr << "No impurity blob with trace impurities possible!\n";
            return -1;
        }
        dg::blas1::axpby( 1./p.a[2], y0[2], 0., y0[2]); //n_z ~1./a_z n_e
        y0[1] = dg::evaluate( dg::zero, grid);
    }
    std::vector<dg::MDVec> y1( y0);
    //////////////////initialisation of timestepper and first step///////////////////
    if(rank==0)std::cout << "init timestepper...\n";
    double time = 0;
    dg::Karniadakis< std::vector<dg::MDVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    karniadakis.init( toeflI, diffusion, y0, p.dt);
    /////////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    MPI_Info info = MPI_INFO_NULL;
    err = nc_create_par( argv[2], NC_NETCDF4|NC_MPIIO|NC_CLOBBER, comm, info, &ncid); //MPI ON
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    const int version[3] = {FELTOR_MAJOR_VERSION, FELTOR_MINOR_VERSION, FELTOR_SUBMINOR_VERSION};
    err = nc_put_att_int( ncid, NC_GLOBAL, "feltor_major_version",    NC_INT, 1, &version[0]);
    err = nc_put_att_int( ncid, NC_GLOBAL, "feltor_minor_version",    NC_INT, 1, &version[1]);
    err = nc_put_att_int( ncid, NC_GLOBAL, "feltor_subminor_version", NC_INT, 1, &version[2]);
    int dim_ids[3], tvarID;
    err = file::define_dimensions( ncid, dim_ids, &tvarID, grid_out.global());
    //field IDs
    std::string names[5] = {"electrons", "ions", "impurities", "potential", "vorticity"};
    int dataIDs[5];
    for( unsigned i=0; i<5; i++)
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);
    //energy IDs
    int EtimeID, EtimevarID;
    err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, dissID, dEdtID, accuracyID;
    err = nc_def_var( ncid, "energy",      NC_DOUBLE, 1, &EtimeID, &energyID);
    err = nc_def_var( ncid, "mass",        NC_DOUBLE, 1, &EtimeID, &massID);
    err = nc_def_var( ncid, "dissipation", NC_DOUBLE, 1, &EtimeID, &dissID);
    err = nc_def_var( ncid, "dEdt",        NC_DOUBLE, 1, &EtimeID, &dEdtID);
    err = nc_def_var( ncid, "accuracy",    NC_DOUBLE, 1, &EtimeID, &accuracyID);
    for( unsigned i=0; i<5; i++)
        err = nc_var_par_access( ncid, dataIDs[i], NC_COLLECTIVE);
    err = nc_var_par_access( ncid, tvarID,     NC_COLLECTIVE);
    err = nc_var_par_access( ncid, EtimevarID, NC_COLLECTIVE);
    err = nc_var_par_access( ncid, energyID,   NC_COLLECTIVE);
    err = nc_var_par_access( ncid, massID,     NC_COLLECTIVE);
    err = nc_var_par_access( ncid, dissID,     NC_COLLECTIVE);
    err = nc_var_par_access( ncid, dEdtID,     NC_COLLECTIVE);
    err = nc_var_par_access( ncid, accuracyID, NC_COLLECTIVE);
    err = nc_enddef(ncid);
    ///////////////////////////////////first output/////////////////////////
    if(rank==0)std::cout << "First output ... \n";
    int dims[2],  coords[2];
    MPI_Cart_get( comm, 2, dims, periods, coords);
    size_t count[3] = {1, grid_out.n()*grid_out.Ny(), grid_out.n()*grid_out.Nx()};
    size_t start[3] = {0, coords[1]*count[1], coords[0]*count[2]};
    dg::MDVec transfer( dg::evaluate(dg::zero, grid));
    dg::DVec transferD( dg::evaluate(dg::zero, grid_out.local()));
    dg::HVec transferH( dg::evaluate(dg::zero, grid_out.local()));
    dg::IDMatrix interpolate = dg::create::interpolation( grid_out.local(), grid.local()); //create local interpolation matrix
    for( unsigned i=0; i<3; i++)
    {   dg::blas2::gemv( interpolate, y0[i].data(), transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[i], start, count, transferH.data() );
    }
    //Potential
    transfer = toeflI.polarization( y0);
    dg::blas2::gemv( interpolate, transfer.data(), transferD);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[3], start, count, transferH.data() );
    //Vorticity
    dg::blas2::gemv( diffusion.laplacianM(), transfer, y0[1]);
    dg::blas2::gemv( interpolate, y0[1].data(), transferD);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[4], start, count, transferH.data() );
    err = nc_put_vara_double( ncid, tvarID,     start, count, &time);
    err = nc_put_vara_double( ncid, EtimevarID, start, count, &time);
    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    double energy0 = toeflI.energy(), mass0 = toeflI.mass(), E0 = energy0, mass = mass0, E1 = 0.0, dEdt = 0., diss = 0., accuracy=0.;
    err = nc_put_vara_double( ncid, energyID,   Estart, Ecount, &energy0);
    err = nc_put_vara_double( ncid, massID,     Estart, Ecount, &mass0);
    err = nc_put_vara_double( ncid, dissID,     Estart, Ecount,&diss);
    err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount,&dEdt);
    err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount,&accuracy);
    if(rank==0)std::cout << "First write successful!\n";
    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Timer t;
    t.tic();
    try
    {

#ifdef DG_BENCHMARK
        unsigned step = 0;
#endif //DG_BENCHMARK

        for( unsigned i=1; i<=p.maxout; i++)
        {

#ifdef DG_BENCHMARK
            dg::Timer ti;
            ti.tic();
#endif//DG_BENCHMARK
            for( unsigned j=0; j<p.itstp; j++)
            {   karniadakis( toeflI, diffusion, y0);
                y0.swap( y1);
                step++;
                time += p.dt;
                Estart[0] = step;
                E0 = E1;
                E1 = toeflI.energy();
                mass = toeflI.mass();
                dEdt = (E1 - E0)/p.dt;
                diss = toeflI.energy_diffusion( );
                accuracy = 2.*fabs((dEdt-diss)/(dEdt + diss));

                //store accuracy details
                err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
                err = nc_put_vara_double( ncid, energyID,   Estart, Ecount, &E1);
                err = nc_put_vara_double( ncid, massID,     Estart, Ecount, &mass);
                err = nc_put_vara_double( ncid, dissID,     Estart, Ecount, &diss);
                err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount, &dEdt);
                err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount, &accuracy);
                if(rank==0)std::cout << "(m_tot-m_0)/m_0: "<< (mass-mass0)/mass0<<"\t";
                if(rank==0)std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                if(rank==0)std::cout <<" d E/dt = " << dEdt <<" Lambda = " << diss << " -> Accuracy: "<< accuracy << "\n";
            }

#ifdef DG_BENCHMARK
            ti.toc();
            if(rank==0)std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
            if(rank==0)std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s";
            ti.tic();
#endif//DG_BENCHMARK

            //output all three fields
            //////////////////////////write fields////////////////////////
            start[0] = i;
            for( unsigned j=0; j<3; j++)
            {   dg::blas2::gemv( interpolate, y0[j].data(), transferD);
                dg::blas1::transfer( transferD, transferH);
                err = nc_put_vara_double( ncid, dataIDs[j], start, count, transferH.data());
            }
            transfer = toeflI.potential()[0];
            dg::blas2::gemv( interpolate, transfer.data(), transferD);
            dg::blas1::transfer( transferD, transferH);
            err = nc_put_vara_double( ncid, dataIDs[3], start, count, transferH.data() );
            transfer = toeflI.potential()[0];
            dg::blas2::gemv( diffusion.laplacianM(), transfer, y1[1]);        //correct?
            dg::blas2::gemv( interpolate, y1[1].data(), transferD);
            dg::blas1::transfer( transferD, transferH);
            err = nc_put_vara_double( ncid, dataIDs[4], start, count, transferH.data() );
            err = nc_put_vara_double( ncid, tvarID, start, count, &time);
#ifdef DG_BENCHMARK
            ti.toc();
            if(rank==0)std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
        }
    }
    catch( dg::Fail& fail)
    {   if(rank==0)std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
        if(rank==0)std::cerr << "Does Simulation respect CFL condition?\n";
    }
    t.toc();
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    if(rank==0)std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    if(rank==0)std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    if(rank==0)std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";
    nc_close(ncid);
    MPI_Finalize();

    return 0;
}

