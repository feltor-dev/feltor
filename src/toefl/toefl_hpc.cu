#include <iostream>
#include <iomanip>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif //FELTOR_MPI

#include "dg/algorithm.h"
#include "dg/file/file.h"

#include "toeflR.cuh"
#include "parameters.h"

int main( int argc, char* argv[])
{
#ifdef WITH_MPI
    ////////////////////////////////setup MPI///////////////////////////////
    dg::mpi_init( argc, argv);
    MPI_Comm comm;
    dg::mpi_init2d( dg::DIR, dg::PER, comm, std::cin, true);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif//WITH_MPI
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Value js;
    if( argc != 3)
    {
        DG_RANK0 std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n";
        return -1;
    }
    else
        dg::file::file2Json( argv[1], js, dg::file::comments::are_forbidden);
    DG_RANK0 std::cout << js<<std::endl;
    const Parameters p( js);
    DG_RANK0 p.display( std::cout);

    ////////////////////////////////set up computations///////////////////////////
    dg::x::CartesianGrid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
    );
    dg::x::CartesianGrid2d grid_out( 0, p.lx, 0, p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
    );
    //create RHS
    toefl::Explicit< dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec > exp( grid, p);
    toefl::Implicit< dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec > imp( grid, p.nu);
    /////////////////////create initial vector////////////////////////////////////
    dg::Gaussian g( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);
    std::vector<dg::x::DVec> y0(2, dg::evaluate( g, grid)); // n_e' = gaussian
    dg::blas2::symv( exp.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    if( p.equations == "gravity_local" || p.equations == "gravity_global" || p.equations == "drift_global"){
        y0[1] = dg::evaluate( dg::zero, grid);
    }
    //////////////////initialisation of timekarniadakis and first step///////////////////
    double time = 0;
    dg::DefaultSolver<std::vector<dg::x::DVec>> solver( imp, y0, 1000, p.eps_time);
    dg::Adaptive<dg::ARKStep<std::vector<dg::x::DVec>>> stepper( "ARK-4-2-3", y0);
    //dg::Adaptive<dg::ERKStep<std::vector<dg::x::DVec>>> stepper( "ARK-4-2-3 (explicit)", y0);
    /////////////////////////////set up netcdf/////////////////////////////////////
    dg::file::NC_Error_Handle err;
    int ncid;
    DG_RANK0 err = nc_create( argv[2],NC_NETCDF4|NC_CLOBBER, &ncid);
    std::string input = js.toStyledString();
    DG_RANK0 err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    int dim_ids[3], tvarID;
    DG_RANK0 err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid_out);
    //field IDs
    std::string names[4] = {"electrons", "ions", "potential", "vorticity"};
    int dataIDs[4];
    for( unsigned i=0; i<4; i++){
        DG_RANK0 err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

    //energy IDs
    int EtimeID, EtimevarID;
    DG_RANK0 err = dg::file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, dissID, dEdtID;
    DG_RANK0 err = nc_def_var( ncid, "energy",      NC_DOUBLE, 1, &EtimeID, &energyID);
    DG_RANK0 err = nc_def_var( ncid, "mass",        NC_DOUBLE, 1, &EtimeID, &massID);
    DG_RANK0 err = nc_def_var( ncid, "dissipation", NC_DOUBLE, 1, &EtimeID, &dissID);
    DG_RANK0 err = nc_def_var( ncid, "dEdt",        NC_DOUBLE, 1, &EtimeID, &dEdtID);
    DG_RANK0 err = nc_enddef(ncid);
    dg::x::DVec transfer( dg::evaluate( dg::zero, grid));
    ///////////////////////////////////first output/////////////////////////
    size_t start = 0, count = 1;
    size_t Ecount[] = {1};
    size_t Estart[] = {0};
    std::vector<dg::x::DVec> transferD(4, dg::evaluate(dg::zero, grid_out));
    dg::x::HVec transferH(dg::evaluate(dg::zero, grid_out));
    dg::x::IDMatrix interpolate = dg::create::interpolation( grid_out, grid);
    dg::blas2::symv( interpolate, y0[0], transferD[0]);
    dg::blas2::symv( interpolate, y0[1], transferD[1]);
    dg::blas2::symv( interpolate, exp.potential()[0], transferD[2]);
    dg::blas2::symv( imp.laplacianM(), exp.potential()[0], transfer);
    dg::blas2::symv( interpolate, transfer, transferD[3]);
    for( int k=0;k<4; k++)
    {
        dg::assign( transferD[k], transferH);
        dg::file::put_vara_double( ncid, dataIDs[k], start, grid_out, transferH);
    }
    DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
    DG_RANK0 err = nc_close(ncid);
    ///////////////////////////////////////Timeloop/////////////////////////////////
    const double mass0 = exp.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = exp.energy(), E1 = 0, diff = 0;
    unsigned failed_counter = 0;
    double dt = 1e-6;
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
        {
            stepper.step( std::tie(exp, imp, solver), time, y0, time, y0, dt, dg::pid_control, dg::l2norm, 1e-5, 1e-10);
            //stepper.step( exp, time, y0, time, y0, dt, dg::pid_control, dg::l2norm, 1e-5, 1e-10);
            if ( stepper.failed() ) failed_counter ++;
            DG_RANK0 std::cout << "Time "<<time<<" dt "<<dt<<" failed counter "<<failed_counter<<"\n";
            //store accuracy details
            {
                DG_RANK0 std::cout << "(m_tot-m_0)/m_0: "<< (exp.mass()-mass0)/mass_blob0<<"\t";
                E0 = E1;
                E1 = exp.energy();
                diff = (E1 - E0)/p.dt;
                double diss = exp.energy_diffusion( );
                DG_RANK0 std::cout << "diff: "<< diff<<" diss: "<<diss<<"\t";
                DG_RANK0 std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";
            }
            Estart[0] += 1;
            {
                DG_RANK0 err = nc_open(argv[2], NC_WRITE, &ncid);
                double ener=exp.energy(), mass=exp.mass(), diff=exp.mass_diffusion(), dEdt=exp.energy_diffusion();
                DG_RANK0 err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
                DG_RANK0 err = nc_put_vara_double( ncid, energyID,   Estart, Ecount, &ener);
                DG_RANK0 err = nc_put_vara_double( ncid, massID,     Estart, Ecount, &mass);
                DG_RANK0 err = nc_put_vara_double( ncid, dissID,     Estart, Ecount, &diff);
                DG_RANK0 err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount, &dEdt);
                DG_RANK0 err = nc_close(ncid);
            }
        }
        //////////////////////////write fields////////////////////////
        start = i;
        dg::blas2::symv( interpolate, y0[0], transferD[0]);
        dg::blas2::symv( interpolate, y0[1], transferD[1]);
        dg::blas2::symv( interpolate, exp.potential()[0], transferD[2]);
        dg::blas2::symv( imp.laplacianM(), exp.potential()[0], transfer);
        dg::blas2::symv( interpolate, transfer, transferD[3]);
        DG_RANK0 err = nc_open(argv[2], NC_WRITE, &ncid);
        for( int k=0;k<4; k++)
        {
            dg::assign( transferD[k], transferH);
            dg::file::put_vara_double( ncid, dataIDs[k], start, grid_out, transferH);
        }
        DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
        DG_RANK0 err = nc_close(ncid);

#ifdef DG_BENCHMARK
        ti.toc();
        step+=p.itstp;
        DG_RANK0 std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        DG_RANK0 std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
    }
    }
    catch( dg::Fail& fail) {
        DG_RANK0 std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
        DG_RANK0 std::cerr << "Does Simulation respect CFL condition?\n";
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

