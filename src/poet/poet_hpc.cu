// #define DG_DEBUG
#define SILENT

#include <iostream>
#include <iomanip>
#include <vector>

#ifdef TOEFL_MPI
#include <mpi.h>
#endif //FELTOR_MPI

#include "dg/algorithm.h"
#include "dg/file/file.h"

#include "poet.cuh"
#include "parameters.h"

#ifdef TOEFL_MPI
using HVec = dg::MHVec;
using DVec = dg::MDVec;
using HMatrix = dg::MHMatrix;
using DMatrix = dg::MDMatrix;
using IDMatrix = dg::MIDMatrix;
using IHMatrix = dg::MIHMatrix;
using Geometry = dg::CartesianMPIGrid2d;
using DDiaMatrix =  cusp::dia_matrix<int, dg::get_value_type<DVec>, cusp::device_memory>;
using DCooMatrix =  cusp::coo_matrix<int, dg::get_value_type<DVec>, cusp::device_memory>;
#define MPI_OUT if(rank==0)
#else //TOEFL_MPI
using HVec = dg::HVec;
using DVec = dg::DVec;
using HMatrix = dg::HMatrix;
using DMatrix = dg::DMatrix;
using IDMatrix = dg::IDMatrix;
using IHMatrix = dg::IHMatrix;
using Geometry = dg::CartesianGrid2d;
using DDiaMatrix =  cusp::dia_matrix<int, dg::get_value_type<DVec>, cusp::device_memory>;
using DCooMatrix =  cusp::coo_matrix<int, dg::get_value_type<DVec>, cusp::device_memory>;
#define MPI_OUT
#endif //TOEFL_MPI

struct ShearLayer{

    ShearLayer( double rho, double delta, double lx, double ly): m_rho(rho), m_delta(delta), m_lx(lx), m_ly(ly) {}
    DG_DEVICE
    double operator()(double x, double y) const{
    if( x<= m_lx/2.)
        return m_delta*cos(2.*M_PI*y/m_ly) - 1./m_rho/cosh( (2.*M_PI*x/m_lx-M_PI/2.)/m_rho)/cosh( (2.*M_PI*x/m_lx-M_PI/2.)/m_rho);
    return m_delta*cos(2.*M_PI*y/m_ly) + 1./m_rho/cosh( (3.*M_PI/2.-2.*M_PI*x/m_lx)/m_rho)/cosh( (3.*M_PI/2.-2.*M_PI*x/m_lx)/m_rho);
    }
    private:
    double m_rho, m_delta, m_lx, m_ly;    
};


int main( int argc, char* argv[])
{
#ifdef TOEFL_MPI
    ////////////////////////////////setup MPI///////////////////////////////
    int provided;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if( provided != MPI_THREAD_FUNNELED)
    {
        std::cerr << "wrong mpi-thread environment provided!\n";
        return -1;
    }
    int periods[2] = {false, true}; //non-, periodic
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices==0){std::cerr << "No CUDA capable devices found"<<std::endl; return -1;}
    int device = rank % num_devices; //assume # of gpus/node is fixed
    cudaSetDevice( device);
#endif//cuda
    int np[2];
    if(rank==0)
    {
        std::cin>> np[0] >> np[1];
        std::cout << "Computing with "<<np[0]<<" x "<<np[1]<<" = "<<size<<std::endl;
        assert( size == np[0]*np[1]);
    }
    MPI_Bcast( np, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);
#endif//TOEFL_MPI
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Value js;
    if( argc != 3)
    {
        MPI_OUT std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n";
        return -1;
    }
    else
        dg::file::file2Json( argv[1], js, dg::file::comments::are_forbidden);
    MPI_OUT std::cout << js<<std::endl;
    const Parameters p( js);
    MPI_OUT p.display( std::cout);

    ////////////////////////////////set up computations///////////////////////////
    Geometry grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y
        #ifdef TOEFL_MPI
        , comm
        #endif //TOEFL_MPI
    );
    Geometry grid_out( 0, p.lx, 0, p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y
        #ifdef TOEFL_MPI
        , comm
        #endif //TOEFL_MPI
    );
    //create RHS
    poet::Explicit< Geometry, DMatrix, DDiaMatrix, DCooMatrix, DVec > ex( grid, p);
    poet::Implicit< Geometry, DMatrix, DVec > imp( grid, p.nu);
    /////////////////////create initial vector////////////////////////////////////
//     dg::Gaussian g( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);
//     std::vector<DVec> y0(2, dg::evaluate( g, grid)), y1(y0); // n_e' = gaussian
// //     ex.gamma1_y(y0[0],y0[1]);
//     ex.gamma1inv_y(y0[0],y0[1]);
    
      ShearLayer layer(M_PI/15., 0.05, p.lx, p.ly); //shear layer
    std::vector<DVec> y0(2, dg::evaluate( layer, grid)), y1(y0);
    dg::blas1::scal(y0[0], p.amp);
    ex.invLap_y(y0[0], y1[0]); //phi 
    dg::blas1::scal(y0[0], 0.);
    ex.solve_Ni_lwl(y0[0], y1[0], y0[1]);  
        {
                size_t start = 0;
        dg::file::NC_Error_Handle err;
        int ncid;
        err = nc_create( "visual.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
        int dim_ids[3], tvarID;
        err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid);

        std::string names[3] = {"ne", "Ni", "phi"};
        int dataIDs[3];
        for( unsigned i=0; i<3; i++){
            err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

        dg::HVec transferH(dg::evaluate(dg::zero, grid));

        dg::blas1::transfer( y0[0], transferH);
        dg::file::put_vara_double( ncid, dataIDs[0], start, grid, transferH);
        dg::blas1::transfer( y0[1], transferH);
        dg::file::put_vara_double( ncid, dataIDs[1], start, grid, transferH);
        dg::blas1::transfer( y1[0], transferH);
        dg::file::put_vara_double( ncid, dataIDs[2], start, grid, transferH);
        err = nc_close(ncid); 
    }
    //////////////////initialisation of timekarniadakis and first step///////////////////
    double time = 0;
    dg::Karniadakis< std::vector<DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    karniadakis.init( ex, imp, time, y0, p.dt);
    y1 = y0;
    /////////////////////////////set up netcdf/////////////////////////////////////
    dg::file::NC_Error_Handle err;
    int ncid;
    MPI_OUT err = nc_create( argv[2],NC_NETCDF4|NC_CLOBBER, &ncid);
    std::string input = js.toStyledString();
    MPI_OUT err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    int dim_ids[3], tvarID;
    MPI_OUT err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid_out);
    //field IDs
    std::string names[4] = {"electrons", "ions", "potential", "vorticity"};
    int dataIDs[4];
    for( unsigned i=0; i<4; i++){
        MPI_OUT err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

    //energy IDs
    int EtimeID, EtimevarID;
    MPI_OUT err = dg::file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, dissID, dEdtID;
    MPI_OUT err = nc_def_var( ncid, "energy",      NC_DOUBLE, 1, &EtimeID, &energyID);
    MPI_OUT err = nc_def_var( ncid, "mass",        NC_DOUBLE, 1, &EtimeID, &massID);
    MPI_OUT err = nc_def_var( ncid, "dissipation", NC_DOUBLE, 1, &EtimeID, &dissID);
    MPI_OUT err = nc_def_var( ncid, "dEdt",        NC_DOUBLE, 1, &EtimeID, &dEdtID);
    MPI_OUT err = nc_enddef(ncid);
    DVec transfer( dg::evaluate( dg::zero, grid));
    ///////////////////////////////////first output/////////////////////////
    size_t start = 0, count = 1;
    size_t Ecount[] = {1};
    size_t Estart[] = {0};
    std::vector<DVec> transferD(4, dg::evaluate(dg::zero, grid_out));
    HVec transferH(dg::evaluate(dg::zero, grid_out));
    IDMatrix interpolate = dg::create::interpolation( grid_out, grid);
    dg::blas2::symv( interpolate, y1[0], transferD[0]);
    dg::blas2::symv( interpolate, y1[1], transferD[1]);
    dg::blas2::symv( interpolate, ex.potential()[0], transferD[2]);
    dg::blas2::symv( imp.laplacianM(), ex.potential()[0], transfer);
    dg::blas2::symv( interpolate, transfer, transferD[3]);
    for( int k=0;k<4; k++)
    {
        dg::blas1::transfer( transferD[k], transferH);
        dg::file::put_vara_double( ncid, dataIDs[k], start, grid_out, transferH);
    }
    MPI_OUT err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
    MPI_OUT err = nc_close(ncid);
    ///////////////////////////////////////Timeloop/////////////////////////////////
    const double mass0 = ex.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = ex.energy(), E1 = 0, diff = 0;
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
            karniadakis.step( ex, imp, time, y1);
            //store accuracy details
            {
                MPI_OUT std::cout << "(m_tot-m_0)/m_0: "<< (ex.mass()-mass0)/mass_blob0<<"\t";
                E0 = E1;
                E1 = ex.energy();
                diff = (E1 - E0)/p.dt;
                double diss = ex.energy_diffusion( );
                MPI_OUT std::cout << "diff: "<< diff<<" diss: "<<diss<<"\t";
                MPI_OUT std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";
            }
            Estart[0] += 1;
            {
                MPI_OUT err = nc_open(argv[2], NC_WRITE, &ncid);
                double ener=ex.energy(), mass=ex.mass(), diff=ex.mass_diffusion(), dEdt=ex.energy_diffusion();
                MPI_OUT err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
                MPI_OUT err = nc_put_vara_double( ncid, energyID,   Estart, Ecount, &ener);
                MPI_OUT err = nc_put_vara_double( ncid, massID,     Estart, Ecount, &mass);
                MPI_OUT err = nc_put_vara_double( ncid, dissID,     Estart, Ecount, &diff);
                MPI_OUT err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount, &dEdt);
                MPI_OUT err = nc_close(ncid);
            }
        }
        //////////////////////////write fields////////////////////////
        start = i;
        dg::blas2::symv( interpolate, y1[0], transferD[0]);
        dg::blas2::symv( interpolate, y1[1], transferD[1]);
        dg::blas2::symv( interpolate, ex.potential()[0], transferD[2]);
        dg::blas2::symv( imp.laplacianM(), ex.potential()[0], transfer);
        dg::blas2::symv( interpolate, transfer, transferD[3]);
        MPI_OUT err = nc_open(argv[2], NC_WRITE, &ncid);
        for( int k=0;k<4; k++)
        {
            dg::blas1::transfer( transferD[k], transferH);
            dg::file::put_vara_double( ncid, dataIDs[k], start, grid_out, transferH);
        }
        MPI_OUT err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
        MPI_OUT err = nc_close(ncid);

#ifdef DG_BENCHMARK
        ti.toc();
        step+=p.itstp;
        MPI_OUT std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        MPI_OUT std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
    }
    }
    catch( dg::Fail& fail) {
        MPI_OUT std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
        MPI_OUT std::cerr << "Does Simulation respect CFL condition?\n";
    }
    t.toc();
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    MPI_OUT std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    MPI_OUT std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    MPI_OUT std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";
#ifdef TOEFL_MPI
    MPI_Finalize();
#endif //TOEFL_MPI

    return 0;

}
