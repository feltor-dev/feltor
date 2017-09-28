#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits.h>  // UINT_MAX is needed in cusp (v0.5.1) but limits.h is not included
#include <mpi.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "dg/backend/timer.cuh"
#include "dg/backend/evaluation.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/runge_kutta.h"
#include "dg/multistep.h"
#include "dg/helmholtz.h"
#include "dg/backend/typedefs.cuh"
#include "dg/functors.h"

#include "geometries/geometries.h"

#include "ns.h"
#include "parameters.h"

using namespace std;
using namespace dg;

#ifdef LOG_POLAR
    typedef dg::geo::LogPolarGenerator Generator;
#else
    typedef dg::geo::PolarGenerator Generator;
#endif

int main(int argc, char* argv[])
{
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

    ////Parameter initialisation ////////////////////////////////////////////
    Json::Reader reader;
    Json::Value js;
    if( argc == 1)
    {
        std::ifstream is("input.json");
        reader.parse(is,js,false);
    }
    else if( argc == 2)
    {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const Parameters p( js);
    if(rank==0)
        p.display( std::cout);

    int np[2];
    np[0] = js["mpi_x"].asUInt();
    np[1]  = js["mpi_y"].asUInt();


#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices==0){std::cerr << "No CUDA capable devices found"<<std::endl; return -1;}
    int device = rank % num_devices; //assume # of gpus/node is fixed
    cudaSetDevice( device);
#endif//cuda
    if(rank==0)
    {
        std::cout << "Computing with "<<np[0]<<" x "<<np[1]<<" = "<<size<<std::endl;
        assert( size == np[0]*np[1]);
    }
    MPI_Bcast( np, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);

    Timer t;

    //Grid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    Generator generator(p.r_min, p.r_max); // Generator is defined by the compiler
    dg::geo::CurvilinearMPIGrid2d grid( generator, p.n, p.Nx, p.Ny, dg::DIR, dg::PER, comm); 

    MDVec w2d( create::volume(grid));

    dg::Lamb lamb( p.posX, p.posY, p.R, p.U);
    MHVec omega = evaluate ( lamb, grid);
#if LOG_POLAR
    MDVec stencil = evaluate( one, grid);
#else
    MDVec stencil = evaluate( LinearX(1.0, p.r_min), grid);
#endif
    MDVec y0( omega ), y1( y0);

    //make solver and stepper
    polar::Explicit<aMPIGeometry2d, MDMatrix, MDVec> shu( grid, p.eps);
    polar::Diffusion<aMPIGeometry2d, MDMatrix, MDVec> diffusion( grid, p.nu);
    Karniadakis< MDVec > ab( y0, y0.size(), p.eps_time);

    t.tic();
    shu( y0, y1);
    t.toc();
    if(rank == 0)
        cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";

    double vorticity = blas2::dot( stencil , w2d, y0);
    MDVec ry0(stencil);
    blas1::pointwiseDot( stencil, y0, ry0);
    double enstrophy = 0.5*blas2::dot( ry0, w2d, y0);
    double energy =    0.5*blas2::dot( ry0, w2d, shu.potential()) ;
    if(rank==0) {
        cout << "Total vorticity:  "<<vorticity<<"\n";
        cout << "Total enstrophy:  "<<enstrophy<<"\n";
        cout << "Total energy:     "<<energy<<"\n";
    }

    double time = 0;
    ab.init( shu, diffusion, y0, p.dt);
    ab( shu, diffusion, y0); //make potential ready

    t.tic();
    while (time < p.maxout*p.itstp*p.dt)
    {
        //step 
        for( unsigned i=0; i<p.itstp; i++)
        {
            ab( shu, diffusion, y0 );
        }
        time += p.itstp*p.dt;

    }
    t.toc();

    double vorticity_end = blas2::dot( stencil , w2d, ab.last());
    blas1::pointwiseDot( stencil, ab.last(), ry0);
    double enstrophy_end = 0.5*blas2::dot( ry0, w2d, ab.last());
    double energy_end    = 0.5*blas2::dot( ry0, w2d, shu.potential()) ;
    if(rank == 0) {
        cout << "Vorticity error           :  "<<vorticity_end-vorticity<<"\n";
        cout << "Enstrophy error (relative):  "<<(enstrophy_end-enstrophy)/enstrophy<<"\n";
        cout << "Energy error    (relative):  "<<(energy_end-energy)/energy<<"\n";

        cout << "Runtime: " << t.diff() << endl;
    }

    MPI_Finalize();
    return 0;
}
