#include <iostream>

#ifdef WITH_MPI
#include <mpi.h>
#endif
#include "dg/algorithm.h"

#include "json_probes.h"
#include "probes.h"

double function( double x, double y){return sin(x)*sin(y);}
double gradientX(double x, double y){return cos(x)*sin(y);}
double gradientY(double x, double y){return sin(x)*cos(y);}
double cosine( double x, double y){return cos(x)*cos(y);}

struct Record
{
    std::string name;
    std::string long_name;
    std::function<void (dg::x::DVec&, const dg::x::Grid2d&, double)> function;
};
struct StaticRecord
{
    std::string name;
    std::string long_name;
    std::function<void (dg::x::HVec&, const dg::x::Grid2d&)> function;
};

std::vector<Record> records_list = {
    {"vectorX", "X-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid2d& g, double time){
            resultD = dg::evaluate( gradientX, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorY", "Y-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid2d& g, double time){
            resultD = dg::evaluate( gradientY, g);
            dg::blas1::scal( resultD, cos( time));
        }
    }
};

std::vector<StaticRecord> records_static_list = {
    {"Sine", "A Sine function",
        [] ( dg::x::HVec& resultH, const dg::x::Grid2d& g){
            resultH = dg::evaluate( function, g);
        }
    },
    {"Cosine", "A Cosine function",
        [] ( dg::x::HVec& resultH, const dg::x::Grid2d& g){
            resultH = dg::evaluate( cosine, g);
        }
    }
};

int main(int argc, char* argv[])
{
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm;
    //create a grid and some data
    if( size != 4){ std::cerr << "Please run with 4 threads!\n"; return -1;}
    std::stringstream ss;
    ss<< "2 2";
    dg::mpi_init2d( dg::PER, dg::PER, comm, ss);
#endif
    auto js_direct = dg::file::file2Json("probes_direct.json");
    auto params = dg::file::parse_probes( js_direct);

    int ncid=0;
    dg::file::NC_Error_Handle err;
    DG_RANK0 err = nc_create( "probes.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::Grid2d grid( x0,x1,x0,x1,3,100,100, dg::PER, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    int dim_ids[3], tvarID;
    // This caught an error in define_dimensions
    DG_RANK0 err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid);

    dg::file::Probes probes( ncid, 5, grid, params, records_list);
    probes.static_write( records_static_list, grid);

    double Tmax=2.*M_PI;
    double NT = 10;
    double dt = Tmax/NT;
    double time = 0;
    for(unsigned i=0; i<=NT; i++)
    {
        DG_RANK0 std::cout<<"Write timestep "<<i<<"\n";
        time = i*dt;
        probes.write( time, records_list, grid, time);
    }

#ifdef WITH_MPI
    MPI_Finalize();
#endif

    return 0;
}
