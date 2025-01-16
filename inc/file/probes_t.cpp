#include <iostream>

#ifdef WITH_MPI
#include <mpi.h>
#endif
#include "dg/algorithm.h"

#define _FILE_INCLUDED_BY_DG_
#include "json_probes.h"
#include "probes.h"

double function( double x, double y){return sin(x)*sin(y);}
double gradientX(double x, double y){return cos(x)*sin(y);}
double gradientY(double x, double y){return sin(x)*cos(y);}
double cosine( double x, double y){return cos(x)*cos(y);}

std::vector<dg::file::Record<void(dg::x::DVec&,const dg::x::Grid2d&,double)>> records_list = {
    {"vectorX", {{"long_name", "X-component of vector"}, {"units", "rho_s"}},
        [] ( dg::x::DVec& resultD, const dg::x::Grid2d& g, double time){
            resultD = dg::evaluate( gradientX, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorY", {{"long_name", "Y-component of vector"}, {"units", "rho_s"}},
        [] ( dg::x::DVec& resultD, const dg::x::Grid2d& g, double time){
            resultD = dg::evaluate( gradientY, g);
            dg::blas1::scal( resultD, cos( time));
        }
    }
};

std::vector<dg::file::Record<void( dg::x::HVec&, const dg::x::Grid2d&),
    dg::file::LongNameAttribute>> records_static_list = {
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
    int dims[2] = {0,0};
    MPI_Dims_create( size, 2, dims);
    std::stringstream ss;
    ss<< dims[0]<<" "<<dims[1];
    dg::mpi_init2d( dg::PER, dg::PER, comm, ss);
#endif
    auto js_direct = dg::file::file2Json("probes_direct.json");
    auto params = dg::file::parse_probes( js_direct);

#ifdef WITH_MPI
    std::string filename = "probesmpi.nc";
#else
    std::string filename = "probes.nc";
#endif
    DG_RANK0 std::cout << "WRITE A TIMEDEPENDENT VECTOR FIELD AND PROBE DATA TO NETCDF4 FILE "
                       << filename<<"\n";
    dg::file::NcFile file( filename, dg::file::nc_clobber);
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::Grid2d grid( x0,x1,x0,x1,3,100,100, dg::PER, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    file.defput_dim_as<double>( "time", NC_UNLIMITED, {{"axis", "T"}});
    file.defput_dim( "y", {{"axis", "Y"},
        {"long_name", "y-coordinate in Cartesian system"}}, grid.abscissas(1));
    file.defput_dim( "x", {{"axis", "X"},
        {"long_name", "x-coordinate in Cartesian system"}}, grid.abscissas(0));

    dg::x::DVec resultD = dg::evaluate( dg::zero, grid);
    dg::file::Probes probes( file, grid, params);
    probes.static_write( records_static_list, grid);

    double Tmax=2.*M_PI;
    double NT = 10;
    double dt = Tmax/NT;
    double time = 0;
    for(unsigned i=0; i<=NT; i++)
    {
        time = i*dt;
        if( i <= 3)
        {
            DG_RANK0 std::cout<<"Write timestep "<<i<<"\n";
            probes.write( time, records_list, grid, time);
        }
        else
        {
            if( i % 2)
            {
                DG_RANK0 std::cout<<"Buffer timestep "<<i<<"\n";
                probes.buffer( time, records_list, grid, time);
            }
            else
            {
                DG_RANK0 std::cout<<"Buffer & Flush timestep "<<i<<"\n";
                probes.buffer( time, records_list, grid, time);
                probes.flush();
            }
        }
        //write vector field
        for( auto& record : records_list)
        {
            record.function ( resultD, grid, time);
            if( i==0)
            {
                file.def_var( record.name, NC_DOUBLE, {"time", "y", "x"});
                file.put_atts( record.name, record.atts);
            }
            file.put_var( record.name, {i,grid}, resultD);
        }
        //write time
        file.put_var( "time", {i}, time);
    }


    file.close();
#ifdef WITH_MPI
    MPI_Finalize();
#endif

    return 0;
}
