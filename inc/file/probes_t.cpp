#include <iostream>

#include "catch2/catch_all.hpp"
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include "dg/algorithm.h"

#define _FILE_INCLUDED_BY_DG_
#include "json_probes.h"
#include "probes.h"

static double function( double x, double y){return sin(x)*sin(y);}
static double gradientX(double x, double y){return cos(x)*sin(y);}
static double gradientY(double x, double y){return sin(x)*cos(y);}
static double cosine( double x, double y){return cos(x)*cos(y);}

static std::vector<dg::file::Record<void(dg::x::DVec&,const dg::x::Grid2d&,double)>> records_list = {
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

static std::vector<dg::file::Record<void( dg::x::HVec&, const dg::x::Grid2d&),
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

// TODO More precise test
TEST_CASE( "Probes")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1, 1});
#endif
    auto js_direct = dg::file::file2Json("probes_direct.json");
    auto params = dg::file::parse_probes( js_direct);

    INFO("Write a time-dependent vector field and probe data to netcdf4 file "
                       << "probes.nc");
    dg::file::NcFile file( "probes.nc", dg::file::nc_clobber);
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::Grid2d grid( x0,x1,x0,x1,3,100,100, dg::PER, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    file.def_dimvar_as<double>( "time", NC_UNLIMITED, {{"axis", "T"}});
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
            INFO("Write timestep "<<i);
            probes.write( time, records_list, grid, time);
        }
        else
        {
            if( i % 2)
            {
                INFO("Buffer timestep "<<i);
                probes.buffer( time, records_list, grid, time);
            }
            else
            {
                INFO("Buffer & Flush timestep "<<i);
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
                file.put_atts( record.atts, record.name);
            }
            file.put_var( record.name, {i,grid}, resultD);
        }
        //write time
        file.put_var( "time", {i}, time);
    }


    file.close();
}
