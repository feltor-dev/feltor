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

TEST_CASE( "Probes")
{
    using namespace Catch::Matchers;
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1, 1});
#endif
    std::string json = R"unique({
        "probes":
        {
            "input" : "coords",
            "coords" :
            {
                "coords-names" : ["x", "y"],
                "format" : {},
                "x" : [0,1,2,3],
                "y" : [1,2,3,4]
            }
        }
    })unique";
    auto js_direct = dg::file::string2Json(json);
    auto params = dg::file::parse_probes( js_direct);

    INFO("Write a time-dependent vector field and probe data to netcdf4 file "
                       << "probes.nc");
    dg::file::NcFile file( "probes.nc", dg::file::nc_clobber);
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::Grid2d grid( x0,x1,x0,x1,5,25,25, dg::PER, dg::PER
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
    file.open( "probes.nc", dg::file::nc_nowrite);
    REQUIRE(file.grp_is_defined( "probes"));
    file.set_grp( "probes");
    CHECK( file.att_is_defined( "format"));
    std::list<std::string> names  = {"x", "y", "vectorX", "vectorY", "Sine", "Cosine"};
    for( auto name : names)
    {
        INFO( "Checking "<<name);
        CHECK( file.var_is_defined( name));
        CHECK( file.att_is_defined( "long_name", name));
    }
    auto dims = file.get_var_dims( "x");
    CHECK( file.get_dim_size( dims[0]) == 4);
    CHECK( file.dim_is_defined( "ptime"));
    CHECK( file.get_dim_size( "ptime") == NT+1);
    double point, x, y;
    for( unsigned u=0; u<4; u++)
    {
        file.get_var("x", {u}, x);
        CHECK( x == u);
        file.get_var("y", {u}, y);
        CHECK( y == u+1);

        file.get_var("Sine", {u}, point);
        INFO( "Sine "<<point<<" "<<sin(x)*sin(y)<<" "<<point - sin(x)*sin(y));
        CHECK( fabs( point - sin(x)*sin(y) ) < 1e-7);

        file.get_var("Cosine", {u}, point);
        INFO( "Cosine "<<point<<" "<<cos(x)*cos(y)<<" "<<point - cos(x)*cos(y));
        CHECK_THAT( point, WithinAbs( cos(x)*cos(y) , 1e-7));
        for( unsigned k=0; k<NT; k++)
        {
            double time = k*dt;
            file.get_var( "ptime", {k}, point);
            INFO( "Time "<<time<<" "<<point);
            CHECK( time == point);
            file.get_var("vectorX", {k,u}, point);
            CHECK_THAT( point, WithinAbs( gradientX(x,y)*cos(time), 1e-7));
            file.get_var("vectorY", {k,u}, point);
            CHECK_THAT( point, WithinAbs( gradientY(x,y)*cos(time), 1e-7));
        }
    }

    file.close();
    DG_RANK0 std::filesystem::remove( "probes.nc");
}
