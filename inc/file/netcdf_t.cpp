#include <iostream>
#include <string>
#include <netcdf.h>
#include <cmath>

#include "catch2/catch_all.hpp"
#ifdef WITH_MPI
#include <mpi.h>
#include "nc_mpi_file.h"
#endif
#include "nc_file.h"
#include "records.h"

#include "dg/algorithm.h"

static double gradientX(double x, double y, double z){return cos(x)*sin(y)*cos(z);}
static double gradientY(double x, double y, double z){return sin(x)*cos(y)*cos(z);}
static double gradientZ(double x, double y, double z){return -sin(x)*sin(y)*sin(z);}


/// [doxygen]
std::vector<dg::file::Record<void(dg::x::DVec&,const dg::x::Grid3d&,double),
    dg::file::LongNameAttribute>> records = {
    {"vectorX", "X-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid3d& g, double time){
            resultD = dg::evaluate( gradientX, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorY", "Y-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid3d& g, double time){
            resultD = dg::evaluate( gradientY, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorZ", "Z-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid3d& g, double time){
            resultD = dg::evaluate( gradientZ, g);
            dg::blas1::scal( resultD, cos( time));
        }
    }
};
/// [doxygen]

TEST_CASE( "Input Output test of the NcFile class")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    std::vector<int> dims = {0,0,0};
    MPI_Dims_create( size, 3, &dims[0]);
    auto i = GENERATE( 0,1,2,3,4,5);
    std::sort( dims.begin(), dims.end());
    for( int u=0; u<i; u++)
        std::next_permutation( dims.begin(), dims.end());
    INFO( "Permutation of dims "<<dims[0]<<" "<<dims[1]<<" "<<dims[2]);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, dims, {1, 1, 1}, false);
#endif
    INFO( "Write a timedependent scalar, scalar field, and vector field to NetCDF4 file "
                   << "test.nc");
    double Tmax=2.*M_PI;
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::Grid3d grid( x0,x1,x0,x1,x0,x1,3,4,4,3
#ifdef WITH_MPI
    , comm
#endif
    );
    //create NetCDF File
    dg::file::NcFile file("test.nc", dg::file::nc_clobber);
    REQUIRE( file.is_open());
    REQUIRE( std::filesystem::exists( "test.nc"));
    SECTION( "Dimensions from grid")
    {
        file.defput_dim( "x", {{"axis", "X"},
            {"long_name", "x-coordinate in Cartesian system"}},
            grid.abscissas(0));
        file.defput_dim( "y", {{"axis", "Y"},
            {"long_name", "y-coordinate in Cartesian system"}},
            grid.abscissas(1));
        file.defput_dim( "z", {{"axis", "Z"},
            {"long_name", "z-coordinate in Cartesian system"}},
            grid.abscissas(2));
        file.close();
        auto mode = GENERATE( dg::file::nc_nowrite, dg::file::nc_write);
        file.open( "test.nc", mode);
        // This is how to fully check a dimension
        std::map<std::string, int> map {{"x",0}, {"y",1}, {"z", 2}};
        for( auto str : map)
        {
            CHECK( file.dim_is_defined( str.first));
            CHECK( file.var_is_defined( str.first));
            CHECK( file.get_dim_size( str.first) == grid.shape(str.second));
            auto abs = grid.abscissas(str.second) , test(abs);
            file.get_var( str.first, {grid.axis(str.second)}, test);
            dg::blas1::axpby( 1.,abs,-1., test);
            double result = dg::blas1::dot( 1., test);
            CHECK( result == 0);
        }
    }
    SECTION( "Test record variables")
    {
        file.def_dim("y", 2);
        file.def_dim("x", 2);
        for( auto& record : records)
        {
            file.def_var_as<double>( record.name, {"y", "x"},
                record.atts);
        }
        file.get_att_as<std::string>( "long_name", "vectorX");
        for( auto str : {"vectorX", "vectorY", "vectorZ"})
        {
            CHECK(file.var_is_defined( str));
            CHECK(file.att_is_defined( "long_name", str));
            CHECK(file.get_var_dims( str).size() == 2);
        }
    }
    SECTION( "Variables can be written intermittently")
    {
        file.def_dimvar_as<double>( "time", NC_UNLIMITED, {{"axis", "T"}});
        file.def_var_as<double>( "Energy", {"time"}, {{"long_name", "Energy"}});
        for( unsigned i=0; i<=6; i++)
        {
            file.put_var("time", {i}, i);
            if( i%2 == 0)
                file.put_var( "Energy", {i}, i);
        }
        file.close();
        auto mode = GENERATE( dg::file::nc_nowrite, dg::file::nc_write);
        file.open( "test.nc", mode);
        double time, energy;
        for( unsigned i=0; i<=6; i++)
        {
            file.get_var("time", {i}, time);
            CHECK( time == (double)i);
            file.get_var("Energy", {i}, energy);
            if( i%2 == 0)
                CHECK( energy == (double)i);
        }
    }
    SECTION( "Write projected variables")
    {
        file.def_grp( "projected");
        file.set_grp( "projected");
        auto grid_out = grid;
        grid_out.multiplyCellNumbers( 0.5, 0.5);
        file.defput_dim( "xr", {{"axis", "X"},
            {"long_name", "reduced x-coordinate in Cartesian system"}},
            grid_out.abscissas(0));
        file.defput_dim( "yr", {{"axis", "Y"},
            {"long_name", "reduced y-coordinate in Cartesian system"}},
            grid_out.abscissas(1));
        file.defput_dim( "zr", {{"axis", "Z"},
            {"long_name", "reduced z-coordinate in Cartesian system"}},
            grid_out.abscissas(2));
        file.def_dimvar_as<double>( "ptime", NC_UNLIMITED, {{"axis", "T"}});
        for( auto& record : records)
            file.def_var_as<double>( record.name, {"ptime", "zr", "yr", "xr"},
                record.atts);
        dg::MultiMatrix<dg::x::DMatrix, dg::x::DVec> project =
            dg::create::fast_projection( grid, 1, 2, 2);
        dg::x::DVec result = dg::evaluate( dg::zero, grid);
        dg::x::DVec tmp = dg::evaluate( dg::zero, grid_out);

        typename dg::file::NcFile::Hyperslab slab{grid_out};

        for(unsigned i=0; i<2; i++)
        {
            INFO("Write timestep "<<i<<"\n");
            double time = i*Tmax/2.;
            file.put_var( "ptime", {i}, time);
            for( auto& record : records)
            {
                record.function ( result, grid, time);
                dg::apply( project, result, tmp);
                // Hyperslab can be constructed from hyperslab...
                file.put_var( record.name, {i, slab}, tmp);
            }
        }

        file.close();
        auto mode = GENERATE( dg::file::nc_nowrite, dg::file::nc_write);
        INFO("TEST "<<( mode == dg::file::nc_write ? "WRITE" : "READ")<<" OPEN MODE\n");
        file.open( "test.nc", mode);

        file.set_grp( "projected");
        unsigned num_slices = file.get_dim_size("ptime");
        INFO( "Found "<<num_slices<<" timesteps in file");
        CHECK( num_slices == 2);
        auto variables = file.get_var_names();
        for( auto str : {"ptime", "vectorX", "vectorY", "vectorZ"})
            CHECK_NOTHROW( std::find( variables.begin(), variables.end(), str)
                != variables.end());
        // Test: we read what we have written ...
        for(unsigned i=0; i<2; i++)
        {
            INFO("Read timestep "<<i<<"\n");
            double time;
            file.get_var( "ptime", {i}, time);
            CHECK ( time == (double)i*Tmax/2.);

            for( auto& record : records)
            {
                record.function ( result, grid, time);
                dg::apply( project, result, tmp);
                // Hyperslab can be constructed from hyperslab...
                auto test (tmp);
                file.get_var( record.name, {i, slab}, test);
                dg::blas1::axpby( 1.,tmp,-1., test);
                double result = dg::blas1::dot( 1., test);
                CHECK( result == 0);
            }
        }


    }
    file.close();
    DG_RANK0 std::filesystem::remove( "test.nc");
#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD);
#endif

}
