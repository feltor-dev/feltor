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

static double function( double x, double y){ return sin(x)*sin(y);}
static double gradientX(double x, double y, double z){return cos(x)*sin(y)*cos(z);}
static double gradientY(double x, double y, double z){return sin(x)*cos(y)*cos(z);}
static double gradientZ(double x, double y, double z){return -sin(x)*sin(y)*sin(z);}


/// [record]
std::vector<dg::file::Record<void(dg::x::DVec&,const dg::x::CartesianGrid3d&,double),
    dg::file::LongNameAttribute>> records = {
    {"vectorX", "X-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::CartesianGrid3d& g, double time){
            resultD = dg::evaluate( gradientX, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorY", "Y-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::CartesianGrid3d& g, double time){
            resultD = dg::evaluate( gradientY, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorZ", "Z-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::CartesianGrid3d& g, double time){
            resultD = dg::evaluate( gradientZ, g);
            dg::blas1::scal( resultD, cos( time));
        }
    }
};
/// [record]

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
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, dims, {1, 1, 1});
#endif
    INFO( "Write a timedependent scalar, scalar field, and vector field to NetCDF4 file "
                   << "inout.nc");
    double Tmax=2.*M_PI;
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::CartesianGrid3d grid( x0,x1,x0,x1,x0,x1,3,4,4,3
#ifdef WITH_MPI
    , comm
#endif
    );
    //create NetCDF File
    dg::file::NcFile file("inout.nc", dg::file::nc_clobber);
    dg::file::NcFile read{}; // MW: Catch a bug in MPI when get_var does not set value_type before get on AnyVector
    REQUIRE( file.is_open());
    REQUIRE( std::filesystem::exists( "inout.nc"));
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
        read.open( "inout.nc", mode);
        //! [check_dim]
        // This is how to fully check a dimension
        std::map<std::string, int> map {{"x",0}, {"y",1}, {"z", 2}};
        for( auto str : map)
        {
            CHECK( read.dim_is_defined( str.first));
            CHECK( read.var_is_defined( str.first));
            CHECK( read.get_dim_size( str.first) == grid.shape(str.second));
            auto abs = grid.abscissas(str.second) , test(abs);
            // In MPI when mode == nc_write only the group containing rank 0 in file comm reads
            read.get_var( str.first, {grid.axis(str.second)}, test);
            dg::blas1::axpby( 1.,abs,-1., test);
            double result = sqrt(dg::blas1::dot( test, test));
            CHECK( result < 1e-15);
        }
        read.close();
        //! [check_dim]
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
        // ! [def_dimvar]
        file.def_dimvar_as<double>( "time", NC_UNLIMITED, {{"axis", "T"}});
        file.def_var_as<double>( "Energy", {"time"}, {{"long_name", "Energy"}});
        for( unsigned i=0; i<=6; i++)
        {
            file.put_var("time", {i}, i);
            if( i%2 == 0)
                file.put_var( "Energy", {i}, i);
        }
        // ! [def_dimvar]
        file.close();
        auto mode = GENERATE( dg::file::nc_nowrite, dg::file::nc_write);
        file.open( "inout.nc", mode);
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

        //! [put_var]
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
        //! [put_var]

        file.close();
        auto mode = GENERATE( dg::file::nc_nowrite, dg::file::nc_write);
        INFO("TEST "<<( mode == dg::file::nc_write ? "WRITE" : "READ")<<" OPEN MODE\n");
        file.open( "inout.nc", mode);

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
                dg::x::DVec data;
                // Test if data is appropriately resized
                file.get_var( record.name, {i, slab}, data);
                dg::blas1::axpby( 1.,tmp,-1., data);
                // ... and communicator is set
                double result = sqrt(dg::blas1::dot( data, data));
                CHECK( result < 1e-15);
            }
        }


    }
    SECTION( "Write sliced variable")
    {
        std::unique_ptr<dg::x::aGeometry2d> perp_grid_ptr( grid.perp_grid());
        dg::x::Grid2d grid2d = (dg::x::Grid2d)*perp_grid_ptr;
        dg::x::HVec data = dg::evaluate( gradientX, grid);
        auto sliced_data = dg::split( data, grid); // a vector of views
        file.defput_dim( "x", {{"axis", "X"},
            {"long_name", "x-coordinate in Cartesian system"}},
            grid2d.abscissas(0));
        file.defput_dim( "y", {{"axis", "Y"},
            {"long_name", "y-coordinate in Cartesian system"}},
            grid2d.abscissas(1));
        // In MPI only the group containing rank 0 in file comm writes
        file.defput_var( "test", {"y", "x"}, {}, grid2d, sliced_data[0]);
        file.close();

        auto mode = GENERATE( dg::file::nc_nowrite, dg::file::nc_write);
        INFO("TEST "<<( mode == dg::file::nc_write ? "WRITE" : "READ")<<" OPEN MODE\n");
        file.open( "inout.nc", mode);
        dg::x::HVec result = dg::evaluate( dg::zero, grid2d), ana(result);
        // ATTENTION sliced_data[0] is not the same on all ranks
        //dg::blas1::copy( sliced_data[0], ana);
        dg::blas1::kronecker( ana, dg::equals(), gradientX,
            grid2d.abscissas(0), grid2d.abscissas(1), grid.hz()/2.);

        // In MPI when mode == nc_write only the group containing rank 0 in file comm reads
        file.get_var( "test", grid2d, result);
        dg::blas1::axpby( 1.,ana, -1., result);
        double norm = sqrt(dg::blas1::dot( result, result));
        if( mode == dg::file::nc_write)
        {
            DG_RANK0 INFO( "Norm is "<<norm);
            DG_RANK0 CHECK( norm < 1e-14);
        }
        else
        {
            INFO( "Norm is "<<norm);
            CHECK( norm < 1e-14);
        }
    }
    file.close();
    DG_RANK0 std::filesystem::remove( "inout.nc");
#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD);
#endif
}
TEST_CASE( "Documentation")
{
    //! [ncfile]
    // This example compiles in both serial and MPI environments
#ifdef WITH_MPI
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1, 1});
#endif
    // Open file and put some attributes
    dg::file::NcFile file( "inout-test.nc", dg::file::nc_clobber);
    file.put_att( {"title", "Hello world"});
    file.put_att( {"truth", 42});

    //![defput_dim]
    // Generate a grid
    const double x0 = 0., x1 = 2.*M_PI;
    dg::x::CartesianGrid2d grid( x0,x1,x0,x1,3,10,10
#ifdef WITH_MPI
    , comm
#endif
    );
    // and put dimensions to file
    file.defput_dim( "x", {{"axis", "X"},
        {"long_name", "x-coordinate in Cartesian system"}},
        grid.abscissas(0));
    file.defput_dim( "y", {{"axis", "Y"},
        {"long_name", "y-coordinate in Cartesian system"}},
        grid.abscissas(1));
    //![defput_dim]
    //! [defput_var]
    // Generate some data and write to file
    dg::x::HVec data = dg::evaluate( function, grid);
    // Defne and write a variable in one go
    file.defput_var( "variable", {"y", "x"},
                {{"long_name", "A long explanation"}, {"unit", "m/s"}},
                grid, data);
    //! [defput_var]

    // Generate an unlimited dimension and define another variable
    file.def_dimvar_as<double>( "time", NC_UNLIMITED, {{"axis", "T"}});
    file.def_var_as<double>( "dependent", {"time", "y", "x"},
        {{"long_name", "Really interesting"}});
    // Write timeseries
    for( unsigned u=0; u<=2; u++)
    {
        double time = u*0.01;
        file.put_var("time", {u}, time);
        // We can write directly from GPU
        dg::x::DVec data = dg::evaluate( function, grid);
        dg::blas1::scal( data, cos(time));
        file.put_var( "dependent", {u, grid}, data);
    }
    file.close();

    // Open file for reading
    file.open( "inout-test.nc", dg::file::nc_nowrite);
    std::string title = file.get_att_as<std::string>( "title");
    //![get_var]
    // In MPI all ranks automatically get the right chunk of data
    file.get_var( "variable", grid, data);
    //![get_var]
    //![get_dim_size]
    unsigned NT = file.get_dim_size( "time");
    CHECK( NT == 3);
    //![get_dim_size]
    double time;
    file.get_var( "time", {0}, time);
    file.close();
    //! [ncfile]
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Barrier( MPI_COMM_WORLD);
#endif
    DG_RANK0 std::filesystem::remove( "inout-test.nc");
}
