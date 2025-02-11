#include <iostream>

#include "catch2/catch_all.hpp"
#ifdef WITH_MPI
#include <mpi.h>
#include "nc_mpi_file.h"
#else
#include "nc_file.h"
#endif
#include "dg/backend/timer.h"


TEST_CASE( "Test the NcFile class")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    //create a grid and some data
#endif

    SECTION( "Creating a file exists on system")
    {
        //! [constructor]
        dg::file::NcFile file( "../test.nc", dg::file::nc_clobber);
        CHECK( std::filesystem::exists( "../test.nc"));
        CHECK( file.is_open());
        // Cannot open another file while open
        CHECK_THROWS_AS( file.open("test.nc"), dg::file::NC_Error);
        file.close();
        //! [constructor]
        DG_RANK0 std::filesystem::remove( "../test.nc");
    }
    SECTION("No-clobber throws")
    {
        if( std::filesystem::exists( "test.nc"))
            DG_RANK0 std::filesystem::remove( "test.nc");
        //! [default]
        dg::file::NcFile file;
        file.open("test.nc", dg::file::nc_noclobber);
        CHECK( file.is_open());
        file.close();
        //! [default]
        CHECK( not file.is_open());
        CHECK_THROWS_AS( file.open("test.nc", dg::file::nc_noclobber),
            dg::file::NC_Error);
        file.close();
        DG_RANK0 std::filesystem::remove( "test.nc");
    }
    SECTION( "Create file form absolute path exists")
    {
        dg::file::NcFile file;
        std::filesystem::path current = std::filesystem::current_path();
        INFO("Current path is "<<current);
        std::filesystem::path absolute = current / "absolute.nc";
        file.open( absolute, dg::file::nc_clobber);
        CHECK( std::filesystem::exists( absolute));
        file.close();
        DG_RANK0 std::filesystem::remove( absolute);
    }
#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD);
#endif
}
TEST_CASE( "Open, groups, dims, atts")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    //create a grid and some data
#endif
    dg::file::NcFile file("test.nc", dg::file::nc_clobber);
    REQUIRE( file.is_open());
    REQUIRE( std::filesystem::exists( "test.nc"));
    SECTION( "Opening modes save or remove data")
    {
        file.put_att({"title", "Hello"} );
        CHECK( file.att_is_defined( "title"));
        file.close();
        file.open("test.nc", dg::file::nc_nowrite);
        CHECK( file.is_open());
        CHECK( file.att_is_defined( "title"));
        file.close();
        file.open("test.nc", dg::file::nc_write);
        CHECK( file.is_open());
        CHECK( file.att_is_defined( "title"));
        file.close();
        // clobber deletes file content
        file.open("test.nc", dg::file::nc_clobber);
        CHECK( file.is_open());
        CHECK( not file.att_is_defined( "title"));
    }
    ///////////////////////    Groups
    SECTION( "Managing groups")
    {
        file.set_grp(".."); // test what happens on root
        file.def_grp("subgroup");
        CHECK(file.grp_is_defined( "subgroup"));

        CHECK_THROWS_AS( file.def_grp("subgroup"), dg::file::NC_Error);

        file.set_grp("subgroup");
        CHECK( file.get_current_path() == "/subgroup");
        CHECK_THROWS_AS(file.set_grp("subgroup"), dg::file::NC_Error);
        file.def_grp("subgroup");
        file.def_grp("subgroup2");
        file.set_grp("/subgroup/subgroup2");
        CHECK( file.get_current_path() == "/subgroup/subgroup2");

        file.def_grp_p("/subgroup/subgroup2/subgroup3/subgroup4");
        CHECK(file.grp_is_defined( "/subgroup/subgroup2"));
        CHECK(file.grp_is_defined( "subgroup3/subgroup4"));
        file.set_grp(".."); // go to parent group
        CHECK( file.get_current_path() == "/subgroup");
        file.set_grp("subgroup2/subgroup3");
        file.set_grp("."); // do nothing "." is current group
        CHECK( file.get_current_path() == "/subgroup/subgroup2/subgroup3");
        file.set_grp(); // go back to root group
        CHECK( file.get_current_path() == "/");
        CHECK( file.grp_is_defined( "/subgroup/subgroup2/subgroup3"));
        CHECK( file.grp_is_defined( "subgroup/subgroup2"));
        CHECK( file.grp_is_defined( "subgroup"));
        auto all_grps_r = file.get_grps_r();
        std::list<std::filesystem::path> ana = {
            "/subgroup",
            "/subgroup/subgroup",
            "/subgroup/subgroup2",
            "/subgroup/subgroup2/subgroup3",
            "/subgroup/subgroup2/subgroup3/subgroup4"
        };
        CHECK( all_grps_r == ana);
    }
    SECTION( "Documenting groups")
    {
        // ! [groups]
        dg::file::NcFile file( "grps.nc", dg::file::nc_clobber);
        // mkdir subgroup
        file.def_grp( "subgrp");
        // cd subgroup
        file.set_grp( "subgrp");
        // all subsequent calls will write to subgrp
        file.put_att( {"title", "This is a subgrp attribute"});
        auto path = file.get_current_path();
        CHECK( path == "/subgrp");
        // mkdir -p
        file.def_grp_p("/subgrp/subgrp2/subgrp3/subgrp4");
        CHECK( file.grp_is_defined( "/subgrp/subgrp2/subgrp3"));

        // cd ..
        file.set_grp(".."); // go to parent group
        // cd
        file.set_grp(); // go to root group

        // mv subgrp sub
        file.rename_grp( "subgrp", "sub");

        // ls
        auto subgrps = file.get_grps();
        // ls -R
        auto allgrps = file.get_grps_r();

        file.close();
        // ! [groups]
        DG_RANK0 std::filesystem::remove( "groups.nc");
    }
    ////////////////// Dimensions
    SECTION( "Dimensions")
    {
        file.def_dim("x", 42);
        std::vector<double> abscissas{ 1.,2.,3.,4.,5.};
#ifdef WITH_MPI
        if( size >= 2)
        {
            if( rank==0)
                abscissas = {1.,2.,3.};
            else if ( rank==1)
                abscissas = {4.,5.};
            else
                abscissas.clear();
        }
        dg::MPI_Vector<std::vector<double>> mpi_abscissas( abscissas, MPI_COMM_WORLD);
        file.defput_dim("y", {{"axis", "Y"}, {"long_name", "y-coordinate"}},
                mpi_abscissas); // Create dim and dimension variable
#else
        file.defput_dim("y", {{"axis", "Y"}, {"long_name", "y-coordinate"}},
                abscissas); // Create dim and dimension variable
#endif
        file.def_dimvar_as<double>("time", NC_UNLIMITED, {{"axis", "T"}});

        size_t xsize = file.get_dim_size("x");
        CHECK( xsize == 42);

        size_t ysize = file.get_dim_size("y");
        CHECK( ysize == 5);
        file.def_grp( "subgroup");
        file.set_grp( "subgroup");
        file.def_dim( "y", 57);
        file.def_dim( "z", 12);
        file.set_grp( "..");
        //https://github.com/Unidata/netcdf-c/issues/2873
        file.def_grp( "dimensions");
        file.set_grp( "dimensions");
        file.def_dim( "w", NC_UNLIMITED);
        file.set_grp( "..");
        //![get_dims]
        auto dims = file.get_dims( );
        std::vector<std::string> result = {"x", "y", "time"};
        CHECK( dims == result);
        //![get_dims]
        file.set_grp( "subgroup");
        dims = file.get_dims( );
        result = {"x", "y", "time", "z"};
        CHECK( dims == result);
        file.set_grp( "");
        file.set_grp( "dimensions");
        auto unlims = file.get_unlim_dims( );
        result = {"w"};
        CHECK( unlims == result);
        file.set_grp( "..");
    }
    file.close();
    DG_RANK0 std::filesystem::remove( "test.nc");
#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD);
#endif
}

TEST_CASE( "Attributes")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    //create a grid and some data
#endif
    dg::file::NcFile file("test.nc", dg::file::nc_clobber);
    ///////////////////////   Attributes
    SECTION( "Read/ write individual Attributes")
    {
        // set attributes
        // ! [put_att]
        file.put_att({"title", "Hello"} );
        INFO( "Silently overwrite existing attribute");
        CHECK_NOTHROW( file.put_att({"title", "Hello world"}));
        file.put_att({"truth", 42} );
        // ! [put_att]
        // ! [put_att_x]
        file.put_att( std::tuple{"half truth", NC_INT, 21});
        int half = file.get_att_as<int>( "half truth");
        CHECK( half == 21);
        // ! [put_att_x]
        file.put_atts({{ "this", "is"}, {"a", 42}, {"vector",
            std::vector{1,2,3}}});
        file.close();
        CHECK_THROWS_AS( file.put_att( {"title", "Blurp"}),
            dg::file::NC_Error);

        for( auto mode : {dg::file::nc_nowrite, dg::file::nc_write})
        {
            file.open( "test.nc", mode);
            // What if Attribute does not exist?
            CHECK_THROWS_AS( file.get_att_as<double>( "Does not exist"),
                dg::file::NC_Error);
            // get attributes
            auto title = file.get_att_as<std::string>("title");
            CHECK( title == "Hello world");
            double t = file.get_att_as<double>("truth");
            CHECK( t == 42);
            int tint = file.get_att_as<int>("truth");
            CHECK( tint == 42);
            auto ts = file.get_att_vec_as<double>("vector"); // get all indices as vector
            std::vector<double> result = {1.0,2.0,3.0};
            CHECK( ts == result );

            file.close();
        }
    }
    SECTION( "nc_att_t map attributes")
    {
        //! [put_atts]
        std::map<std::string, dg::file::nc_att_t> att;
        att["text"] = "Hello World!";
        att["number"] = 3e-4;
        att["int"] = -1;
        att["uint"] = 10;
        att["bool"] = true;
        att["realarray"]  = std::vector{-1.1, 42.3};
        att["realarray"]  = std::vector{-1.1, 42.3};
        att["realnumber"] = -1.1;
        att["intarray"]   = std::vector{-11, 423};
        att["uintarray"]  = std::vector{11, 423};
        att["boolarray"]  = std::vector{true, false};
        file.put_atts(att);
        file.close();
        auto mode = GENERATE( dg::file::nc_nowrite, dg::file::nc_write);
        file.open( "test.nc", mode);
        auto read = file.get_atts();
        CHECK( read == att);
        //! [put_atts]
    }
    SECTION( "Deleting and renaming")
    {
        //! [del_att]
        file.put_att({"same", "thing"});
        CHECK( file.att_is_defined( "same"));
        file.del_att( "same", "");
        CHECK( not file.att_is_defined( "same"));
        //! [del_att]
        // Check attribute rename
        //! [rename_att]
        file.put_att({"ttt", 42} );
        CHECK( file.att_is_defined( "ttt"));
        file.rename_att( "ttt", "truth");
        CHECK( not file.att_is_defined( "ttt", ""));
        CHECK( file.att_is_defined( "truth"));
        //! [rename_att]

        int truth = file.get_att_as<int>("truth");
        CHECK( truth == 42);
    }
    file.close();
    DG_RANK0 std::filesystem::remove( "test.nc");
#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD);
#endif
}

TEST_CASE( "Test variables in the NcFile class")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
#endif
    dg::file::NcFile file("test.nc", dg::file::nc_clobber);
    // Writing tests in netcdf_t.cpp
    SECTION("Variable def is seen")
    {
        //! [get_var_dims]
        file.def_dim("time", NC_UNLIMITED);
        file.def_dim( "y", 57);
        file.def_dim( "z", 12);
        file.def_var_as<double>("variable", {"time", "z","y"});
        CHECK( file.get_var_dims("variable") == std::vector<std::string>{"time", "z", "y"});
        //! [get_var_dims]
        // Dimension misspelled
        CHECK_THROWS_AS( file.def_var_as<double>("another", {"time", "y","x"}),
            dg::file::NC_Error);
        CHECK_THROWS_AS( file.def_var("variable", NC_DOUBLE, {"time",
            "y","x"}), dg::file::NC_Error);

        //! [var_is_defined]
        file.def_var_as<int>( "scalar", {}, {});
        CHECK( file.var_is_defined( "scalar"));
        //! [var_is_defined]
        file.put_var( "scalar", {}, 42);
        CHECK( file.get_var_dims("scalar").empty());
        int scalar;
        file.get_var( "scalar", {}, scalar);
        CHECK( scalar == 42);
    }
    SECTION( "Variable attributes exist")
    {
        file.def_dim("time", NC_UNLIMITED);
        file.def_var_as<double>("variable", {"time"});
        file.put_att({"long_name", "blabla"}, "variable");
        auto att = file.get_att_as<std::string>( "long_name", "variable");
        CHECK( att == "blabla");
        //! [def_var_as]
        std::map<std::string, dg::file::nc_att_t> atts = {{"axis", "T"}};
        file.def_var_as<double>("another", {"time"}, atts);
        att = file.get_att_as<std::string>( "axis", "another");
        CHECK( att == "T");
        //! [def_var_as]
    }
    SECTION( "Unlimited Variable can write anywhere")
    {
        file.def_dimvar_as<double>( "time", NC_UNLIMITED, {{"axis", "T"}});
        file.put_var("time", {52}, 10);
        file.close();
        double test;
        auto mode = GENERATE( dg::file::nc_nowrite, dg::file::nc_write);
        file.open( "test.nc", mode);
        CHECK( file.var_is_defined( "time"));
        CHECK( file.att_is_defined( "axis", "time"));
        CHECK( file.get_att_as<std::string>( "axis", "time") == "T");
        //![get_var]
        file.get_var("time", {52}, test);
        CHECK( test == 10);
        //![get_var]
    }
    SECTION( "Group sees dimension")
    {
        file.def_dim("time", NC_UNLIMITED);
        file.def_grp( "subgroup");
        file.set_grp( "subgroup");
        //! [get_var_type]
        file.def_var_as<double>(   "variable", {"time"});
        CHECK( (file.get_var_type( "variable") == NC_DOUBLE ));
        //! [get_var_type]
        file.def_var_as<double>( "another", {"time"});
        file.def_var_as<int>( "int_var", {"time"});
        CHECK( (file.get_var_type( "int_var") == NC_INT));
        file.set_grp("..");
        //! [get_var_names]
        //Get a list of variables with type double
        auto pred = [&file]( std::string name) {
            return file.get_var_type(name) != NC_DOUBLE;
        };
        file.set_grp("subgroup");
        auto vars = file.get_var_names();
        vars.remove_if( pred);
        CHECK( vars == std::list<std::string>{"variable", "another"} );
        //! [get_var_names]
    }
    file.close();
    DG_RANK0 std::filesystem::remove( "test.nc");
#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD);
#endif
}

TEST_CASE( "File Benchmarks")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    //create a grid and some data
#endif
    dg::file::NcFile file( "benchmark.nc", dg::file::nc_clobber);
    file.def_dim( "time", NC_UNLIMITED);
    file.def_dim( "y", 42);
    file.def_dim( "x", 42);
    dg::Timer timer;
    timer.tic();
    for( unsigned u=0; u<1000; u++)
        file.def_var_as<double>( "var"+std::to_string(u), {"time", "y","x"});
    timer.toc();
    DG_RANK0 std::cout << "Defining 1000 variables took "<<timer.diff()<<"s\n";
    timer.tic();
    for( unsigned u=0; u<1000; u++)
        file.put_att({"long_name", std::to_string(u)}, "var"+std::to_string(u));
    timer.toc();
    DG_RANK0 std::cout << "Set attribute of 1000 variables took "<<timer.diff()<<"s\n";
    timer.tic();
    for( int u=1000; u>0; u--)
    {
        int varid;
        DG_RANK0 nc_inq_varid( file.get_grpid(), ("var"+std::to_string(u)).c_str(), &varid);
    }
    timer.toc();
    DG_RANK0 std::cout << "Getting id of 1000 variables took "<<timer.diff()<<"s\n";
    file.close();
    DG_RANK0 std::filesystem::remove( "benchmark.nc");
#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD);
#endif
}
