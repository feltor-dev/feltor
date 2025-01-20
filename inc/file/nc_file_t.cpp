#include <iostream>

#ifdef WITH_MPI
#include <mpi.h>
#include "nc_mpi_file.h"
#else
#include "nc_file.h"
#endif
#include "dg/backend/timer.h"


int main(int argc, char* argv[])
{
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    //create a grid and some data
#endif
#ifdef WITH_MPI
    std::string filename = "filempi.nc";
#else
    std::string filename = "file.nc";
#endif
    std::filesystem::path current = std::filesystem::current_path();
    DG_RANK0 std::cout << "Current path is "<<current<<std::endl;

    dg::file::NcFile testfile( "../test.nc", dg::file::nc_clobber);
    DG_RANK0 assert( std::filesystem::exists( "../test.nc"));
    DG_RANK0 std::filesystem::remove( "../test.nc");
    testfile.close();
    std::filesystem::path absolute = current / "absolute.nc";
    testfile.open( absolute, dg::file::nc_clobber);
    DG_RANK0 assert( std::filesystem::exists( absolute));
    DG_RANK0 std::filesystem::remove( absolute);
    testfile.close();
    dg::file::NcFile file;
    try {
        file.open(filename, dg::file::nc_clobber);
    }
    catch(dg::file::NC_Error & e)	{
        std::cerr << e.what();
    }
    assert( file.is_open());
    ///////////////////////    Groups
    file.set_grp(".."); // test what happens on root
    file.def_grp("subgroup");
    //file.def_grp("subgroup");
    file.set_grp("subgroup");
    //file.set_grp("subgroup");
    file.def_grp("subgroup");
    file.def_grp("subgroup2");
    file.set_grp("/subgroup/subgroup2");
    file.def_grp_p("/subgroup/subgroup2/subgroup3/subgroup4");
    file.set_grp(".."); // go to parent group
    file.set_grp("subgroup2/subgroup3");
    file.set_grp("."); // do nothing "." is current group
    file.set_grp(); // go back to root group
    assert( file.grp_is_defined( "/subgroup/subgroup2/subgroup3"));
    assert( file.grp_is_defined( "subgroup/subgroup2"));
    assert( file.grp_is_defined( "subgroup"));
    auto all_grps = file.get_grps();
    for( auto grp : all_grps)
        DG_RANK0 std::cout << "Found "<<grp<<"\n";
    auto all_grps_r = file.get_grps_r();
    std::vector<std::filesystem::path> ana = {
        "/subgroup",
        "/subgroup/subgroup",
        "/subgroup/subgroup2",
        "/subgroup/subgroup2/subgroup3",
        "/subgroup/subgroup2/subgroup3/subgroup4"
    };
    for( unsigned i=0; i<ana.size(); i++)
    {
        assert( all_grps_r[i] == ana[i]);
    }
    DG_RANK0 std::cout << "Groups PASSED\n";
    ///////////////////////   Attributes
    // set attributes

    file.put_att(".", {"title", "Hello"} );
    file.put_att(".", {"same", "thing"} );
    file.put_att(".", {"title", "Hello world"});
    file.put_att(".", {"ttt", 42} );
    std::map<std::string, dg::file::nc_att_t> map_atts = {
        {"anumber", 42},
        {"axis", "X"},
        {"real_vector", std::vector{1.0,2.0,3.0}},
        {"int_vector", std::vector{1,2,3}}
    };
    file.put_atts(".", map_atts);
    file.put_atts(".", {{ "this", "is"}, {"a", 42}, {"map", std::vector{1,2,3}}});
    file.close();
    file.open( filename, dg::file::nc_nowrite);
    // get attributes
    auto title = file.get_att_as<std::string>(".", "title");
    assert( title == "Hello world");
    double t = file.get_att_as<double>(".", "ttt");
    assert( t == 42);
    auto ts = file.get_att_vec_as<double>(".", "real_vector"); // get all indices as vector
    std::vector<double> result = {1.0,2.0,3.0};
    assert( ts == result );

    std::map<std::string, dg::file::nc_att_t> atts = file.get_atts();
    std::string hello = "Hello world";
    assert( std::get<std::string>(atts.at("title")) == hello);
    int fortytwo = 42;
    assert( std::get<int>(atts.at("ttt")) == fortytwo);
    file.close();
    DG_RANK0 std::cout << "Attributes Write PASSED\n";

    file.open( filename, dg::file::nc_write);
    file.del_att( ".", "same");
    assert( not file.att_is_defined( ".", "same"));
    file.rename_att( ".", "ttt", "truth");
    int truth = file.get_att_as<int>(".", "truth");
    assert( not file.att_is_defined( ".", "ttt"));
    assert( file.att_is_defined( ".", "truth"));
    assert( truth == fortytwo);
    DG_RANK0 std::cout << "Attributes Read PASSED\n";

    ////////////////// Dimensions
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
    file.defput_dim_as<double>("time", NC_UNLIMITED, {{"axis", "T"}});

    size_t xsize = file.get_dim_size("x");
    assert( xsize == 42);

    size_t ysize = file.get_dim_size("y");
    assert( ysize == 5);
    file.set_grp( "subgroup");
    file.def_dim( "y", 57);
    file.def_dim( "z", 12);
    file.set_grp( "..");
    //https://github.com/Unidata/netcdf-c/issues/2873
    file.def_grp( "dimensions");
    file.set_grp( "dimensions");
    file.def_dim( "w", NC_UNLIMITED);
    file.set_grp( "..");
    {
    auto dims = file.get_dims( );
    std::vector<std::string> result = {"x", "y", "time"};
    assert( dims == result);
    }
    file.set_grp( "subgroup");
    {
    auto dims = file.get_dims( );
    std::vector<std::string> result = {"x", "y", "time", "z"};
    assert( dims == result);
    }
    file.set_grp( "");
    file.set_grp( "dimensions");
    {
    auto dims = file.get_unlim_dims( );
    std::vector<std::string> result = {"w"};
    assert( dims == result);
    }
    file.set_grp( "..");
    DG_RANK0 std::cout << "PASSED Dimensions\n";
    /////////////////// Variables put

    file.def_var_as<double>("variable", {"time", "y","x"});
    //file.def_var("variable", NC_DOUBLE, {"time", "y","x"}); // this will throw
    std::vector<double> data( xsize*ysize, 7);
#ifdef WITH_MPI
    // TODO Make mpi test
#else
    file.put_var("variable", {{0,0,0},{0,ysize,xsize}}, data);
#endif
    file.def_var_as<int>( "scalar", {}, {});
    file.put_var( "scalar", {}, 42);
    assert( file.get_var_dims("scalar").empty());

    file.put_att("variable", {"long_name", "blabla"});
    std::vector<int> data2(ysize, 42);
    //file.defput_var("name", { "y" }, {{"long_name", "blub"}}, data2);
    //file.put("name", data);
    file.put_var("time", {52}, 10);
    {
    double test;
    file.get_var("time", {52}, test);
    assert( test == 10);
    }

    file.set_grp( "subgroup");
    file.def_var_as<double>( "variable", {"time", "x"});
    file.def_var_as<int>( "int_var", {"time", "y"});
    file.set_grp("..");
    auto vars = file.get_var_names_r();
    DG_RANK0 std::cout <<"ALL VARIABLES (compare to ncdump -h "<<filename<<")\n";
    for( auto& grp : vars)
    {
        DG_RANK0 std::cout << "\tGroup " << grp.first<<"\n";
        file.set_grp( grp.first);
        for( auto& var : grp.second)
        {
            DG_RANK0 std::cout << "\t\t"<<var <<" ( ";
            auto dims = file.get_var_dims(var);
            for( unsigned u=0; u<dims.size(); u++)
                DG_RANK0 std::cout<<dims[u]<<", ";
            DG_RANK0 std::cout << ")\n";
        }
    }
    file.set_grp();



    /////////////////// Variables get
#ifdef WITH_MPI
    // TODO Make mpi test
#else
    file.get_var( "variable", {{0,0,0}, {0, ysize, xsize}}, data);
    unsigned vsize = xsize*ysize;
    assert( data.size() == vsize);
    assert( data[0] == 7);
#endif
    static_assert( dg::is_scalar_v<int>);
    int scalar;
    file.get_var( "scalar", {}, scalar);
    assert( scalar == 42);
    DG_RANK0 std::cout << "PASSED Getters\n";

    file.close();
    DG_RANK0 std::cout << "\n\n Some Benchmarks...\n";
    // Some benchmarks

    file.open( "benchmark.nc", dg::file::nc_clobber);
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
        file.put_att( "var"+std::to_string(u), {"long_name", std::to_string(u)});
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
    MPI_Finalize();
#endif
    return 0;
}
