#include <iostream>
#include <ctime>
#include <iomanip>
#include "catch2/catch.hpp"

#include "easy_atts.h"

template<class T>
void display( const T& v)
{
    UNSCOPED_INFO( v << "\n");
}

template<class T>
void display( std::vector<T>& v)
{
    UNSCOPED_INFO( "[ ");
    for( unsigned u=0; u<v.size(); u++)
        UNSCOPED_INFO( v[u]<<", ");
    UNSCOPED_INFO( "]\n");
}

// Only serial testing, no MPI

TEST_CASE( "Easy attributes")
{
    INFO( "Write attributes to file and read them back in");
    std::string inputfile = "{\
    \"physical\": {\"value\" : 1.2,\n\
    \"another\" : \"something\"},\n\
    \"bla\": \"blub\"}";
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
    int argc = 1;
    char *argv[] = {
        (char*)"./easy_atts_t",
        NULL
    };

    att["title"] = "Output file of feltor/src/toefl/toefl.cpp";
    att["Conventions"] = "CF-1.8";
    att["history"] = dg::file::timestamp( argc, argv);
    att["source"] = "FELTOR";
    att.insert( dg::file::version_flags.begin(), dg::file::version_flags.end());
    // Here we put the inputfile as a string without comments so that it can be read later by another parser
    att["inputfile"] = inputfile;
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_create( "atts.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    INFO( "Attributes to write to \"atts.nc\"\n");
    for( auto& at : att)
    {
        INFO( at.first <<" : ");
        std::visit( []( auto&& arg) { display(arg);}, at.second);
    }
    dg::file::detail::put_atts( ncid, NC_GLOBAL, att);
    // TEST EMPTY Value
    std::map<std::string, dg::file::nc_att_t> empty;
    dg::file::detail::put_atts( ncid, NC_GLOBAL, empty);
    err = nc_close(ncid);
    INFO("\nTEST BY USING ncdump -h atts.nc\n");

    INFO("NOW test reading of attributes");
    err = nc_open( "atts.nc", 0, &ncid);
    auto read = dg::file::detail::get_atts_as<dg::file::nc_att_t>( ncid, NC_GLOBAL);
    INFO("Read attributes from \"atts.nc\"\n");
    for( auto& at : read)
    {
        INFO( at.first <<" : ");
        std::visit( []( auto&& arg) { display(arg);}, at.second);
        CHECK( read[at.first] == att[at.first]);
    }
    CHECK( att == read);
    INFO( "History is\n"<<std::get<std::string>( read["history"]));
    INFO( "Git hash is\n"<<std::get<std::string>( read["git_hash"]));

    err = nc_close(ncid);
    err = nc_open( "atts.nc", NC_WRITE, &ncid);
    // TEST const char text
    dg::file::detail::put_att( ncid, NC_GLOBAL, std::pair{ "Text" , "test"});
    std::string test = std::get<std::string>(dg::file::detail::get_att_t( ncid,
                NC_GLOBAL, "Text"));
    CHECK( test == "test");
    err = nc_close(ncid);
}
