#include <iostream>
#include <ctime>
#include <iomanip>
#include "catch2/catch.hpp"

#include "nc_file.h"

TEST_CASE( "Easy attributes")
{
    dg::file::NcFile file("test.nc", dg::file::nc_clobber);
    SECTION( "Special history attribute")
    {
        // history can be parsed
        // https://en.cppreference.com/w/cpp/io/manip/get_time
        int argc = 2;
        char *argv[] = {
            (char*)"./netcdf_t",
            (char*)"input.json",
            NULL
        };
        file.put_att( ".", {"history", dg::file::timestamp(argc, argv)});
        file.put_atts( ".", dg::file::version_flags);
        auto history = file.get_att_as<std::string>( ".", "history");
        INFO("history "<<history);
        std::istringstream ss( history);
        std::tm t = {};
        ss >> std::get_time( &t, "%Y-%m-%d %H:%M:%S %Z");
        CHECK( not ss.fail());
        std::string program;
        ss >> program;
        std::cout << program<<"\n";
        CHECK( program == "./netcdf_t");
        std::string arg;
        ss >> arg;
        CHECK( arg == "input.json");
        CHECK( not ss.fail());
    }
    SECTION( "Special version flags attribute")
    {
        file.put_atts(".", dg::file::version_flags);
        CHECK_NOTHROW( file.get_att_as<std::string>(".", "git_hash"));
        CHECK_NOTHROW( file.get_att_as<std::string>(".", "git_branch"));
        CHECK_NOTHROW( file.get_att_as<std::string>(".", "compile_time"));
        CHECK( file.get_att_as<std::string>(".", "git_hash").size() == 40);
        std::string compile = file.get_att_as<std::string>(".", "compile_time");
        std::istringstream ss( compile);
        std::tm t = {};
        ss >> std::get_time( &t, "%Y-%m-%d %H:%M:%S");
        CHECK( not ss.fail());
    }
    file.close();
    DG_RANK0 std::filesystem::remove( "test.nc");
}
