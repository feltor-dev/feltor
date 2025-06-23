#include <iostream>
#include <fstream>
#include <cassert>
#include <filesystem>
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include "json_probes.h"
#include "catch2/catch_all.hpp"

TEST_CASE( "Probes Parser")
{
    int rank = 0;
#ifdef WITH_MPI
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    SECTION( "Empty")
    {
        // Amazing string literals use (they existed since C++-11 !!!)
        std::string probes = R"unique( { "I" : "have no probes"})unique";
        auto js_empty = dg::file::string2Json(probes);
        auto empty = dg::file::parse_probes( js_empty );
        CHECK( not empty.probes );
    }
    SECTION( "Direct")
    {
        std::string probes = R"unique({
            "probes":
            {
                "input" : "coords",
                "coords" :
                {
                    "coords-names" : ["x", "y"],
                    "format" : {},
                    "x" : [0,1,2,3],
                    "y" : [0,1,2,3]
                }
            }
        })unique";
        auto js_direct = dg::file::string2Json(probes);
        auto direct = dg::file::parse_probes( js_direct);
        CHECK( direct.coords.size() == 2);
        std::vector<std::string> names = {"x", "y"};
        for( unsigned i=0; i<2; i++)
            CHECK( direct.coords_names[i] == names[i]);
        auto js_format = dg::file::string2Json( direct.format);
        INFO( "Read format "<<direct.format);
        CHECK( js_format.empty());
        CHECK( direct.probes == true);
        unsigned num_pins = direct.get_coords_sizes();
        if(rank==0) CHECK( num_pins == 4);
        for( unsigned i=0; i<num_pins; i++)
        {
            if(rank==0) CHECK( direct.coords[0][i] == (double)i);
            if(rank==0) CHECK( direct.coords[1][i] == (double)i);
        }
    }
    SECTION( "From file")
    {
        // We need to generate a file
        if( rank == 0)
        {
            std::ofstream file( "file.json");
            file << R"unique({
                "coords-names" : ["x", "y"],
                "format" : [
                    {"name" : "divertor", "pos" : [0,1], "shape" : [2]},
                    {"name" : "omp", "pos" : [1,3], "shape" : [3]}
                ],
                "x" : [0,1,2,3],
                "y" : [0,1,2,3]
            })unique";
            file.close();
        }
#ifdef WITH_MPI
        MPI_Barrier( MPI_COMM_WORLD);
#endif
        REQUIRE( std::filesystem::exists( "file.json"));
        std::string probes = R"unique({
            "probes":
            {
                "input" : "file",
                "file" : "file.json",
                "scale" : [10,0.5]
            }
        })unique";
        auto js_fromfile = dg::file::string2Json(probes);
        auto fromfile = dg::file::parse_probes( js_fromfile);
        CHECK( fromfile.coords.size() == 2);
        std::vector<std::string> names = {"x", "y"};
        for( unsigned i=0; i<2; i++)
            CHECK( fromfile.coords_names[i] == names[i]);
        INFO( "Read probe format:\n"<<fromfile.format);
        dg::file::WrappedJsonValue js_format = dg::file::string2Json( fromfile.format);
        CHECK( js_format[0]["name"].asString() == "divertor");
        CHECK( js_format[0]["pos"][0].asInt() == 0);
        CHECK( js_format[0]["pos"][1].asInt() == 1);
        CHECK( js_format[0]["shape"][0].asInt() == 2);
        CHECK( js_format[1]["name"].asString() == "omp");
        CHECK( js_format[1]["pos"][0].asInt() == 1);
        CHECK( js_format[1]["pos"][1].asInt() == 3);
        CHECK( js_format[1]["shape"][0].asInt() == 3);

        CHECK( fromfile.probes == true);
        unsigned num_pins = fromfile.get_coords_sizes();
        if(rank==0) CHECK( num_pins == 4);
        for( unsigned i=0; i<num_pins; i++)
        {
            if(rank==0) CHECK( fromfile.coords[0][i] == 10.*(double)i);
            if(rank==0) CHECK( fromfile.coords[1][i] == 0.5*(double)i);
        }
#ifdef WITH_MPI
        MPI_Barrier( MPI_COMM_WORLD);
#endif
        if(rank==0) std::filesystem::remove( "file.json");
    }

}

