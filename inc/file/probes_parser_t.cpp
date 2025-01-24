#include <iostream>
#include <cassert>
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include "json_probes.h"
#include "catch2/catch.hpp"

TEST_CASE( "Probes Parser")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
#endif

    auto js_empty = dg::file::file2Json("probes_empty.json");
    auto empty = dg::file::parse_probes( dg::file::file2Json("probes_empty.json") );
    DG_RANK0 CHECK( !empty.probes );
    auto js_direct = dg::file::file2Json("probes_direct.json");
    auto direct = dg::file::parse_probes( js_direct);
    DG_RANK0 CHECK( direct.coords.size() == 2);
    std::vector<std::string> names = {"x", "y"};
    for( unsigned i=0; i<2; i++)
        DG_RANK0 CHECK( direct.coords_names[i] == names[i]);
    DG_RANK0 std::cout << "Expected result "<<direct.format<<" should be same as \"none\" "<<std::endl;
    //DG_RANK0 assert( direct.format == "none");
    DG_RANK0 CHECK( direct.probes == true);
    unsigned num_pins = direct.get_coords_sizes();
    for( unsigned i=0; i<num_pins; i++)
    {
        DG_RANK0 CHECK( direct.coords[0][i] == (double)i);
        DG_RANK0 CHECK( direct.coords[1][i] == (double)i);
    }
    auto js_fromfile = dg::file::file2Json("probes_fromfile.json");
    auto fromfile = dg::file::parse_probes( js_fromfile);
    DG_RANK0 CHECK( fromfile.coords.size() == 2);
    for( unsigned i=0; i<2; i++)
        DG_RANK0 CHECK( fromfile.coords_names[i] == names[i]);
    DG_RANK0 std::cout << "Read probe format\n";
    DG_RANK0 std::cout << "Expected result\n"<<fromfile.format<<std::endl;
    DG_RANK0 std::cout << "Compare to content of probes_file.json!\n";

    //DG_RANK0 assert( fromfile.format == "none");
    DG_RANK0 CHECK( fromfile.probes == true);

    for( unsigned i=0; i<num_pins; i++)
    {
        DG_RANK0 CHECK( fromfile.coords[0][i] == 10.*(double)i);
        DG_RANK0 CHECK( fromfile.coords[1][i] == 0.5*(double)i);
    }

}

