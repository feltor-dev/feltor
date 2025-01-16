#include <iostream>
#include <string>
#include <netcdf.h>
#include <cmath>

#ifdef WITH_MPI
#include <mpi.h>
#include "nc_mpi_file.h"
#endif
#include "nc_file.h"
#include "records.h"

#include "dg/algorithm.h"

double function( double x, double y, double z){return sin(x)*sin(y)*cos(z);}
double gradientX(double x, double y, double z){return cos(x)*sin(y)*cos(z);}
double gradientY(double x, double y, double z){return sin(x)*cos(y)*cos(z);}
double gradientZ(double x, double y, double z){return -sin(x)*sin(y)*sin(z);}


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

int main(int argc, char* argv[])
{
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm;
    int dims[3] = {0,0,0};
    MPI_Dims_create( size, 3, dims);
    std::stringstream ss;
    ss<< dims[0]<<" "<<dims[1]<<" "<<dims[2];
    dg::mpi_init3d( dg::PER, dg::PER, dg::PER, comm, ss);
#endif
#ifdef WITH_MPI
    std::string filename = "netcdfmpi.nc";
#else
    std::string filename = "netcdf.nc";
#endif
    DG_RANK0 std::cout << "WRITE A TIMEDEPENDENT SCALAR, SCALAR FIELD, AND VECTOR FIELD TO NETCDF4 FILE "
                       << filename<<"\n";
    double Tmax=2.*M_PI;
    double NT = 10;
    double h = Tmax/NT;
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::Grid3d grid( x0,x1,x0,x1,x0,x1,3,20,20,20
#ifdef WITH_MPI
    , comm
#endif
    );
    //create NetCDF File
    dg::file::NcFile file( filename, dg::file::nc_clobber);

    file.put_att( ".", {"title", "NetCDF test file"});
    file.put_att( ".", {"history", dg::file::timestamp(argc, argv)});
    file.put_atts( ".", dg::file::version_flags);
    file.defput_dim_as<double>( "time", NC_UNLIMITED, {{"axis", "T"}});
    // It is possible to write to any index in an unlimited variable
    file.put_var("time", {5}, Tmax);
    file.defput_dim( "z", {{"axis", "Z"},
        {"long_name", "z-coordinate in Cartesian system"}}, grid.abscissas(2));
    file.defput_dim( "y", {{"axis", "Y"},
        {"long_name", "y-coordinate in Cartesian system"}}, grid.abscissas(1));
    file.defput_dim( "x", {{"axis", "X"},
        {"long_name", "x-coordinate in Cartesian system"}}, grid.abscissas(0));
    file.def_var_as<double>( "Energy", {"time"}, {{"long_name", "Energy"}});
    for( auto& record : records)
        file.def_var_as<double>( record.name, {"time", "z", "y", "x"},
            record.atts);
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
    file.defput_dim_as<double>( "ptime", NC_UNLIMITED, {{"axis", "T"}});
    for( auto& record : records)
        file.def_var_as<double>( record.name, {"ptime", "zr", "yr", "xr"},
            record.atts);
    file.set_grp("..");
    dg::MultiMatrix<dg::x::DMatrix, dg::x::DVec> project =
        dg::create::fast_projection( grid, 1, 2, 2);
    dg::x::DVec result = dg::evaluate( dg::zero, grid);
    dg::x::DVec tmp = dg::evaluate( dg::zero, grid_out);


    for(unsigned i=0; i<=NT; i++)
    {
        DG_RANK0 std::cout<<"Write timestep "<<i<<"\n";
        double time = i*h;
        auto data = dg::evaluate( function, grid);
        dg::blas1::scal( data, cos( time));
        double energy = dg::blas1::dot( data, data);
        if( i%2 == 0)
            file.put_var( "Energy", {i}, energy);
        file.put_var("time", {i}, time);
        file.set_grp( "projected");
        file.put_var( "ptime", {i}, time);
        file.set_grp( "..");
        for( auto& record : records)
        {
            record.function ( result, grid, time);
            file.put_var( record.name, {i, grid}, result);
            file.set_grp( "projected");
            dg::apply( project, result, tmp);
            file.put_var( record.name, {i, grid_out}, tmp);
            file.set_grp( "..");
        }
    }
    file.close();
    DG_RANK0 std::cout << "\n\n";

    // open and read back in
    for( auto mode : {dg::file::nc_nowrite, dg::file::nc_write})
    {
        std::cout << "TEST "<<( mode == dg::file::nc_write ? "WRITE" : "READ")<<" OPEN MODE\n";
        file.open( filename, mode);
        // This is how to fully check a dimension
        assert( file.dim_is_defined( "x"));
        assert( file.var_is_defined( "x"));
        assert( file.get_dim_size( "x") == grid.shape(0));
        dg::x::HVec absx = grid.abscissas(0);
        file.get_var( "x", {absx}, absx);
#ifdef WITH_MPI
        assert( absx.data() == grid.abscissas(0).data());
#else
        assert( absx == grid.abscissas(0));
#endif
        auto variables = file.get_vars_r();
        for ( auto name : variables["/"])
        {
            if ( file.get_var_dims( name) == std::vector<std::string>{"time"})
                DG_RANK0 std::cout << "Found 0d name "<<name<<"\n";
            if ( file.get_var_dims( name) == std::vector<std::string>{"time", "z", "y", "x"})
                DG_RANK0 std::cout << "Found 3d name "<<name<<"\n";
        }
        for ( auto name : variables["/projected"])
            DG_RANK0 std::cout << "Found Projected 3d name "<<name<<"\n";

        unsigned num_slices = file.get_dim_size("time");
        assert(num_slices == NT+1);
        DG_RANK0 std::cout << "Found "<<num_slices<<" timesteps in file\n";
        // Test that dimension is indeed what we expect
        auto abscissas = grid.abscissas(0), test( abscissas);
        file.get_var( "x", {grid.axis(0)}, test);
        dg::blas1::axpby( 1., abscissas, -1., test);
        assert( dg::blas1::dot( test, test) == 0);

        auto data = dg::evaluate( function, grid);
        auto dataP = dg::evaluate( function, grid_out);
        for(unsigned i=0; i<num_slices; i++)
        {
            DG_RANK0 std::cout<<"Read timestep "<<i<<"\n";
            double time, energy;
            file.get_var("time", {i}, time);
            file.get_var("Energy", {i}, energy);
            DG_RANK0 std::cout << "Time "<<time<<" Energy "<<energy<<"\t";
            file.get_var( "vectorX", {i, grid}, data);
            file.set_grp("projected");
            file.get_var( "vectorX", {i, 1, grid_out}, dataP);
            file.set_grp("..");
#ifdef MPI_VERSION
            DG_RANK0 std::cout << "data "<<data.data()[0]<<" dataP "<<dataP.data()[0]<<"\n";
#else
            std::cout << "data "<<data[0]<<" dataP "<<dataP[0]<<"\n";
#endif
        }
        file.close();
    }

#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
