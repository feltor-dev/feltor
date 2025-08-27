#pragma once


#include "dg/file/nc_utilities.h"
#include "parameters.h"
#include "feltor.h"

namespace feltor
{

using Feltor = feltor::Explicit< dg::x::CylindricalGrid3d, dg::x::IDMatrix,
        dg::x::DMatrix, dg::x::DVec>;

std::vector<dg::file::Record<void(dg::x::DVec&, Feltor&), dg::file::LongNameAttribute>> restart3d_list = {
    {"restart_electrons", "electron density",
        []( dg::x::DVec& result, Feltor& f ) {
             dg::blas1::copy(f.restart_density(0), result);
        }
    },
    {"restart_ions", "ion density",
        []( dg::x::DVec& result, Feltor& f ) {
             dg::blas1::copy(f.restart_density(1), result);
        }
    },
    {"restart_Ue", "parallel electron velocity",
        []( dg::x::DVec& result, Feltor& f ) {
             dg::blas1::copy(f.restart_velocity(0), result);
        }
    },
    {"restart_Ui", "parallel ion velocity",
        []( dg::x::DVec& result, Feltor& f ) {
             dg::blas1::copy(f.restart_velocity(1), result);
        }
    },
    {"restart_aparallel", "parallel magnetic potential",
        []( dg::x::DVec& result, Feltor& f ) {
             dg::blas1::copy(f.restart_aparallel(), result);
        }
    }
};

std::array<std::array<dg::x::DVec,2>,2> init_from_file( std::string file_name,
        const dg::x::CylindricalGrid3d& grid, const Parameters& p,
        double& time)
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    std::array<std::array<dg::x::DVec,2>,2> y0;
    ///////////////////read in and show inputfile

    dg::file::NcFile file( file_name, dg::file::nc_nowrite);
    dg::file::WrappedJsonValue jsIN = dg::file::string2Json(
        file.get_att_as<std::string>("inputfile"),
        dg::file::comments::are_forbidden);
    feltor::Parameters pIN( jsIN);
    DG_RANK0 std::cout << "# RESTART from file "<<file_name<< std::endl;
    DG_RANK0 std::cout << "#  file parameters:" << std::endl;
    DG_RANK0 std::cout << pIN.n<<" x "<<pIN.Nx<<" x "<<pIN.Ny<<" x "<<pIN.Nz
                <<" : symmetric "<<std::boolalpha<<pIN.symmetric<<std::endl;

    // Now read in last timestep
    dg::x::CylindricalGrid3d grid_IN( grid.x0(), grid.x1(), grid.y0(),
        grid.y1(), grid.z0(), grid.z1(),
        pIN.n, pIN.Nx, pIN.Ny, pIN.Nz, dg::DIR, dg::DIR, dg::PER
        #ifdef WITH_MPI
        , grid.communicator()
        #endif //WITH_MPI
        );
    // Theoretically we can change resolution
    dg::x::IHMatrix interpolateIN;
    dg::x::HVec transferIN;
    if( pIN.symmetric)
    {
        std::unique_ptr<dg::x::aGeometry2d> grid_perp( grid.perp_grid());
        interpolateIN = dg::create::prolongation( grid, std::array{2u});
        transferIN = dg::evaluate(dg::zero, *grid_perp);
    }
    else
    {
        interpolateIN = dg::create::interpolation( grid, grid_IN);
        transferIN = dg::evaluate(dg::zero, grid_IN);
    }
    std::vector<dg::x::HVec> transferOUTvec( 5, dg::evaluate( dg::zero, grid));

    /////////////////////Get time length and initial data///////////////////////////
    unsigned size_time = file.get_dim_size("time");
    file.get_var( "time", {size_time-1}, time);
    DG_RANK0 std::cout << "# Current time = "<< time <<  std::endl;
    for( unsigned i=0; i<restart3d_list.size(); i++)
    {
        file.get_var( restart3d_list[i].name, {grid_IN}, transferIN);
        dg::blas2::gemv( interpolateIN, transferIN, transferOUTvec[i]);
    }
    file.close();
    /// ///////////////Now Construct initial fields ////////////////////////
    //
    //Convert to W
    dg::blas1::axpby( 1., transferOUTvec[2], 1./p.mu[0], transferOUTvec[4],
            transferOUTvec[2]);
    dg::blas1::axpby( 1., transferOUTvec[3], 1./p.mu[1], transferOUTvec[4],
            transferOUTvec[3]);

    dg::assign( transferOUTvec[0], y0[0][0]); //ne
    dg::assign( transferOUTvec[1], y0[0][1]); //Ni
    dg::assign( transferOUTvec[2], y0[1][0]); //We
    dg::assign( transferOUTvec[3], y0[1][1]); //Wi
    return y0;
}

}//namespace feltor
