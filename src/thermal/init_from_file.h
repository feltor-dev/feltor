#pragma once


#include "dg/file/nc_utilities.h"
#include "parameters.h"
#include "thermal.h"

namespace thermal
{

using Feltor = thermal::Explicit< dg::x::CylindricalGrid3d, dg::x::IDMatrix,
        dg::x::DMatrix, dg::x::DVec>;
using Vector = std::array<std::vector<dg::x::DVec>, 6>;

std::vector<dg::file::Record<void(dg::x::DVec&, const Vector&, unsigned), dg::file::LongNameAttribute>> restart3d_list = {
    {"n", "gyro-centre density",
        []( dg::x::DVec& result, const Vector& y, unsigned s ) {
             dg::blas1::copy(y[0][s], result);
        }
    },
    {"pperp", "perpendicular pressure",
        []( dg::x::DVec& result, const Vector& y, unsigned s ) {
             dg::blas1::copy(y[1][s], result);
        }
    },
    {"ppara", "parallel pressure",
        []( dg::x::DVec& result, const Vector& y, unsigned s ) {
             dg::blas1::copy(y[2][s], result);
        }
    },
    {"w", "canonical parallel velocity",
        []( dg::x::DVec& result, const Vector& y, unsigned s ) {
             dg::blas1::copy(y[3][s], result);
        }
    },
    {"qperp", "perpendicular heat flux",
        []( dg::x::DVec& result, const Vector& y, unsigned s ) {
             dg::blas1::copy(y[4][s], result);
        }
    },
    {"qpara", "parallel heat flux",
        []( dg::x::DVec& result, const Vector& y, unsigned s ) {
             dg::blas1::copy(y[5][s], result);
        }
    }
};

Vector init_from_file( std::string file_name,
        const dg::x::CylindricalGrid3d& grid, const Parameters& p,
        double& time)
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    Vector y0;
    ///////////////////read in and show inputfile

    dg::file::NcFile file( file_name, dg::file::nc_nowrite);
    dg::file::WrappedJsonValue jsIN = dg::file::string2Json(
        file.get_att_as<std::string>("inputfile"), dg::file::comments::are_forbidden);
    thermal::Parameters pIN( jsIN);
    DG_RANK0 std::cout << "# RESTART from file "<<file_name<< std::endl;
    DG_RANK0 std::cout << "#  file parameters:" << std::endl;
    DG_RANK0 std::cout << pIN.n<<" x "<<pIN.Nx<<" x "<<pIN.Ny<<" x "<<pIN.Nz
                <<" : symmetric "<<std::boolalpha<<pIN.symmetric<<std::endl;
    if ( pIN.num_species != p.num_species)
        throw std::runtime_error( "Cannot restart simulation with different number of species!");
    if ( pIN.name != p.name)
        throw std::runtime_error( "Cannot restart simulation with different species names!");
    if ( pIN.mu != p.mu)
        throw std::runtime_error( "Cannot restart simulation with different species mass!");
    if ( pIN.z != p.z)
        throw std::runtime_error( "Cannot restart simulation with different species charge!");
    if ( pIN.kappa != p.kappa)
        throw std::runtime_error( "Cannot restart simulation with different species kappa!");
    if ( pIN.pi != p.pi)
        throw std::runtime_error( "Cannot restart simulation with different species pi!");

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

    /////////////////////Get time length and initial data///////////////////////////
    unsigned size_time = file.get_dim_size("time");
    file.get_var( "time", {size_time-1}, time);
    DG_RANK0 std::cout << "# Current time = "<< time <<  std::endl;

    dg::x::HVec transferOUTvec = dg::evaluate( dg::zero, grid);
    for( unsigned s=0; s<pIN.num_species; s++)
    {
        for( unsigned k=0; k<restart3d_list.size(); k++)
        {
            file.get_var( "restart_" + pIN.name[s] + "-" + restart3d_list[k].name, {grid_IN}, transferIN);
            dg::blas2::gemv( interpolateIN, transferIN, transferOUTvec);
            dg::assign( transferOUTvec, y0[k][s]);
        }
    }
    file.close();
    /// ///////////////Now Construct initial fields ////////////////////////
    return y0;
}

}//namespace thermal
