#pragma once


#include "dg/file/nc_utilities.h"
#include "parameters.h"

namespace feltor
{//We use the typedefs and DG_RANK0
//
//everyone reads their portion of the input data
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

    dg::file::NC_Error_Handle errIN;
    int ncidIN;
    errIN = nc_open( file_name.data(), NC_NOWRITE, &ncidIN);
    size_t length;
    errIN = nc_inq_attlen( ncidIN, NC_GLOBAL, "inputfile", &length);
    std::string input(length, 'x');
    errIN = nc_get_att_text( ncidIN, NC_GLOBAL, "inputfile", &input[0]);
    dg::file::WrappedJsonValue jsIN = dg::file::string2Json( input, dg::file::comments::are_forbidden);
    feltor::Parameters pIN( jsIN);
    unsigned  pINn  = pIN.n;
    unsigned  pINNx = pIN.Nx;
    unsigned  pINNy = pIN.Ny;
    unsigned  pINNz = pIN.Nz;
    bool      pINsymmetric = pIN.symmetric;
    DG_RANK0 std::cout << "RESTART from file "<<file_name<< std::endl;
    DG_RANK0 std::cout << " file parameters:" << std::endl;
    DG_RANK0 std::cout << pINn<<" x "<<pINNx<<" x "<<pINNy<<" x "<<pINNz
                <<" : symmetric "<<std::boolalpha<<pINsymmetric<<std::endl;

    // Now read in last timestep
    dg::x::CylindricalGrid3d grid_IN( grid.x0(), grid.x1(), grid.y0(),
        grid.y1(), grid.z0(), grid.z1(),
        pINn, pINNx, pINNy, pINNz, dg::DIR, dg::DIR, dg::PER
        #ifdef WITH_MPI
        , grid.communicator()
        #endif //WITH_MPI
        );
    dg::x::IHMatrix interpolateIN;
    dg::x::HVec transferIN;
    if( pINsymmetric)
    {
        std::unique_ptr< typename dg::x::CylindricalGrid3d::perpendicular_grid>
            grid_perp ( static_cast<typename
                dg::x::CylindricalGrid3d::perpendicular_grid*>(grid.perp_grid()));
        interpolateIN = dg::create::interpolation( grid, *grid_perp);
        transferIN = dg::evaluate(dg::zero, *grid_perp);
    }
    else
    {
        interpolateIN = dg::create::interpolation( grid, grid_IN);
        transferIN = dg::evaluate(dg::zero, grid_IN);
    }

    #ifdef WITH_MPI
    int dimsIN[3],  coordsIN[3];
    int periods[3] = {false, false, true}; //non-, non-, periodic
    MPI_Cart_get( grid.communicator(), 3, dimsIN, periods, coordsIN);
    size_t countIN[3] = {grid_IN.local().Nz(),
            grid_IN.n()*(grid_IN.local().Ny()),
            grid_IN.n()*(grid_IN.local().Nx())};
    size_t startIN[3] = {coordsIN[2]*countIN[0],
                         coordsIN[1]*countIN[1],
                         coordsIN[0]*countIN[2]};
    #else //WITH_MPI
    size_t startIN[3] = {0, 0, 0};
    size_t countIN[3] = {grid_IN.Nz(), grid_IN.n()*grid_IN.Ny(),
        grid_IN.n()*grid_IN.Nx()};
    #endif //WITH_MPI
    if( pINsymmetric)
    {
        countIN[0] = 1;
        startIN[0] = 0;
    }
    std::vector<dg::x::HVec> transferOUTvec( 5, dg::evaluate( dg::zero, grid));

    std::string namesIN[5] = {"restart_electrons", "restart_ions",
        "restart_Ue", "restart_Ui", "restart_aparallel"};

    int timeIDIN;
    size_t size_time, count_time = 1;
    /////////////////////Get time length and initial data///////////////////////////
    errIN = nc_inq_dimid( ncidIN, "time", &timeIDIN);
    errIN = nc_inq_dimlen(ncidIN, timeIDIN, &size_time);
    errIN = nc_inq_varid( ncidIN, "time", &timeIDIN);
    size_time -= 1;
    errIN = nc_get_vara_double( ncidIN, timeIDIN, &size_time, &count_time, &time);
    DG_RANK0 std::cout << " Current time = "<< time <<  std::endl;
    for( unsigned i=0; i<5; i++)
    {
        int dataID;
        errIN = nc_inq_varid( ncidIN, namesIN[i].data(), &dataID);
        errIN = nc_get_vara_double( ncidIN, dataID, startIN, countIN,
            #ifdef WITH_MPI
                transferIN.data().data()
            #else //WITH_MPI
                transferIN.data()
            #endif //WITH_MPI
            );
        dg::blas2::gemv( interpolateIN, transferIN, transferOUTvec[i]);
    }
    errIN = nc_close(ncidIN);
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
