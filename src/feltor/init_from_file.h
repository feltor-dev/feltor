#pragma once


#include "dg/file/nc_utilities.h"

namespace feltor
{//We use the typedefs and MPI_OUT
//
//everyone reads their portion of the input data
//don't forget to also read source profiles
std::array<std::array<DVec,2>,2> init_from_file( std::string file_name, const Geometry& grid, const Parameters& p, double& time){
#ifdef FELTOR_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    std::array<std::array<DVec,2>,2> y0;
    ///////////////////read in and show inputfile

    file::NC_Error_Handle errIN;
    int ncidIN;
    errIN = nc_open( file_name.data(), NC_NOWRITE, &ncidIN);
    Json::Value jsIN;
    size_t length;
    errIN = nc_inq_attlen( ncidIN, NC_GLOBAL, "inputfile", &length);
    std::string input(length, 'x');
    errIN = nc_get_att_text( ncidIN, NC_GLOBAL, "inputfile", &input[0]);
    file::string2Json( input, jsIN, file::comments::are_forbidden);
    unsigned  pINn  = jsIN["n"].asUInt();
    unsigned  pINNx = jsIN["Nx"].asUInt();
    unsigned  pINNy = jsIN["Ny"].asUInt();
    unsigned  pINNz = jsIN["Nz"].asUInt();
    bool      pINsymmetric   = jsIN.get( "symmetric", false).asBool();
    MPI_OUT std::cout << "RESTART from file "<<file_name<< std::endl;
    MPI_OUT std::cout << " file parameters:" << std::endl;
    MPI_OUT std::cout << pINn<<" x "<<pINNx<<" x "<<pINNy<<" x "<<pINNz<<" : symmetric "<<std::boolalpha<<pINsymmetric<<std::endl;

    // Now read in last timestep
    Geometry grid_IN( grid.x0(), grid.x1(), grid.y0(), grid.y1(), grid.z0(), grid.z1(),
        pINn, pINNx, pINNy, pINsymmetric ? 1 : pINNz, dg::DIR, dg::DIR, dg::PER
        #ifdef FELTOR_MPI
        , grid.communicator()
        #endif //FELTOR_MPI
        );
    IHMatrix interpolateIN = dg::create::interpolation( grid, grid_IN);

    #ifdef FELTOR_MPI
    int dimsIN[3],  coordsIN[3];
    int periods[3] = {false, false, true}; //non-, non-, periodic
    MPI_Cart_get( grid.communicator(), 3, dimsIN, periods, coordsIN);
    size_t countIN[3] = {grid_IN.local().Nz(),
            grid_IN.n()*(grid_IN.local().Ny()),
            grid_IN.n()*(grid_IN.local().Nx())};
    size_t startIN[3] = {coordsIN[2]*countIN[0],
                         coordsIN[1]*countIN[1],
                         coordsIN[0]*countIN[2]};
    #else //FELTOR_MPI
    size_t startIN[3] = {0, 0, 0};
    size_t countIN[3] = {grid_IN.Nz(), grid_IN.n()*grid_IN.Ny(),
        grid_IN.n()*grid_IN.Nx()};
    #endif //FELTOR_MPI
    std::vector<HVec> transferINHvec( 5, dg::evaluate( dg::zero, grid));
    HVec transferINH( dg::evaluate(dg::zero, grid_IN));

    std::string namesIN[5] = {"restart_electrons", "restart_ions", "restart_Ue", "restart_Ui", "restart_induction"};

    int timeIDIN;
    size_t size_time, count_time = 1;
    /////////////////////Get time length and initial data///////////////////////////
    errIN = nc_inq_dimid( ncidIN, "time", &timeIDIN);
    errIN = nc_inq_dimlen(ncidIN, timeIDIN, &size_time);
    errIN = nc_inq_varid( ncidIN, "time", &timeIDIN);
    size_time -= 1;
    errIN = nc_get_vara_double( ncidIN, timeIDIN, &size_time, &count_time, &time);
    MPI_OUT std::cout << " Current time = "<< time <<  std::endl;
    for( unsigned i=0; i<5; i++)
    {
        int dataID;
        errIN = nc_inq_varid( ncidIN, namesIN[i].data(), &dataID);
        errIN = nc_get_vara_double( ncidIN, dataID, startIN, countIN,
            #ifdef FELTOR_MPI
                transferINH.data().data()
            #else //FELTOR_MPI
                transferINH.data()
            #endif //FELTOR_MPI
            );
        dg::blas2::gemv( interpolateIN, transferINH, transferINHvec[i]);
    }
    errIN = nc_close(ncidIN);
    /// ///////////////Now Construct initial fields ////////////////////////
    //
    //Convert to N-1 and W
    dg::blas1::plus( transferINHvec[0], -1.);
    dg::blas1::plus( transferINHvec[1], -1.);
    dg::blas1::axpby( 1., transferINHvec[2], 1./p.mu[0], transferINHvec[4], transferINHvec[2]);
    dg::blas1::axpby( 1., transferINHvec[3], 1./p.mu[1], transferINHvec[4], transferINHvec[3]);

    dg::assign( transferINHvec[0], y0[0][0]); //ne-1
    dg::assign( transferINHvec[1], y0[0][1]); //Ni-1
    dg::assign( transferINHvec[2], y0[1][0]); //We
    dg::assign( transferINHvec[3], y0[1][1]); //Wi
    return y0;
}
}//namespace feltor
