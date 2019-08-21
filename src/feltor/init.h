#pragma once
#include "dg/file/nc_utilities.h"

namespace feltor
{

struct TorpexSource
{
    TorpexSource( double R0, double Z0, double a, double b, double c):m_R0(R0), m_Z0(Z0), m_a(a), m_b(b), m_c(c){}
    DG_DEVICE
    double operator()( double R, double Z) const{
        if( R > m_R0)
            return exp( - (R-m_R0)*(R-m_R0)/m_a/m_a - (Z-m_Z0)*(Z-m_Z0)/m_b/m_b);
        return 0.5*exp( - (R-m_R0)*(R-m_R0)/m_a/m_a -2.*m_c*(R-m_R0)*(Z-m_Z0)- (Z-m_Z0)*(Z-m_Z0)/m_b/m_b )
              +0.5*exp( - (R-m_R0)*(R-m_R0)/m_a/m_a +2.*m_c*(R-m_R0)*(Z-m_Z0)- (Z-m_Z0)*(Z-m_Z0)/m_b/m_b );
    }
    private:
    double m_R0, m_Z0, m_a, m_b, m_c;
};

//We use the typedefs and MPI_OUT
struct Initialize
{
    Initialize( feltor::Parameters p, dg::geo::solovev::Parameters gp,
        dg::geo::TokamakMagneticField mag) : p(p), gp(gp), mag(mag)
    {
    }
    HVec profile(const Geometry& grid)const{
        //First the profile and the source (on the host since we want to output those)
        HVec profile = dg::pullback( dg::geo::Compose<dg::LinearX>( mag.psip(),
            p.nprofamp/mag.psip()(mag.R0(), 0.), 0.), grid);
        dg::blas1::pointwiseDot( profile_damping(grid), profile, profile);
        return profile;
    }
    HVec xpoint_damping(const Geometry& grid)const{
        HVec xpoint_damping = dg::evaluate( dg::one, grid);
        if( gp.hasXpoint() )
            xpoint_damping = dg::pullback(
                dg::geo::ZCutter(-1.1*gp.elongation*gp.a), grid);
        return xpoint_damping;
    }
    HVec source_damping(const Geometry& grid)const{
        HVec source_damping = dg::pullback(dg::geo::Compose<dg::PolynomialHeaviside>(
            //first change coordinate from psi to (psi_0 - psip)/psi_0
            dg::geo::Compose<dg::LinearX>( mag.psip(), -1./mag.psip()(mag.R0(), 0.),1.),
            //then shift
            p.rho_source, p.alpha, -1), grid);
        dg::blas1::pointwiseDot( xpoint_damping(grid), source_damping, source_damping);
        return source_damping;
    }
    HVec damping_damping(const Geometry& grid)const{
        HVec damping_damping = dg::pullback(dg::geo::Compose<dg::PolynomialHeaviside>(
            //first change coordinate from psi to (psi_0 - psip)/psi_0
            dg::geo::Compose<dg::LinearX>( mag.psip(), -1./mag.psip()(mag.R0(), 0.),1.),
            //then shift
            p.rho_damping, p.alpha, +1), grid);
        return damping_damping;
    }
    HVec profile_damping(const Geometry& grid)const{
        HVec profile_damping = dg::pullback( dg::geo::Compose<dg::PolynomialHeaviside>(
            mag.psip(), -p.alpha, p.alpha, -1), grid);
        dg::blas1::pointwiseDot( xpoint_damping(grid), profile_damping, profile_damping);
        return profile_damping;
    }
    template<class Feltor>
    std::array<std::array<DVec,2>,2> init_from_parameters(Feltor& feltor, const Geometry& grid){
#ifdef FELTOR_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
        std::array<std::array<DVec,2>,2> y0;
        //Now perturbation
        HVec ntilde = dg::evaluate(dg::zero,grid);
        if( p.initne == "blob" || p.initne == "straight blob")
        {
            dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
            dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
            if( p.symmetric)
                ntilde = dg::pullback( init0, grid);
            else if( p.initne == "blob")//rounds =3 ->2*3-1
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 5, 5);
                //evaluate should always be used with mx,my > 1
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 3);
            }
            else if( p.initne == "straight blob")//rounds =1 ->2*1-1
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 5, 5);
                //evaluate should always be used with mx,my > 1
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 1);
            }
        }
        else if( p.initne == "turbulence")
        {
            dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
            dg::BathRZ init0(16,16,grid.x0(),grid.y0(), 30.,2.,p.amp);
            if( p.symmetric)
                ntilde = dg::pullback( init0, grid);
            else
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 5, 5);
                //evaluate should always be used with mx,my > 1
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 1);
            }
            dg::blas1::pointwiseDot( profile_damping(grid), ntilde, ntilde);
        }
        else if( p.initne == "zonal")
        {
            dg::geo::ZonalFlow init0(mag.psip(), p.amp, 0., p.k_psi);
            ntilde = dg::pullback( init0, grid);
            dg::blas1::pointwiseDot( profile_damping(grid), ntilde, ntilde);
        }
        else
            MPI_OUT std::cerr <<"WARNING: Unknown initial condition!\n";
        y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(profile(grid));
        dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0]);
        MPI_OUT std::cout << "initialize ni" << std::endl;
        feltor.initializeni( y0[0][0], y0[0][1], p.initphi);
        double minimalni = dg::blas1::reduce( y0[0][1], 1, thrust::minimum<double>());
        MPI_OUT std::cerr << "Minimum Ni value "<<minimalni+1<<std::endl;
        if( minimalni <= -1)
        {
            throw dg::Error(dg::Message()<< "ERROR: invalid initial condition. Increase value for alpha since now the ion gyrocentre density is negative!\n"
                << "Minimum Ni value "<<minimalni+1);
        }

        dg::blas1::copy( 0., y0[1][0]); //set we = 0
        dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
        return y0;
    }

    //everyone reads their portion of the input data
    std::array<std::array<DVec,2>,2> init_from_file( std::string file_name, const Geometry& grid, double& time){
#ifdef FELTOR_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
        std::array<std::array<DVec,2>,2> y0;
        ///////////////////read in and show inputfile
        file::NC_Error_Handle errIN;
        int ncidIN;
        errIN = nc_open( file_name.data(), NC_NOWRITE, &ncidIN);
        size_t lengthIN;
        errIN = nc_inq_attlen( ncidIN, NC_GLOBAL, "inputfile", &lengthIN);
        std::string inputIN( lengthIN, 'x');
        errIN = nc_get_att_text( ncidIN, NC_GLOBAL, "inputfile", &inputIN[0]);

        Json::Value jsIN;
        std::stringstream is(inputIN);
        Json::CharReaderBuilder parser;
        parser["collectComments"] = false;
        std::string errs;
        parseFromStream( parser, is, &jsIN, &errs); //read input without comments
        const feltor::Parameters pIN(  jsIN);
        MPI_OUT std::cout << "RESTART from file "<<file_name<< std::endl;
        MPI_OUT std::cout << " file parameters:" << std::endl;
        MPI_OUT pIN.display( std::cout);

        // Now read in last timestep
        Geometry grid_IN( grid.x0(), grid.x1(), grid.y0(), grid.y1(), grid.z0(), grid.z1(),
            pIN.n_out, pIN.Nx_out, pIN.Ny_out, pIN.symmetric ? 1 : pIN.Nz_out, pIN.bcxN, pIN.bcyN, dg::PER
            #ifdef FELTOR_MPI
            , grid.communicator()
            #endif //FELTOR_MPI
            );
        IHMatrix interpolateIN = dg::create::interpolation( grid, grid_IN);

        #ifdef FELTOR_MPI
        int dimsIN[3],  coordsIN[3];
        int periods[3] = {false, false, true}; //non-, non-, periodic
        MPI_Cart_get( grid.communicator(), 3, dimsIN, periods, coordsIN);
        size_t countIN[4] = {1, grid_IN.local().Nz(),
            grid_IN.n()*(grid_IN.local().Ny()),
            grid_IN.n()*(grid_IN.local().Nx())};
        size_t startIN[4] = {0, coordsIN[2]*countIN[1],
                                coordsIN[1]*countIN[2],
                                coordsIN[0]*countIN[3]};
        #else //FELTOR_MPI
        size_t startIN[4] = {0, 0, 0, 0};
        size_t countIN[4] = {1, grid_IN.Nz(), grid_IN.n()*grid_IN.Ny(),
            grid_IN.n()*grid_IN.Nx()};
        #endif //FELTOR_MPI
        std::vector<HVec> transferINHvec( 5, dg::evaluate( dg::zero, grid));
        HVec transferINH( dg::evaluate(dg::zero, grid_IN));

        std::string namesIN[5] = {"electrons", "ions", "Ue", "Ui", "induction"};

        int timeIDIN;
        /////////////////////Get time length and initial data///////////////////////////
        errIN = nc_inq_dimid( ncidIN, "time", &timeIDIN);
        errIN = nc_inq_dimlen(ncidIN, timeIDIN, &startIN[0]);
        startIN[0] -= 1;
        errIN = nc_inq_varid( ncidIN, "time", &timeIDIN);
        errIN = nc_get_vara_double( ncidIN, timeIDIN, startIN, countIN, &time);
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
        /// ///////////////Now Construct initial fields
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

    private:
    feltor::Parameters p;
    dg::geo::solovev::Parameters gp;
    dg::geo::TokamakMagneticField mag;
};


} //namespace feltor
