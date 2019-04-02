#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include "json/json.h"

#include "dg/algorithm.h"
#include "dg/file/nc_utilities.h"
#include "dg/geometries/geometries.h"
#include "feltor/feltor.cuh"
#include "feltor/parameters.h"

struct RadialParticleFlux{
    RadialParticleFlux( double tau, double mu):
        m_tau(tau), m_mu(mu){
    }

    DG_DEVICE double operator()( double ne, double ue, double A,
        double d0A, double d1A, double d2A,
        double d0P, double d1P, double d2P, //Phi
        double d0S, double d1S, double d2S, //Psip
        double b_0,         double b_1,         double b_2,
        double curvNabla0,  double curvNabla1,  double curvNabla2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double curvNablaS = curvNabla0*d0S+curvNabla1*d1S+curvNabla2*d2S;
        double SA = b_0*( d1S*d2A-d2S*d1A)+
                    b_1*( d2S*d0A-d0S*d2A)+
                    b_2*( d0S*d1A-d1S*d0A);
        double PS = b_0*( d1P*d2S-d2P*d1S)+
                    b_1*( d2P*d0S-d0P*d2S)+
                    b_2*( d0P*d1S-d1P*d0S);
        double JPsi =
            ne*ue* (A*curvKappaS + SA )
            + ne * PS
            + ne * (m_tau + m_mu*ue*ue)*curvKappaS
            + ne * m_tau*curvNablaS;
        return JPsi;
    }
    private:
    double m_tau, m_mu;
};
struct RadialEnergyFlux{
    RadialEnergyFlux( double tau, double mu):
        m_tau(tau), m_mu(mu){
    }

    DG_DEVICE double operator()( double ne, double ue, double A, double P,
        double d0A, double d1A, double d2A,
        double d0P, double d1P, double d2P, //Phi
        double d0S, double d1S, double d2S, //Psip
        double b_0,         double b_1,         double b_2,
        double curvNabla0,  double curvNabla1,  double curvNabla2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double curvNablaS = curvNabla0*d0S+curvNabla1*d1S+curvNabla2*d2S;
        double SA = b_0*( d1S*d2A-d2S*d1A)+
                    b_1*( d2S*d0A-d0S*d2A)+
                    b_2*( d0S*d1A-d1S*d0A);
        double PS = b_0*( d1P*d2S-d2P*d1S)+
                    b_1*( d2P*d0S-d0P*d2S)+
                    b_2*( d0P*d1S-d1P*d0S);
        double JN =
            ne*ue* (A*curvKappaS + SA )
            + ne * PS
            + ne * (m_tau + m_mu*ue*ue)*curvKappaS
            + ne * m_tau*curvNablaS;
        double Je = (m_tau * log(ne) + 0.5*m_mu*ue*ue + P)*JN
            + m_mu*m_tau*ne*ue*ue*curvKappaS
            + m_tau*ne*ue* (A*curvKappaS + SA );
        return Je;
    }
    private:
    double m_tau, m_mu;
};

int main( int argc, char* argv[])
{
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [output.nc]\n";
        return -1;
    }
    std::cout << argv[1]<< " -> "<<argv[2]<<std::endl;

    //------------------------open input nc file--------------------------------//
    file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input( length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
    err = nc_inq_attlen( ncid, NC_GLOBAL, "geomfile", &length);
    std::string geom( length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "geomfile", &geom[0]);
    err = nc_close(ncid);

    std::cout << "input "<<input<<std::endl;
    std::cout << "geome "<<geom <<std::endl;
    Json::Value js,gs;
    Json::CharReaderBuilder parser;
    parser["collectComments"] = false;
    std::string errs;
    std::stringstream ss( input);
    parseFromStream( parser, ss, &js, &errs); //read input without comments
    ss.str( geom);
    parseFromStream( parser, ss, &gs, &errs); //read input without comments
    const feltor::Parameters p(js);
    const dg::geo::solovev::Parameters gp(gs);
    p.display();
    gp.display();
    //-------------------Construct grids-------------------------------------//

    const double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    const double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    const double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    const double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    const double Z_X = -1.1*gp.elongation*gp.a;

    dg::CylindricalGrid3d g3d_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n_out, p.Nx_out, p.Ny_out, p.Nz_out, p.bcxN, p.bcyN, dg::PER);
    dg::Grid2d   g2d_out( Rmin,Rmax, Zmin,Zmax,
        p.n_out, p.Nx_out, p.Ny_out, p.bcxN, p.bcyN);

    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    const double psip0 = mag.psip()(gp.R_0, 0);
    mag = dg::geo::createModifiedSolovevField(gp, (1.-p.rho_damping)*psip0, p.alpha);
    dg::HVec psipog2d = dg::evaluate( mag.psip(), g2d_out);
    //double psipmin = (double)thrust::reduce(
    //    psipog2d.begin(), psipog2d.end(), 0.0,thrust::minimum<double>() );
    double psipmin = dg::blas1::reduce( psipog2d, 0,thrust::minimum<double>() );
    double psipmax = dg::blas1::reduce( psipog2d, -1000,thrust::maximum<double>() );

    unsigned Npsi = 50;//set number of psivalues
    dg::Grid1d g1d_out(psipmin, psipmax, 3, Npsi, dg::NEU);

    // Construct weights and temporaries

    const dg::DVec w3d = dg::create::volume( g3d_out);
    const dg::DVec w1d = dg::create::weights( g1d_out);

    dg::HVec transfer3d = dg::evaluate(dg::zero,g3d_out);
    dg::HVec transfer2d = dg::evaluate(dg::zero,g2d_out);
    dg::HVec transfer1d = dg::evaluate(dg::zero,g1d_out);

    dg::DVec t1d = dg::evaluate( dg::zero, g1d_out);
    dg::DVec t2d = dg::evaluate( dg::zero, g2d_out);
    dg::DVec t3d = dg::evaluate( dg::zero, g3d_out);


    // interpolate fsa back to 2d or 3d grid

    dg::IDMatrix fsaonrzmatrix;
    fsaonrzmatrix = dg::create::interpolation(psipog2d, g1d_out);

    //perp laplacian for computation of vorticity

    dg::DVec vor3d    = dg::evaluate( dg::zero, g3d_out);
    dg::Elliptic3d<dg::CylindricalGrid3d, dg::DMatrix, dg::DVec>
        laplacianM(g3d_out, p.bcxP, p.bcyP, dg::PER, dg::normed, dg::centered);
    auto bhatF = dg::geo::createEPhi();
    if( p.curvmode == "true")
        bhatF = dg::geo::createBHat( mag);
    dg::SparseTensor<dg::DVec> hh = dg::geo::createProjectionTensor( bhatF, g3d_out);
    laplacianM.set_chi( hh);

    // The curvature vectors

    dg::geo::CylindricalVectorLvl0 curvNablaF, curvKappaF;
    if( p.curvmode == "true" )
    {
        curvNablaF = dg::geo::createTrueCurvatureNablaB(mag);
        curvKappaF = dg::geo::createTrueCurvatureKappa(mag);
    }
    else if( p.curvmode == "low beta")
    {
        curvNablaF = curvKappaF = dg::geo::createCurvatureNablaB(mag);
    }
    else if( p.curvmode == "toroidal")
    {
        curvNablaF = dg::geo::createCurvatureNablaB(mag);
        curvKappaF = dg::geo::createCurvatureKappa(mag);
    }
    std::array<dg::DVec, 3> curvNabla, curvKappa, bhat;
    dg::pushForward(curvNablaF.x(), curvNablaF.y(), curvNablaF.z(),
        curvNabla[0], curvNabla[1], curvNabla[2], g3d_out);
    dg::pushForward(curvKappaF.x(), curvKappaF.y(), curvKappaF.z(),
        curvKappa[0], curvKappa[1], curvKappa[2], g3d_out);

    //in DS we take the true bhat

    bhatF = dg::geo::createBHat( mag);
    dg::geo::DS<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec>
        dsN( bhatF, g3d_out, p.bcxN, p.bcyN, dg::geo::NoLimiter(),
        dg::forward, p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz );

    // create ingredients for perp Poisson bracket

    bhatF = dg::geo::createEPhi();
    if( p.curvmode == "true")
        bhatF = dg::geo::createBHat(mag);
    dg::pushForward(bhatF.x(), bhatF.y(), bhatF.z(),
        bhat[0], bhat[1], bhat[2], g3d_out);
    dg::SparseTensor<dg::DVec> metric = g3d_out.metric();
    dg::tensor::inv_multiply3d( metric, bhat[0], bhat[1], bhat[2],
                                        bhat[0], bhat[1], bhat[2]);
    dg::DVec vol = dg::tensor::volume( metric);
    const dg::DVec binv = dg::evaluate( dg::geo::InvB(mag) , g3d_out) ;
    dg::blas1::pointwiseDivide( binv, vol, vol); //1/vol/B
    for( int i=0; i<3; i++)
        dg::blas1::pointwiseDot( vol, bhat[i], bhat[i]); //b_i/vol/B
    dg::DMatrix dxA = dg::create::dx( g3d_out, p.bcxU);
    dg::DMatrix dyA = dg::create::dy( g3d_out, p.bcyU);
    dg::DMatrix dxP = dg::create::dx( g3d_out, p.bcxP);
    dg::DMatrix dyP = dg::create::dy( g3d_out, p.bcyP);
    dg::DMatrix dxN = dg::create::dx( g3d_out, p.bcxN);
    dg::DMatrix dyN = dg::create::dy( g3d_out, p.bcyN);
    dg::DMatrix dz = dg::create::dz( g3d_out, dg::PER);
    dg::DVec dx_A( t3d), dy_A(t3d), dz_A(t3d);
    dg::DVec dx_P( t3d), dy_P(t3d), dz_P(t3d);
    dg::DVec dx_N( t3d), dy_N(t3d), dz_N(t3d);
    const dg::DVec psipR =  dg::evaluate( mag.psipR(), g3d_out);
    const dg::DVec psipZ =  dg::evaluate( mag.psipZ(), g3d_out);
    const dg::DVec psipP =  t3d; //zero

    /////////////////// Construct names to output /////////////
    std::vector<std::string> names_direct{
        "electrons","ions","Ue","Ui", "potential","induction"
    };
    std::vector<std::string> names_derived{
        "vorticity","fluxe","Lperpinv","Lparallelinv"
    };
    std::vector<std::string> names_0d{
        "aligned", "perp_aligned", "correlationNPhi", "total_flux"
    };
    std::map<std::string, dg::DVec> v3d;
    for( auto name : names_direct)
        v3d[name] = t3d;
    for( auto name : names_derived)
        v3d[name] = t3d;

    // Create Netcdf output file and ids
    int ncid_out;
    err = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid_out);
    err = nc_put_att_text( ncid_out, NC_GLOBAL, "inputfile",
        input.size(), input.data());
    err = nc_put_att_text( ncid_out, NC_GLOBAL, "geomfile",
        geom.size(), geom.data());
    // define 2d and 1d and 0d dimensions and variables
    int dim_ids[3], tvarID;
    err = file::define_dimensions( ncid_out, dim_ids, &tvarID, g2d_out);
    int dim_ids1d[2] = {dim_ids[0],dim_ids[1]}; //time , psi
    err = file::define_dimension( ncid_out, "psi", &dim_ids1d[1], g1d_out);

    std::map<std::string, int> id0d, id1d, id2d;
    for( auto pair : v3d)
    {
        std::string name = pair.first + "_avg";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 3, dim_ids,
            &id2d[name]);
        name = pair.first + "_fsa_mp";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 3, dim_ids,
            &id2d[name]);
        name = pair.first + "_fsa";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 2, dim_ids1d,
            &id1d[name]);
    }
    std::map<std::string, double> v0d;
    for( auto name : names_0d)
    {
        v0d[name] = 0.;
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 1, dim_ids,
            &id0d[name]);
    }

    size_t count1d[2] = {1, g1d_out.n()*g1d_out.N()};
    size_t start1d[2] = {0, 0};
    size_t count2d[3] = {1, g2d_out.n()*g2d_out.Ny(), g2d_out.n()*g2d_out.Nx()};
    size_t start2d[3] = {0, 0, 0};
    size_t count3d[4] = {1, g3d_out.Nz(), g3d_out.n()*g3d_out.Ny(), g3d_out.n()*g3d_out.Nx()};
    size_t start3d[4] = {0, 0, 0, 0};

    //put safety factor and rho into file
    std::cout << "Compute safety factor   "<< "\n";
    dg::HVec xpoint_damping = dg::evaluate( dg::one, g2d_out);
    if( gp.hasXpoint())
        xpoint_damping = dg::evaluate( dg::geo::ZCutter(Z_X), g2d_out);
    dg::geo::SafetyFactor qprofile(mag);
    dg::HVec sf = dg::evaluate(qprofile, g1d_out);
    int qID, rhoID;
    err = nc_def_var( ncid_out, "q", NC_DOUBLE, 1, &dim_ids1d[1], &qID);
    err = nc_def_var( ncid_out, "rho", NC_DOUBLE, 1, &dim_ids1d[1], &rhoID);
    err = nc_enddef(ncid_out);
    err = nc_put_vara_double( ncid_out, qID, &start1d[1], &count1d[1], sf.data());
    dg::HVec rho = dg::evaluate( dg::cooX1d, g1d_out);//evaluate psi
    dg::blas1::axpby( -1./psip0, rho, +1., 1., rho); //transform psi to rho
    err = nc_put_vara_double( ncid_out, rhoID, &start1d[1], &count1d[1], rho.data());
    err = nc_close(ncid_out);

    /////////////////////////////////////////////////////////////////////////
    int timeID;
    double time=0.;

    size_t steps;
    err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
    err = nc_inq_unlimdim( ncid, &timeID); //Attention: Finds first unlimited dim, which hopefully is time and not energy_time
    err = nc_inq_dimlen( ncid, timeID, &steps);
    err = nc_close( ncid); //close 3d file

    dg::Average<dg::DVec> toroidal_average( g3d_out, dg::coo3d::z);
    dg::geo::FluxSurfaceAverage<dg::DVec> fsa(g2d_out, mag, t2d, dg::DVec(xpoint_damping) );

    for( unsigned i=0; i<steps; i++)//timestepping
    {
        start3d[0] = i;
        start2d[0] = i;
        start1d[0] = i;
        err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
        err = nc_get_vara_double( ncid, timeID, start3d, count3d, &time);
        std::cout << "Timestep = " << i << "  time = " << time << "\n";
        //read in Ne,Ni,Ue,Ui,Phi,Apar
        for( auto name : names_direct)
        {
            int dataID;
            err = nc_inq_varid(ncid, name.data(), &dataID);
            err = nc_get_vara_double( ncid, dataID, start3d, count3d,
                transfer3d.data());
            dg::assign( transfer3d, v3d[name]);
        }
        err = nc_close(ncid);  //close 3d file

        //----------------vorticity computation

        dg::blas2::gemv( laplacianM, v3d["potential"], v3d["vorticity"]);

        //----------------radial flux computation

        dg::blas2::symv( dxA, v3d["induction"], dx_A);
        dg::blas2::symv( dyA, v3d["induction"], dy_A);
        dg::blas2::symv( dz , v3d["induction"], dz_A);
        dg::blas2::symv( dxP, v3d["potential"], dx_P);
        dg::blas2::symv( dyP, v3d["potential"], dy_P);
        dg::blas2::symv( dz , v3d["potential"], dz_P);
        dg::blas1::evaluate( v3d["fluxe"], dg::equals(),
            RadialParticleFlux( p.tau[0], p.mu[0]),
            v3d["electrons"], v3d["Ue"], v3d["induction"],
            dx_A, dy_A, dz_A, dx_P, dy_P, dz_P, psipR, psipZ, psipP,
            bhat[0], bhat[1], bhat[2],
            curvNabla[0], curvNabla[1], curvNabla[2],
            curvKappa[0], curvKappa[1], curvKappa[2]
        );
        v0d["total_flux"] = dg::blas1::dot( w3d, v3d["fluxe"]);

        //----------------perp length scale computation

        dg::blas1::axpby( 1., v3d["electrons"], -1., 1., t3d);
        dg::blas2::symv( dxN, t3d, dx_N);
        dg::blas2::symv( dyN, t3d, dy_N);
        dg::blas2::symv( dz , t3d, dz_N);
        dg::tensor::multiply3d( hh, //grad_perp
            dx_N, dy_N, dz_N, dx_A, dy_A, dz_A);
        dg::blas1::subroutine( feltor::routines::ComputePsi(),
            t3d, dx_N, dy_N, dz_N, dx_A, dy_A, dz_A);
        dg::blas1::pointwiseDivide( t3d, v3d["electrons"], t3d);
        v0d["perp_aligned"] = dg::blas1::dot( t3d, w3d);
        dg::blas1::pointwiseDivide( t3d, v3d["electrons"], t3d);
        dg::blas1::transform( t3d, v3d["Lperpinv"], dg::SQRT<double>());

        //----------------parallel length scale computation

        dg::blas1::axpby( 1., v3d["electrons"], -1., 1., t3d);
        dsN.centered( t3d, dx_N);
        dg::blas1::pointwiseDot ( dx_N, dx_N, t3d);
        dg::blas1::pointwiseDivide( t3d, v3d["electrons"], t3d);
        v0d["aligned"] = dg::blas1::dot( t3d, w3d);
        dg::blas1::pointwiseDivide( t3d, v3d["electrons"], t3d);
        dg::blas1::transform( t3d, v3d["Lparallelinv"], dg::SQRT<double>());

        //------------------correlation------------//

        dg::blas1::transform( v3d["potential"], t3d, dg::EXP<double>());
        double norm1 = sqrt(dg::blas2::dot(t3d, w3d, t3d));
        double norm2 = sqrt(dg::blas2::dot(v3d["electrons"], w3d, v3d["electrons"]));
        v0d["correlationNPhi"] = dg::blas2::dot( t3d, w3d, v3d["electrons"])
            /norm1/norm2;  //<e^phi, N>/||e^phi||/||N||


        //now write out 2d and 1d quantities
        err = nc_open(argv[2], NC_WRITE, &ncid_out);
        for( auto pair : v3d)// {name, DVec}
        {
            toroidal_average( pair.second, t2d, false);
            dg::blas1::transfer( t2d, transfer2d);
            err = nc_put_vara_double( ncid_out, id2d.at(pair.first+"_avg"),
                start2d, count2d, transfer2d.data());

            //computa fsa of quantities
            fsa.set_container(t2d);
            transfer1d = dg::evaluate(fsa, g1d_out);
            err = nc_put_vara_double( ncid_out, id1d.at(pair.first+"_fsa"),
                start1d, count1d, transfer1d.data());

            //get 2d data of MidPlane
            unsigned kmp = g3d_out.Nz()/2;
            dg::DVec t2d_mp(pair.second.begin() + kmp*g2d_out.size(),
                pair.second.begin() + (kmp+1)*g2d_out.size() );

            //compute delta f on midplane : df = f_mp - <f>
            dg::assign( transfer1d, t1d);
            dg::blas2::gemv(fsaonrzmatrix, t1d, t2d); //fsa on RZ grid
            dg::blas1::axpby( 1.0, t2d_mp, -1.0, t2d);
            dg::assign( t2d, transfer2d);
            err = nc_put_vara_double( ncid_out, id2d.at(pair.first+"_fsa_mp"),
                start2d, count2d, transfer2d.data() );

        }
        //and the 0d quantities
        for( auto pair : v0d) //{name, double}
            err = nc_put_vara_double( ncid_out, id0d.at(pair.first),
                start2d, count2d, &pair.second );
        //write time data
        err = nc_put_vara_double( ncid_out, tvarID, start2d, count2d, &time);
        //and close file
        err = nc_close(ncid_out);

    } //end timestepping
    //relative fluctuation amplitude(R,Z,phi) = delta n(R,Z,phi)/n0(psi)

    return 0;
}
