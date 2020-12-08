#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include "json/json.h"

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/file.h"
using HVec = dg::HVec;
using DVec = dg::DVec;
using DMatrix = dg::DMatrix;
using IDMatrix = dg::IDMatrix;
using IHMatrix = dg::IHMatrix;
using Geometry = dg::CylindricalGrid3d;
#define MPI_OUT
#include "feltordiag.h"

int main( int argc, char* argv[])
{
    if( argc < 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input0.nc ... inputN.nc] [output.nc]\n";
        return -1;
    }
    for( int i=1; i<argc-1; i++)
        std::cout << argv[i]<< " ";
    std::cout << " -> "<<argv[argc-1]<<std::endl;

    //------------------------open input nc file--------------------------------//
    file::NC_Error_Handle err;
    int ncid_in;
    err = nc_open( argv[1], NC_NOWRITE, &ncid_in); //open 3d file
    size_t length;
    err = nc_inq_attlen( ncid_in, NC_GLOBAL, "inputfile", &length);
    std::string inputfile(length, 'x');
    err = nc_get_att_text( ncid_in, NC_GLOBAL, "inputfile", &inputfile[0]);
    err = nc_inq_attlen( ncid_in, NC_GLOBAL, "geomfile", &length);
    std::string geomfile(length, 'x');
    err = nc_get_att_text( ncid_in, NC_GLOBAL, "geomfile", &geomfile[0]);
    err = nc_close( ncid_in);
    Json::Value js,gs;
    file::string2Json(inputfile, js, file::comments::are_forbidden);
    file::string2Json(geomfile, gs, file::comments::are_forbidden);
    //we only need some parameters from p, not all
    const feltor::Parameters p(js, file::error::is_warning);
    const dg::geo::solovev::Parameters gp(gs);
    p.display();
    gp.display();
    std::vector<std::string> names_input{
        "electrons", "ions", "Ue", "Ui", "potential", "induction"
    };

    //-----------------Create Netcdf output file with attributes----------//
    int ncid_out;
    err = nc_create(argv[argc-1],NC_NETCDF4|NC_NOCLOBBER, &ncid_out);

    /// Set global attributes
    std::map<std::string, std::string> att;
    att["title"] = "Output file of feltor/src/feltor/feltordiag.cu";
    att["Conventions"] = "CF-1.7";
    ///Get local time and begin file history
    auto ttt = std::time(nullptr);
    auto tm = *std::localtime(&ttt);
    std::ostringstream oss;
    ///time string  + program-name + args
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    for( int i=0; i<argc; i++) oss << " "<<argv[i];
    att["history"] = oss.str();
    att["comment"] = "Find more info in feltor/src/feltor.tex";
    att["source"] = "FELTOR";
    att["references"] = "https://github.com/feltor-dev/feltor";
    att["inputfile"] = inputfile;
    att["geomfile"] = geomfile;
    for( auto pair : att)
        err = nc_put_att_text( ncid_out, NC_GLOBAL,
            pair.first.data(), pair.second.size(), pair.second.data());

    //-------------------Construct grids-------------------------------------//

    const double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    const double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    const double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    const double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    const unsigned FACTOR=10;

    dg::Grid2d g2d_out( Rmin,Rmax, Zmin,Zmax,
        p.n_out, p.Nx_out, p.Ny_out, p.bcxN, p.bcyN);
    /////////////////////////////////////////////////////////////////////////
    Geometry g3d( Rmin, Rmax, Zmin, Zmax, 0., 2.*M_PI,
        p.n_out, p.Nx_out, p.Ny_out, p.Nz, p.bcxN, p.bcyN, dg::PER);
    Geometry g3d_fine( Rmin, Rmax, Zmin, Zmax, 0., 2.*M_PI,
        p.n_out, p.Nx_out, p.Ny_out, FACTOR*p.Nz, p.bcxN, p.bcyN, dg::PER);

    dg::geo::CylindricalFunctor wall, transition;
    dg::geo::TokamakMagneticField mag =
        dg::geo::createModifiedField(gs, js, file::error::is_warning, wall, transition);
    dg::HVec psipog2d = dg::evaluate( mag.psip(), g2d_out);
    // Construct weights and temporaries

    dg::HVec transferH2d = dg::evaluate(dg::zero,g2d_out);
    dg::HVec t2d_mp = dg::evaluate(dg::zero,g2d_out);
    std::cout << "Construct Fieldaligned derivative ... \n";

    auto bhat = dg::geo::createBHat( mag);
    dg::geo::Fieldaligned<Geometry, IDMatrix, DVec> fieldaligned(
        bhat, g3d_fine, dg::NEU, dg::NEU, dg::geo::NoLimiter(), //let's take NEU bc because N is not homogeneous
        p.rk4eps, 5, 5);


    ///--------------- Construct X-point grid ---------------------//


    //std::cout << "Type X-point grid resolution (n(3), Npsi(32), Neta(640)) Must be divisible by 8\n";
    //we use so many Neta so that we get close to the X-point
    std::cout << "Using default X-point grid resolution (n(3), Npsi(64), Neta(640))\n";
    unsigned npsi = 3, Npsi = 64, Neta = 640;//set number of psivalues (NPsi % 8 == 0)
    //std::cin >> npsi >> Npsi >> Neta;
    std::cout << "You typed "<<npsi<<" x "<<Npsi<<" x "<<Neta<<"\n";
    std::cout << "Generate X-point flux-aligned grid!\n";
    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    dg::geo::findXpoint( mag.get_psip(), R_X, Z_X);
    dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( mag.get_psip(), R_X, Z_X) ;
    double R_O = gp.R_0, Z_O = 0;
    dg::geo::findOpoint( mag.get_psip(), R_O, Z_O);
    double psipO = mag.psip()(R_O, Z_O);

    dg::geo::SeparatrixOrthogonal generator(mag.get_psip(), monitor_chi, psipO, R_X, Z_X, mag.R0(), 0, 0, false);
    double fx_0 = 1./8.;
    double psipmax = dg::blas1::reduce( psipog2d, 0. ,thrust::maximum<double>()); //DEPENDS ON GRID RESOLUTION!!
    std::cout << "psi max is            "<<psipmax<<"\n";
    psipmax = -fx_0/(1.-fx_0)*psipO;
    std::cout << "psi max in g1d_out is "<<psipmax<<"\n";
    dg::geo::CurvilinearGridX2d gridX2d( generator, fx_0, 0., npsi, Npsi, Neta, dg::DIR_NEU, dg::NEU);
    std::cout << "psi max in gridX2d is "<<gridX2d.x1()<<"\n";
    std::cout << "DONE!\n";
    //Create 1d grid
    dg::Grid1d g1d_out(psipO, psipmax, npsi, Npsi, dg::DIR_NEU); //inner value is always 0
    std::cout << "Cell separatrix boundary is "<<Npsi*(1.-fx_0)*g1d_out.h()+g1d_out.x0()<<"\n";
    const double f0 = ( gridX2d.x1() - gridX2d.x0() ) / ( psipmax - psipO );
    dg::HVec t1d = dg::evaluate( dg::zero, g1d_out), fsa1d( t1d);
    dg::HVec transfer1d = dg::evaluate(dg::zero,g1d_out);

    /// ------------------- Compute 1d flux labels ---------------------//

    std::vector<std::tuple<std::string, dg::HVec, std::string> > map1d;
    /// Compute flux volume label
    dg::Average<dg::HVec > poloidal_average( gridX2d.grid(), dg::coo2d::y);
    dg::HVec dvdpsip;
    //metric and map
    dg::SparseTensor<dg::HVec> metricX = gridX2d.metric();
    std::vector<dg::HVec > coordsX = gridX2d.map();
    dg::HVec volX2d = dg::tensor::volume2d( metricX);
    dg::HVec transferH2dX(volX2d);
    dg::blas1::pointwiseDot( coordsX[0], volX2d, volX2d); //R\sqrt{g}
    poloidal_average( volX2d, dvdpsip, false);
    dg::blas1::scal( dvdpsip, 4.*M_PI*M_PI*f0);
    map1d.emplace_back( "dvdpsi", dvdpsip,
        "Derivative of flux volume with respect to flux label psi");
    dg::HVec X_psi_vol = dg::integrate( dvdpsip, g1d_out);
    map1d.emplace_back( "psi_vol", X_psi_vol,
        "Flux volume evaluated with X-point grid");

    /// Compute flux area label
    dg::HVec gradZetaX = metricX.value(0,0), X_psi_area;
    dg::blas1::transform( gradZetaX, gradZetaX, dg::SQRT<double>());
    dg::blas1::pointwiseDot( volX2d, gradZetaX, gradZetaX); //R\sqrt{g}|\nabla\zeta|
    poloidal_average( gradZetaX, X_psi_area, false);
    dg::blas1::scal( X_psi_area, 4.*M_PI*M_PI);
    map1d.emplace_back( "psi_area", X_psi_area,
        "Flux area evaluated with X-point grid");

    dg::HVec rho = dg::evaluate( dg::cooX1d, g1d_out);
    dg::blas1::axpby( -1./psipO, rho, +1., 1., rho); //transform psi to rho
    map1d.emplace_back("rho", rho,
        "Alternative flux label rho = 1-psi/psimin");
    dg::blas1::transform( rho, rho, dg::SQRT<double>());
    map1d.emplace_back("rho_p", rho,
        "Alternative flux label rho_p = sqrt(1-psi/psimin)");
    dg::geo::SafetyFactor qprof( mag);
    dg::HVec qprofile = dg::evaluate( qprof, g1d_out);
    map1d.emplace_back("q-profile", qprofile,
        "q-profile (Safety factor) using direct integration");
    map1d.emplace_back("psi_psi",    dg::evaluate( dg::cooX1d, g1d_out),
        "Poloidal flux label psi (same as coordinate)");
    dg::HVec psit = dg::integrate( qprofile, g1d_out);
    map1d.emplace_back("psit1d", psit,
        "Toroidal flux label psi_t integrated using q-profile");
    //we need to avoid integrating >=0 for total psi_t
    dg::Grid1d g1d_fine(psipO<0. ? psipO : 0., psipO<0. ? 0. : psipO, npsi ,Npsi,dg::DIR_NEU);
    qprofile = dg::evaluate( qprof, g1d_fine);
    dg::HVec w1d = dg::create::weights( g1d_fine);
    double psit_tot = dg::blas1::dot( w1d, qprofile);
    dg::blas1::scal ( psit, 1./psit_tot);
    dg::blas1::transform( psit, psit, dg::SQRT<double>());
    map1d.emplace_back("rho_t", psit,
        "Toroidal flux label rho_t = sqrt( psit/psit_tot)");

    // interpolate from 2d grid to X-point points
    dg::IHMatrix grid2gridX2d  = dg::create::interpolation(
        coordsX[0], coordsX[1], g2d_out);
    // interpolate fsa back to 2d or 3d grid
    dg::IHMatrix fsa2rzmatrix = dg::create::interpolation(
        psipog2d, g1d_out, dg::DIR_NEU);

    dg::HVec dvdpsip2d = dg::evaluate( dg::zero, g2d_out);
    dg::blas2::symv( fsa2rzmatrix, dvdpsip, dvdpsip2d);
    dg::HMatrix dpsi = dg::create::dx( g1d_out, dg::DIR_NEU, dg::backward); //we need to avoid involving cells outside LCFS in computation (also avoids right boundary)
    //although the first point outside LCFS is still wrong

    // define 2d and 1d and 0d dimensions and variables
    int dim_ids[3], tvarID;
    err = file::define_dimensions( ncid_out, dim_ids, &tvarID, g2d_out);
    //Write long description
    std::string long_name = "Time at which 2d fields are written";
    err = nc_put_att_text( ncid_out, tvarID, "long_name", long_name.size(),
            long_name.data());
    int dim_ids1d[2] = {dim_ids[0], 0}; //time,  psi
    err = file::define_dimension( ncid_out, &dim_ids1d[1], g1d_out, {"psi"} );
    std::map<std::string, int> id0d, id1d, id2d;

    size_t count1d[2] = {1, g1d_out.n()*g1d_out.N()};
    size_t count2d[3] = {1, g2d_out.n()*g2d_out.Ny(), g2d_out.n()*g2d_out.Nx()};
    size_t start2d[3] = {0, 0, 0};

    //write 1d static vectors (psi, q-profile, ...) into file
    for( auto tp : map1d)
    {
        int vid;
        err = nc_def_var( ncid_out, std::get<0>(tp).data(), NC_DOUBLE, 1,
            &dim_ids1d[1], &vid);
        err = nc_put_att_text( ncid_out, vid, "long_name",
            std::get<2>(tp).size(), std::get<2>(tp).data());
        err = nc_enddef( ncid_out);
        err = nc_put_var_double( ncid_out, vid, std::get<1>(tp).data());
        err = nc_redef(ncid_out);
    }

    for( auto& record : feltor::diagnostics2d_list)
    {
        std::string record_name = record.name;
        if( record_name[0] == 'j')
            record_name[1] = 'v';
        std::string name = record_name + "_fluc2d";
        long_name = record.long_name + " (Fluctuations wrt fsa on phi = 0 plane.)";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 3, dim_ids,
            &id2d[name]);
        err = nc_put_att_text( ncid_out, id2d[name], "long_name", long_name.size(),
            long_name.data());

        name = record_name + "_cta2d";
        long_name = record.long_name + " (Convoluted toroidal average on 2d plane.)";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 3, dim_ids,
            &id2d[name]);
        err = nc_put_att_text( ncid_out, id2d[name], "long_name", long_name.size(),
            long_name.data());

        name = record_name + "_fsa2d";
        long_name = record.long_name + " (Flux surface average interpolated to 2d plane.)";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 3, dim_ids,
            &id2d[name]);
        err = nc_put_att_text( ncid_out, id2d[name], "long_name", long_name.size(),
            long_name.data());

        name = record_name + "_fsa";
        long_name = record.long_name + " (Flux surface average.)";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 2, dim_ids1d,
            &id1d[name]);
        err = nc_put_att_text( ncid_out, id1d[name], "long_name", long_name.size(),
            long_name.data());
        name = record_name + "_std_fsa";
        long_name = record.long_name + " (Flux surface average standard deviation on outboard midplane.)";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 2, dim_ids1d,
            &id1d[name]);
        err = nc_put_att_text( ncid_out, id1d[name], "long_name", long_name.size(),
            long_name.data());

        name = record_name + "_ifs";
        long_name = record.long_name + " (wrt. vol integrated flux surface average)";
        if( record_name[0] == 'j')
            long_name = record.long_name + " (wrt. vol derivative of the flux surface average)";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 2, dim_ids1d,
            &id1d[name]);
        err = nc_put_att_text( ncid_out, id1d[name], "long_name", long_name.size(),
            long_name.data());

        name = record_name + "_ifs_lcfs";
        long_name = record.long_name + " (wrt. vol integrated flux surface average evaluated on last closed flux surface)";
        if( record_name[0] == 'j')
            long_name = record.long_name + " (flux surface average evaluated on the last closed flux surface)";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 1, dim_ids,
            &id0d[name]);
        err = nc_put_att_text( ncid_out, id0d[name], "long_name", long_name.size(),
            long_name.data());

        name = record_name + "_ifs_norm";
        long_name = record.long_name + " (wrt. vol integrated square flux surface average from 0 to lcfs)";
        if( record_name[0] == 'j')
            long_name = record.long_name + " (wrt. vol integrated square derivative of the flux surface average from 0 to lcfs)";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 1, dim_ids,
            &id0d[name]);
        err = nc_put_att_text( ncid_out, id0d[name], "long_name", long_name.size(),
            long_name.data());
    }
    /////////////////////////////////////////////////////////////////////////
    size_t counter = 0;
    int ncid;
    for( int j=1; j<argc-1; j++)
    {
        int timeID;

        size_t steps;
        std::cout << "Opening file "<<argv[j]<<"\n";
        try{
            err = nc_open( argv[j], NC_NOWRITE, &ncid); //open 3d file
        } catch ( file::NC_Error& error)
        {
            std::cerr << "An error occurded opening file "<<argv[j]<<"\n";
            std::cerr << error.what()<<std::endl;
            std::cerr << "Continue with next file\n";
            continue;
        }
        err = nc_inq_unlimdim( ncid, &timeID); //Attention: Finds first unlimited dim, which hopefully is time and not energy_time
        err = nc_inq_dimlen( ncid, timeID, &steps);
        //steps = 3;
        for( unsigned i=0; i<steps; i++)//timestepping
        {
            if( j > 1 && i == 0)
                continue; // else we duplicate the first timestep
            start2d[0] = i;
            size_t start2d_out[3] = {counter, 0,0};
            size_t start1d_out[2] = {counter, 0};
            // read and write time
            double time=0.;
            err = nc_get_vara_double( ncid, timeID, start2d, count2d, &time);
            std::cout << counter << " Timestep = " << i <<"/"<<steps-1 << "  time = " << time << std::endl;
            counter++;
            err = nc_put_vara_double( ncid_out, tvarID, start2d_out, count2d, &time);
            for( auto& record : feltor::diagnostics2d_list)
            {
                std::string record_name = record.name;
                if( record_name[0] == 'j')
                    record_name[1] = 'v';
                //1. Read toroidal average
                int dataID =0;
                bool available = true;
                try{
                    err = nc_inq_varid(ncid, (record.name+"_ta2d").data(), &dataID);
                } catch ( file::NC_Error& error)
                {
                    if(  i == 0)
                    {
                        std::cerr << error.what() <<std::endl;
                        std::cerr << "Offending variable is "<<record.name+"_ta2d\n";
                        std::cerr << "Writing zeros ... \n";
                    }
                    available = false;
                }
                if( available)
                {
                    err = nc_get_vara_double( ncid, dataID,
                        start2d, count2d, transferH2d.data());
                    DVec transferD2d = transferH2d;
                    fieldaligned.integrate_between_coarse_grid( g3d, transferD2d, transferD2d);
                    transferH2d = transferD2d;
                    t2d_mp = transferH2d; //save toroidal average
                    //2. Compute fsa and output fsa
                    dg::blas2::symv( grid2gridX2d, transferH2d, transferH2dX); //interpolate onto X-point grid
                    dg::blas1::pointwiseDot( transferH2dX, volX2d, transferH2dX); //multiply by sqrt(g)
                    poloidal_average( transferH2dX, t1d, false); //average over eta
                    dg::blas1::scal( t1d, 4*M_PI*M_PI*f0); //
                    dg::blas1::copy( 0., fsa1d); //get rid of previous nan in fsa1d (nasty bug)
                    if( record_name[0] != 'j')
                        dg::blas1::pointwiseDivide( t1d, dvdpsip, fsa1d );
                    else
                        dg::blas1::copy( t1d, fsa1d);
                    //3. Interpolate fsa on 2d plane : <f>
                    dg::blas2::gemv(fsa2rzmatrix, fsa1d, transferH2d); //fsa on RZ grid
                }
                else
                {
                    dg::blas1::scal( fsa1d, 0.);
                    dg::blas1::scal( transferH2d, 0.);
                    dg::blas1::scal( t2d_mp, 0.);
                }
                err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_fsa"),
                    start1d_out, count1d, fsa1d.data());
                err = nc_put_vara_double( ncid_out, id2d.at(record_name+"_fsa2d"),
                    start2d_out, count2d, transferH2d.data() );
                if( record_name[0] == 'j')
                    dg::blas1::pointwiseDot( t2d_mp, dvdpsip2d, t2d_mp );//make it jv
                err = nc_put_vara_double( ncid_out, id2d.at(record_name+"_cta2d"),
                    start2d_out, count2d, t2d_mp.data() );
                //4. Read 2d variable and compute fluctuations
                available = true;
                try{
                    err = nc_inq_varid(ncid, (record.name+"_2d").data(), &dataID);
                } catch ( file::NC_Error& error)
                {
                    if(  i == 0)
                    {
                        std::cerr << error.what() <<std::endl;
                        std::cerr << "Offending variable is "<<record.name+"_2d\n";
                        std::cerr << "Writing zeros ... \n";
                    }
                    available = false;
                }
                if( available)
                {
                    err = nc_get_vara_double( ncid, dataID, start2d, count2d,
                        t2d_mp.data());
                    if( record_name[0] == 'j')
                        dg::blas1::pointwiseDot( t2d_mp, dvdpsip2d, t2d_mp );
                    dg::blas1::axpby( 1.0, t2d_mp, -1.0, transferH2d);
                    err = nc_put_vara_double( ncid_out, id2d.at(record_name+"_fluc2d"),
                        start2d_out, count2d, transferH2d.data() );

                    //5. flux surface integral/derivative
                    double result =0.;
                    if( record_name[0] == 'j') //j indicates a flux
                    {
                        dg::blas2::symv( dpsi, fsa1d, t1d);
                        dg::blas1::pointwiseDivide( t1d, dvdpsip, transfer1d);

                        result = dg::interpolate( dg::xspace, fsa1d, -1e-12, g1d_out);
                    }
                    else
                    {
                        dg::blas1::pointwiseDot( fsa1d, dvdpsip, t1d);
                        transfer1d = dg::integrate( t1d, g1d_out);

                        result = dg::interpolate( dg::xspace, transfer1d, -1e-12, g1d_out); //make sure to take inner cell for interpolation
                    }
                    err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_ifs"),
                        start1d_out, count1d, transfer1d.data());
                    //flux surface integral/derivative on last closed flux surface
                    err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_lcfs"),
                        start2d_out, count2d, &result );
                    //6. Compute norm of time-integral terms to get relative importance
                    if( record_name[0] == 'j') //j indicates a flux
                    {
                        dg::blas2::symv( dpsi, fsa1d, t1d);
                        dg::blas1::pointwiseDivide( t1d, dvdpsip, t1d); //dvjv
                        dg::blas1::pointwiseDot( t1d, t1d, t1d);//dvjv2
                        dg::blas1::pointwiseDot( t1d, dvdpsip, t1d);//dvjv2
                        transfer1d = dg::integrate( t1d, g1d_out);
                        result = dg::interpolate( dg::xspace, transfer1d, -1e-12, g1d_out);
                        result = sqrt(result);
                    }
                    else
                    {
                        dg::blas1::pointwiseDot( fsa1d, fsa1d, t1d);
                        dg::blas1::pointwiseDot( t1d, dvdpsip, t1d);
                        transfer1d = dg::integrate( t1d, g1d_out);

                        result = dg::interpolate( dg::xspace, transfer1d, -1e-12, g1d_out);
                        result = sqrt(result);
                    }
                    err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_norm"),
                        start2d_out, count2d, &result );
                    //7. Compute midplane fluctuation amplitudes
                    dg::blas1::pointwiseDot( transferH2d, transferH2d, transferH2d);
                    dg::blas2::symv( grid2gridX2d, transferH2d, transferH2dX); //interpolate onto X-point grid
                    dg::blas1::pointwiseDot( transferH2dX, volX2d, transferH2dX); //multiply by sqrt(g)
                    poloidal_average( transferH2dX, t1d, false); //average over eta
                    dg::blas1::scal( t1d, 4*M_PI*M_PI*f0); //
                    dg::blas1::pointwiseDivide( t1d, dvdpsip, fsa1d );
                    dg::blas1::transform ( fsa1d, fsa1d, dg::SQRT<double>() );
                    err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_std_fsa"),
                        start1d_out, count1d, fsa1d.data());
                }
                else
                {
                    dg::blas1::scal( transferH2d, 0.);
                    dg::blas1::scal( transfer1d, 0.);
                    double result = 0.;
                    err = nc_put_vara_double( ncid_out, id2d.at(record_name+"_fluc2d"),
                        start2d_out, count2d, transferH2d.data() );
                    err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_ifs"),
                        start1d_out, count1d, transfer1d.data());
                    err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_lcfs"),
                        start2d_out, count2d, &result );
                    err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_norm"),
                        start2d_out, count2d, &result );
                    err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_std_fsa"),
                        start1d_out, count1d, transfer1d.data());
                }

            }


        } //end timestepping
        err = nc_close(ncid);
    }
    err = nc_close(ncid_out);

    return 0;
}
