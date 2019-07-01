#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include "json/json.h"

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/nc_utilities.h"
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

    //std::cout << "input "<<input<<std::endl;
    //std::cout << "geome "<<geom <<std::endl;
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
    std::vector<std::string> names_input{
        "electrons", "ions", "Ue", "Ui", "potential", "induction"
    };

    //-----------------Create Netcdf output file with attributes----------//
    int ncid_out;
    err = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid_out);

    /// Set global attributes
    std::map<std::string, std::string> att;
    att["title"] = "Output file of feltor/diag/feltordiag.cu";
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
    att["inputfile"] = input;
    att["geomfile"] = geom;
    for( auto pair : att)
        err = nc_put_att_text( ncid, NC_GLOBAL,
            pair.first.data(), pair.second.size(), pair.second.data());

    //-------------------Construct grids-------------------------------------//

    const double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    const double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    const double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    const double Zmax=p.boxscaleZp*gp.a*gp.elongation;

    dg::Grid2d   g2d_out( Rmin,Rmax, Zmin,Zmax,
        p.n_out, p.Nx_out, p.Ny_out, p.bcxN, p.bcyN);

    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    const double psip0 = mag.psip()(gp.R_0, 0);
    mag = dg::geo::createModifiedSolovevField(gp, (1.-p.rho_damping)*psip0, p.alpha_mag);
    dg::HVec psipog2d = dg::evaluate( mag.psip(), g2d_out);
    // Construct weights and temporaries

    dg::HVec transferH2d = dg::evaluate(dg::zero,g2d_out);
    dg::HVec t2d_mp = dg::evaluate(dg::zero,g2d_out);


    ///--------------- Construct X-point grid ---------------------//
    //Find O-point
    double R_O = gp.R_0, Z_O = 0.;
    dg::geo::findXpoint( mag.get_psip(), R_O, Z_O);
    const double psipmin = mag.psip()(R_O, Z_O);


    unsigned npsi = 3, Npsi = 64;//set number of psivalues (NPsi % 8 == 0)
    std::cout << "Generate X-point flux-aligned grid!\n";
    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( mag.get_psip(), R_X, Z_X) ;
    dg::geo::SeparatrixOrthogonal generator(mag.get_psip(), monitor_chi, psipmin, R_X, Z_X, mag.R0(), 0, 0, true);
    double fx_0 = 1./8.;
    double psipmax = dg::blas1::reduce( psipog2d, 0. ,thrust::maximum<double>()); //DEPENDS ON GRID RESOLUTION!!
    std::cout << "psi max is            "<<psipmax<<"\n";
    psipmax = -fx_0/(1.-fx_0)*psipmin;
    std::cout << "psi max in g1d_out is "<<psipmax<<"\n";
    dg::geo::CurvilinearGridX2d gridX2d( generator, fx_0, 0., npsi, Npsi, 160, dg::DIR_NEU, dg::NEU);
    std::cout << "DONE!\n";
    //Create 1d grid
    dg::Grid1d g1d_out(psipmin, psipmax, 3, Npsi, dg::DIR_NEU); //inner value is always 0
    const double f0 = ( gridX2d.x1() - gridX2d.x0() ) / ( psipmax - psipmin );
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
    dg::blas1::axpby( -1./psipmin, rho, +1., 1., rho); //transform psi to rho
    map1d.emplace_back("rho", rho,
        "Alternative flux label rho = -psi/psimin + 1");
    dg::geo::SafetyFactor qprofile( mag);
    map1d.emplace_back("q-profile", dg::evaluate( qprofile,   g1d_out),
        "q-profile (Safety factor) using direct integration");
    map1d.emplace_back("psi_psi",    dg::evaluate( dg::cooX1d, g1d_out),
        "Flux label psi (same as coordinate)");


    // interpolate from 2d grid to X-point points
    dg::IHMatrix grid2gridX2d  = dg::create::interpolation(
        coordsX[0], coordsX[1], g2d_out);
    // interpolate fsa back to 2d or 3d grid
    dg::IHMatrix fsa2rzmatrix = dg::create::interpolation(
        psipog2d, g1d_out, dg::DIR_NEU);

    dg::HVec dvdpsip2d = dg::evaluate( dg::zero, g2d_out);
    dg::blas2::symv( fsa2rzmatrix, dvdpsip, dvdpsip2d);
    dg::HMatrix dpsi = dg::create::dx( g1d_out, dg::DIR_NEU);

    // define 2d and 1d and 0d dimensions and variables
    int dim_ids[3], tvarID;
    err = file::define_dimensions( ncid_out, dim_ids, &tvarID, g2d_out);
    int dim_ids1d[2] = {dim_ids[0], 0}; //time , psi
    err = file::define_dimension( ncid_out, "psi", &dim_ids1d[1], g1d_out);
    //Write long description
    std::string long_name = "Time at which 2d fields are written";
    err = nc_put_att_text( ncid, tvarID, "long_name", long_name.size(),
            long_name.data());
    long_name = "Flux surface label";
    err = nc_put_att_text( ncid, dim_ids1d[1], "long_name",
        long_name.size(), long_name.data());

    std::map<std::string, int> id0d, id1d, id2d;

    size_t count1d[2] = {1, g1d_out.n()*g1d_out.N()};
    size_t start1d[2] = {0, 0};
    size_t count2d[3] = {1, g2d_out.n()*g2d_out.Ny(), g2d_out.n()*g2d_out.Nx()};
    size_t start2d[3] = {0, 0, 0};

    //write 1d static vectors (psi, q-profile, ...) into file
    for( auto tp : map1d)
    {
        int vid;
        err = nc_def_var( ncid, std::get<0>(tp).data(), NC_DOUBLE, 1,
            &dim_ids1d[1], &vid);
        err = nc_put_att_text( ncid, vid, "long_name",
            std::get<2>(tp).size(), std::get<2>(tp).data());
        err = nc_enddef( ncid);
        err = nc_put_var_double( ncid, vid, std::get<1>(tp).data());
        err = nc_redef(ncid);
    }
    err = nc_close(ncid_out);

    /////////////////////////////////////////////////////////////////////////
    int timeID;
    double time=0.;

    size_t steps;
    err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
    err = nc_inq_unlimdim( ncid, &timeID); //Attention: Finds first unlimited dim, which hopefully is time and not energy_time
    err = nc_inq_dimlen( ncid, timeID, &steps);
    err = nc_close( ncid); //close 3d file

    err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
    //read in Ne,Ni,Ue,Ui,Phi,Apar
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

        name = record_name + "_ifs";
        long_name = record.long_name + " (wrt. vol integrated Flux surface average unless it is a current then it is the derived flux surface average)";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 2, dim_ids1d,
            &id1d[name]);
        err = nc_put_att_text( ncid_out, id1d[name], "long_name", long_name.size(),
            long_name.data());

        name = record_name + "_ifs_lcfs";
        long_name = record.long_name + " (wrt. vol integrated Flux surface average evaluated on last closed flux surface unless it is a current then it is the fsa evaluated)";
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, 1, dim_ids,
            &id0d[name]);
        err = nc_put_att_text( ncid_out, id0d[name], "long_name", long_name.size(),
            long_name.data());
    }
    //steps = 3;
    for( unsigned i=0; i<steps; i++)//timestepping
    {
        err = nc_get_vara_double( ncid, timeID, start2d, count2d, &time);
        std::cout << "Timestep = " << i << "  time = " << time << std::endl;
        //write time
        err = nc_put_vara_double( ncid_out, tvarID, start2d, count2d, &time);
        start2d[0] = i;
        start1d[0] = i;
        for( auto& record : feltor::diagnostics2d_list)
        {
            std::string record_name = record.name;
            if( record_name[0] == 'j')
                record_name[1] = 'v';
            //1. Read toroidal average
            err = nc_get_vara_double( ncid, id2d.at(record_name+"_ta2d"),
                start2d, count2d, transferH2d.data());
            //2. Compute fsa and output fsa
            dg::blas2::symv( grid2gridX2d, transferH2d, transferH2dX); //interpolate onto X-point grid
            dg::blas1::pointwiseDot( transferH2dX, volX2d, transferH2dX); //multiply by sqrt(g)
            poloidal_average( transferH2dX, t1d, false); //average over eta
            dg::blas1::scal( t1d, 4*M_PI*M_PI*f0); //
            dg::blas1::pointwiseDivide( t1d, dvdpsip, fsa1d );
            if( record_name[0] == 'j')
                dg::blas1::pointwiseDot( fsa1d, dvdpsip, fsa1d );
            err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_fsa"),
                start1d, count1d, fsa1d.data());
            //3. Interpolate fsa on 2d plane : <f>
            dg::blas2::gemv(fsa2rzmatrix, fsa1d, transferH2d); //fsa on RZ grid
            err = nc_put_vara_double( ncid_out, id2d.at(record_name+"_fsa2d"),
                start2d, count2d, transferH2d.data() );

            //4. Read 2d variable and compute fluctuations
            err = nc_get_vara_double( ncid, id2d.at(record.name+"_2d"), start2d, count2d,
                t2d_mp.data());
            if( record_name[0] == 'j')
                dg::blas1::pointwiseDot( t2d_mp, dvdpsip2d, t2d_mp );
            dg::blas1::axpby( 1.0, t2d_mp, -1.0, transferH2d);
            err = nc_put_vara_double( ncid_out, id2d.at(record_name+"_fluc2d"),
                start2d, count2d, transferH2d.data() );

            //5. flux surface integral/derivative
            double result =0.;
            if( record_name[0] == 'j') //j indicates a flux
            {
                dg::blas2::symv( dpsi, fsa1d, t1d);
                dg::blas1::pointwiseDivide( t1d, dvdpsip, transfer1d);

                result = dg::interpolate( fsa1d, 0., g1d_out);
            }
            else
            {
                dg::blas1::pointwiseDot( fsa1d, dvdpsip, fsa1d);
                transfer1d = dg::integrate( fsa1d, g1d_out);

                result = dg::interpolate( transfer1d, 0., g1d_out);
            }
            err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_ifs"),
                start1d, count1d, transfer1d.data());
            //flux surface integral/derivative on last closed flux surface
            err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_lcfs"),
                start2d, count2d, &result );

        }


    } //end timestepping
    err = nc_close(ncid_out);

    return 0;
}
