#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/file.h"
#include "feltordiag.h"
#include "common.h"

struct Entry
{
    std::string description;
    bool exists;
};

// The only dependence on feltor is through Parameters and equation_list

int main( int argc, char* argv[])
{
    if( argc < 4)
    {
        std::cerr << "Usage: "<<argv[0]<<" [config.json] [input0.nc ... inputN.nc] [output.nc]\n";
        return -1;
    }
    for( int i=1; i<argc-1; i++)
        std::cout << argv[i]<< " ";
    std::cout << " -> "<<argv[argc-1]<<std::endl;

    //------------------------open input nc file--------------------------------//
    dg::file::NcFile file( argv[2], dg::file::nc_nowrite);

    std::string intputfile = file.get_att_as<std::string>( ".", "inputfile");
    file.close();
    // create output early so that netcdf failures register early
    // and simplesimdb knows that file is under construction
    file.open( argv[arc-1], dg::file::nc_noclobber);
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json(inputfile, dg::file::comments::are_forbidden);
    //we only need some parameters from p, not all
    const feltor::Parameters p(js);
    std::cout << js.toStyledString() <<  std::endl;
    dg::file::WrappedJsonValue config( dg::file::error::is_warning);
    try{
        config.asJson() = dg::file::file2Json( argv[1],
                dg::file::comments::are_discarded, dg::file::error::is_warning);
    } catch( std::exception& e) {
        DG_RANK0 std::cerr << "ERROR in input file "<<argv[1]<<std::endl;
        DG_RANK0 std::cerr << e.what()<<std::endl;
        return -1;
    }
    std::string configfile = config.toStyledString();
    std::cout << configfile <<  std::endl;

    //-------------------Construct grids-------------------------------------//

    auto box = common::box( js);
    const unsigned FACTOR=config.get( "Kphi", 10).asUInt();

    unsigned cx = js["output"]["compression"].get(0u,1).asUInt();
    unsigned cy = js["output"]["compression"].get(1u,1).asUInt();
    unsigned n_out = p.n, Nx_out = p.Nx/cx, Ny_out = p.Ny/cy;
    dg::Grid2d g2d_out( box.at("Rmin"),box.at("Rmax"), box.at("Zmin"),box.at("Zmax"),
        n_out, Nx_out, Ny_out, p.bcxN, p.bcyN);
    /////////////////////////////////////////////////////////////////////////
    dg::CylindricalGrid3d g3d( box.at("Rmin"), box.at("Rmax"), box.at("Zmin"), box.at("Zmax"), 0., 2.*M_PI,
        n_out, Nx_out, Ny_out, p.Nz, p.bcxN, p.bcyN, dg::PER);
    dg::CylindricalGrid3d g3d_fine( box.at("Rmin"), box.at("Rmax"), box.at("Zmin"), box.at("Zmax"), 0., 2.*M_PI,
        n_out, Nx_out, Ny_out, FACTOR*p.Nz, p.bcxN, p.bcyN, dg::PER);

    dg::geo::TokamakMagneticField mag, mod_mag, unmod_mag;
    dg::geo::CylindricalFunctor wall, transition;
    common::create_mag_wall( argv[2], js, mag, mod_mag, unmod_mag, wall, transition);

    dg::HVec psipog2d = dg::evaluate( mod_mag.psip(), g2d_out);
    // Construct weights and temporaries

    dg::HVec transferH2d = dg::evaluate(dg::zero,g2d_out);
    dg::HVec cta2d_mp = dg::evaluate(dg::zero,g2d_out);


    ///--------------- Construct X-point grid ---------------------//
    double psipO = 0, psipmax = 0, f0 = 0.;
    auto gridX2d = feltor::generate_XGrid( config, mag, psipO, psipmax, f0);
    double fx_0 = config.get( "fx_0", 1./8.).asDouble(); //must evenly divide Npsi
    unsigned inner_Nx = (unsigned)round((1-fx_0)*(double)gridX2d.Nx());
    /// ------------------- Compute flux labels ---------------------//
    dg::Average<dg::HVec > poloidal_average( gridX2d, dg::coo2d::y);
    dg::Grid1d g1d_out, g1d_out_eta;
    dg::HVec dvdpsip, volX2d;
    auto map1d = feltor::compute_oneflux_labels( poloidal_average,
            gridX2d, mod_mag, psipO, psipmax, f0,
            dvdpsip, volX2d, g1d_out, g1d_out_eta);
    auto map2d = feltor::compute_twoflux_labels( gridX2d);

    dg::direction integration_dir = psipO < psipmax ? dg::forward : dg::backward;
    dg::HVec t1d = dg::evaluate( dg::zero, g1d_out), fsa1d( t1d);
    dg::HVec transfer1d = dg::evaluate(dg::zero,g1d_out);
    dg::HVec transferH2dX(volX2d), cta2dX(volX2d); //NEW: definitions
    const dg::HVec w1d = dg::create::weights( g1d_out);

    //-----------------Create Netcdf output file with attributes----------//
    //-----------------And 1d static output                     ----------//

    /// Set global attributes
    std::map<std::string, dg::file::nc_att_t> att;
    att["title"] = "Output file of feltor/src/feltor/feltordiag.cpp";
    att["Conventions"] = "CF-1.7";
    att["history"] = dg::file::timestamp(argc, argv);
    att["comment"] = "Find more info in feltor/src/feltor.tex";
    att["source"] = "FELTOR";
    att["references"] = "https://github.com/feltor-dev/feltor";
    att["inputfile"] = jsin["inputfile"].asString();
    att["configfile"] = configfile;
    file.put_atts( ".", att);

    file.defput_dim( "psi", {{"axis", "X"}}, g1d_out.abscissas());
    file.defput_dim( "eta", {{"axis", "Y"}}, g1d_out_eta.abscissas());
    //write 1d static vectors (psi, q-profile, ...) into file
    for( auto tp : map1d)
    {
        file.defput_var( std::get<0>(tp), {"psi"}, {{"long_name", std::get<2>(tp)}},
                std::get<1>(tp));
    }
    for( auto tp : map2d)
    {
        file.defput_var( std::get<0>(tp), {"eta", "psi"}, {{"long_name",
                std::get<2>(tp)}}, std::get<1>(tp));
    }
    if( p.calibrate )
    {
        file.close();
        return 0;
    }
    //
    //---------------------END OF CALIBRATION-----------------------------//
    //
    // interpolate from 2d grid to X-point points
    std::vector<dg::HVec > coordsX = gridX2d.map();
    dg::IHMatrix grid2gridX2d  = dg::create::interpolation(
        coordsX[0], coordsX[1], g2d_out, dg::NEU, dg::NEU,
        config.get("x-grid-interpolation","dg").asString());
    // interpolate fsa back to 2d or 3d grid
    dg::IHMatrix fsa2rzmatrix = dg::create::interpolation(
        psipog2d, g1d_out, dg::DIR_NEU);

    dg::HVec dvdpsip2d = dg::evaluate( dg::zero, g2d_out);
    dg::blas2::symv( fsa2rzmatrix, dvdpsip, dvdpsip2d);
    dg::HMatrix dpsi = dg::create::dx( g1d_out, dg::backward); //we need to avoid involving cells outside LCFS in computation (also avoids right boundary)
    if( psipO > psipmax)
        dpsi = dg::create::dx( g1d_out, dg::forward);
    //although the first point outside LCFS is still wrong
    file.defput_dim_as<double>( "time", NC_UNLIMITED, {{"axis", "T"}, {"long_name", "Time at which 2d fields are written"}});

    dg::file::Writer<dg::x::Grid0d> write0d( ncid_out, {}, {"time"});
    dg::file::Writer<dg::x::Grid2d> writer_g2d( ncid_out, g2d_out,
        {"time", "y", "x"});
    writer_psi = {ncid_out, g1d_out, {"time", "psi"}};
    writer_X = {ncid_out, {g1d_out,g1d_out_eta}, {"time", "eta", "psi"}};


    auto equation_list = feltor::generate_equation_list( js);

    std::map<std::string, Entry> diag_list = {
        {"fsa" , {" (Flux surface average.)", false} },
        {"fsa2d" , {" (Flux surface average interpolated to 2d plane.)", false} },
        {"cta2d" , {" (Convoluted toroidal average on 2d plane.)", false}},
        {"cta2dX" , {" (Convoluted toroidal average on magnetic plane.)", false}},
        {"fluc2d" , {" (Fluctuations wrt fsa on phi = 0 plane.)", false}},
        {"ifs", { " (wrt. vol integrated flux surface average)", false}},
        {"ifs_lcfs",
            { " (wrt. vol integrated flux surface average evaluated on last closed flux surface)", false}},
        {"ifs_norm", {" (wrt. vol integrated square derivative of the flux surface average from 0 to lcfs)", false}},
        {"std_fsa", {" (Flux surface average standard deviation on outboard midplane.)", false }}
    };
    bool diagnostics_list_exists = config.isMember("diagnostics");
    if( diagnostics_list_exists)
    {
        for( unsigned i=0; i<config["diagnostics"].size(); i++)
        {
            std::string diag = config["diagnostics"][i].asString();
            try{
                diag_list.at(diag).exists = true;
            }
            catch ( std::out_of_range& error)
            {
                DG_RANK0 std::cerr << "ERROR: in config: "<<error.what();
                DG_RANK0 std::cerr <<"Is there a spelling error? I assume you do not want to continue with the wrong entry so I exit! Bye Bye :)"<<std::endl;
                dg::abort_program();
            }
        }
    }
    else // compute all diagnostics
        for( auto& entry : diag_list)
            entry.second.exists = true;


    for(auto& record : equation_list)
    for( auto entry : diag_list)
    {
        if( !entry.second.exists)
            continue;
        std::string diag = entry.first;
        std::string record_name = record.name;
        if( record_name[0] == 'j')
            if( diag != "cta2dX")
                record_name[1] = 'v';
        std::string name = record_name + "_" +  diag;
        std::string long_name = record.long_name + entry.second.description;
        if((diag=="ifs") && (record_name[0] == 'j'))
            long_name = record.long_name + " (wrt. vol derivative of the flux surface average)";
        if((diag=="ifs_lcfs") && (record_name[0] == 'j'))
            long_name = record.long_name + " (flux surface average evaluated on the last closed flux surface)";
        if((diag=="ifs_norm") && (record_name[0] == 'j'))
            long_name = record.long_name + " (wrt. vol integrated square derivative of the flux surface average from 0 to lcfs)";
        if( diag == "ifs_norm" || diag == "ifs_lcfs")
            file.def_var_as<double>( name, {"time"}, {{"long_name", long_name}});
        else if( diag == "fsa" || diag == "ifs" || diag == "std_fsa")
            writer_psi.def( name, dg::file::long_name(long_name));
        else if( diag == "cta2dX")
            writer_X.def( name, dg::file::long_name(long_name));
        else // fsa2d cta2d fluc2d
            writer_g2d.def( name, dg::file::long_name(long_name));

    }

    std::cout << "Construct Fieldaligned derivative ... \n";
    std::string fsa_mode = config.get( "fsa", "convoluted-toroidal-average").asString();

    auto bhat = dg::geo::createBHat( mod_mag);
    dg::geo::Fieldaligned<dg::CylindricalGrid3d, dg::IDMatrix, dg::DVec> fieldaligned;
    if( fsa_mode == "convoluted-toroidal-average" || diag_list["cta2d"].exists
            || diag_list["cta2dX"].exists)
    {
        fieldaligned.construct(
            bhat, g3d_fine, dg::NEU, dg::NEU, dg::geo::NoLimiter(),
            p.rk4eps, 5, 5, -1,
            config.get("cta-interpolation","dg").asString());
    }
    /////////////////////////////////////////////////////////////////////////
    std::cout << "Using flux-surface-average mode: "<<fsa_mode << "\n";
    size_t stack = 0;
    for( int j=2; j<argc-1; j++)
    {
        std::cout << "Opening file "<<argv[j]<<"\n";
        dg::file::NcFile file_in;
        try{
            file_in.open( argv[j], dg::file::nc_nowrite);
        } catch ( dg::file::NC_Error& error)
        {
            std::cerr << "An error occurded opening file "<<argv[j]<<"\n";
            std::cerr << error.what()<<std::endl;
            std::cerr << "Continue with next file\n";
            continue;
        }
        dg::file::Reader<dg::x::Grid0d> read0d( ncid, {},{"time"});
        dg::file::Reader<dg::x::Grid2d> read2d( ncid, g2d_out,{"time","y","x"});
        size_t steps = file_in.get_dim_size("time");
        auto names = file_in.names();
        //steps = 2; // for testing
        for( unsigned i=0; i<steps; i++)//timestepping
        {
            if( j > 2 && i == 0)
                continue; // else we duplicate the first timestep
            // read and write time
            double time=0.;
            read0d.get( "time", time, i);
            std::cout << " Timestep = " << i <<"/"<<steps-1 << "  time = " << time << std::endl;
            write0d.stack( "time", time);
            for(auto& record : equation_list)
            {
            std::string record_name = record.name;
            if( record_name[0] == 'j')
                record_name[1] = 'v';
            //1. Read toroidal average
            bool available = true;
            if( std::find( names.begin(), names.end(), record.name+"_ta2d") == names.end())
            {
                if(  i == 0)
                {
                    std::cerr << "Variable "<<record.name<<"_ta2d not found!" <<std::endl;
                    std::cerr << "Writing zeros ... \n";
                }
                available = false;
            }
            if( available)
            {
                read2d.get( record.name+"_ta2d", transferH2d, i);
                if( fsa_mode == "convoluted-toroidal-average" || diag_list["cta2d"].exists
                        || diag_list["cta2dX"].exists)
                {
                    dg::DVec transferD2d = transferH2d;
                    fieldaligned.integrate_between_coarse_grid( g3d, transferD2d, transferD2d);
                    cta2d_mp = transferD2d; //save toroidal average
                    if( fsa_mode == "convoluted-toroidal-average" || diag_list["cta2dX"].exists)
                        dg::blas2::symv( grid2gridX2d, cta2d_mp, cta2dX); //interpolate convoluted average onto X-point grid
                }
                //2. Compute fsa and output fsa
                if( fsa_mode == "convoluted-toroidal-average")
                    dg::blas1::copy( cta2dX, transferH2dX);
                else
                    dg::blas2::symv( grid2gridX2d, transferH2d, transferH2dX); //interpolate simple average onto X-point grid
                dg::blas1::pointwiseDot( transferH2dX, volX2d, transferH2dX); //multiply by sqrt(g)
                try{
                    poloidal_average( transferH2dX, t1d, false); //average over eta
                } catch( dg::Error& e)
                {
                    std::cerr << "WARNING: "<<record_name<<" contains NaN or Inf\n";
                    dg::blas1::scal( t1d, NAN);
                }
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
                dg::blas1::scal( cta2d_mp, 0.);
                dg::blas1::scal( cta2dX, 0.);
            }
            if(diag_list["fsa"].exists)
            {
                writer_psi.stack( record_name+"_fsa", fsa1d);
            }
            if(diag_list[ "fsa2d"].exists)
            {
                writer_g2d.stack( record_name+"_fsa2d", transferH2d);
            }
            if(diag_list["cta2d"].exists)
            {
                if( record_name[0] == 'j')
                    dg::blas1::pointwiseDot( cta2d_mp, dvdpsip2d, cta2d_mp );//make it jv
                writer_g2d.stack( record_name+"_cta2d", cta2d_mp);
            }
            if(diag_list["cta2dX"].exists)
            {
                if( record_name[0] == 'j')
                    record_name[1] = 's';
                writer_X.stack( record_name+"_cta2dX", cta2dX);
                if( record_name[0] == 'j')
                    record_name[1] = 'v';
            }
            //4. Read 2d variable and compute fluctuations
            available = true;
            if( std::find( names.begin(), names.end(), record.name+"_2d") == names.end())
            {
                if(  i == 0)
                {
                    std::cerr << "Variable "<<record.name<<"_2d not found!" <<std::endl;
                    std::cerr << "Writing zeros ... \n";
                }
                available = false;
            }
            if( available)
            {
                read2d.get( record.name+"_2d", cta2d_mp, i);
                if( record_name[0] == 'j')
                    dg::blas1::pointwiseDot( cta2d_mp, dvdpsip2d, cta2d_mp );
                dg::blas1::axpby( 1.0, cta2d_mp, -1.0, transferH2d);
                if(diag_list["fluc2d"].exists)
                {
                    writer_g2d.stack( record_name+"_fluc2d",
                        transferH2d);
                }
                //5. flux surface integral/derivative
                double result =0.;
                if( record_name[0] == 'j') //j indicates a flux
                {
                    dg::blas2::symv( dpsi, fsa1d, t1d);
                    dg::blas1::pointwiseDivide( t1d, dvdpsip, transfer1d);
                    result = dg::interpolate( dg::xspace, fsa1d, psipO < psipmax ? -1e-12 : 1e-12, g1d_out);
                }
                else
                {
                    dg::blas1::pointwiseDot( fsa1d, dvdpsip, t1d);
                    transfer1d = dg::integrate( t1d, g1d_out, integration_dir);
                    // dG computation of integral up to lcfs
                    dg::HVec temp1( inner_Nx*gridX2d.n()), temp2(temp1);
                    for( unsigned u=0; u<temp1.size(); u++)
                    {
                        temp1[u] = w1d[u];
                        temp2[u] = integration_dir == dg::forward ? t1d[u] : -t1d[t1d.size() -1 -u];
                    }
                    result = dg::blas1::dot( temp1, temp2);
                }
                if(diag_list[ "ifs"].exists)
                {
                    writer_psi.stack( record_name+"_ifs", transfer1d);
                }
                //flux surface integral/derivative on last closed flux surface
                if(diag_list[ "ifs_lcfs"].exists)
                {
                    write0d.stack( record_name+"_ifs_lcfs", result);
                }
                //6. Compute norm of time-integral terms to get relative importance
                if( record_name[0] == 'j') //j indicates a flux
                {
                    dg::blas2::symv( dpsi, fsa1d, t1d);
                    dg::blas1::pointwiseDivide( t1d, dvdpsip, t1d); //dvjv
                    dg::blas1::pointwiseDot( t1d, t1d, t1d);//dvjv2
                    dg::blas1::pointwiseDot( t1d, dvdpsip, t1d);//dvjv2
                }
                else
                {
                    dg::blas1::pointwiseDot( fsa1d, fsa1d, t1d);
                    dg::blas1::pointwiseDot( t1d, dvdpsip, t1d);
                }
                // dG computation of integral up to lcfs
                dg::HVec temp1( inner_Nx*gridX2d.n()), temp2(temp1);
                for( unsigned u=0; u<temp1.size(); u++)
                {
                    temp1[u] = w1d[u];
                    temp2[u] = integration_dir == dg::forward ? t1d[u] : t1d[t1d.size() -1 -u];
                }
                result = sqrt(dg::blas1::dot( temp1, temp2));
                if(diag_list["ifs_norm"].exists)
                {
                    write0d.stack( record_name+"_ifs_norm", result);
                }
                //7. Compute midplane fluctuation amplitudes
                dg::blas1::pointwiseDot( transferH2d, transferH2d, transferH2d);
                dg::blas2::symv( grid2gridX2d, transferH2d, transferH2dX); //interpolate onto X-point grid
                dg::blas1::pointwiseDot( transferH2dX, volX2d, transferH2dX); //multiply by sqrt(g)
                try{
                    poloidal_average( transferH2dX, t1d, false); //average over eta
                } catch( dg::Error& e)
                {
                    std::cerr << "WARNING: "<<record_name<<" contains NaN or Inf\n";
                    dg::blas1::scal( t1d, NAN);
                }
                dg::blas1::scal( t1d, 4*M_PI*M_PI*f0); //
                dg::blas1::pointwiseDivide( t1d, dvdpsip, fsa1d );
                dg::blas1::transform ( fsa1d, fsa1d, dg::SQRT<double>() );
                if(diag_list["std_fsa"].exists)
                {
                    writer_psi.stack( record_name+"_std_fsa", fsa1d);
                }
            }
            else // make everything zero
            {
                dg::blas1::scal( transferH2d, 0.);
                dg::blas1::scal( transfer1d, 0.);
                double result = 0.;
                if(diag_list["fluc2d"].exists)
                {
                    writer_g2d.stack( record_name+"_fluc2d",
                        transferH2d);
                }
                if(diag_list["ifs"].exists)
                {
                    writer_psi.stack( record_name+"_ifs",
                        transfer1d);
                }
                if(diag_list["ifs_lcfs"].exists)
                {
                    write0d.stack( record_name+"_ifs_lcfs",
                        result);
                }
                if(diag_list["ifs_norm"].exists)
                {
                    write0d.stack( record_name+"_ifs_norm",
                        result);
                }
                if(diag_list["std_fsa"].exists)
                {
                    writer_psi.stack( record_name+"_std_fsa",
                        transfer1d);
                }
            }
            } // equation_list
            stack++;
        } //end timestepping
        file_in.close();
    }
    file.close();
    return 0;
}
