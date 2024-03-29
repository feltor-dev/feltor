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
#include "feltordiag.h"
#include "common.h"

struct Entry
{
    std::string description;
    int dimension_length;
    int* dimensions;
    bool exists;
};

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
    dg::file::NC_Error_Handle err;
    int ncid_in;
    err = nc_open( argv[2], NC_NOWRITE, &ncid_in); //open 3d file
    // create output early so that netcdf failures register early
    // and simplesimdb knows that file is under construction
    int ncid_out;
    err = nc_create(argv[argc-1],NC_NETCDF4|NC_NOCLOBBER, &ncid_out);
    size_t length;
    err = nc_inq_attlen( ncid_in, NC_GLOBAL, "inputfile", &length);
    std::string inputfile(length, 'x');
    err = nc_get_att_text( ncid_in, NC_GLOBAL, "inputfile", &inputfile[0]);
    err = nc_close( ncid_in);
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    dg::file::string2Json(inputfile, js.asJson(), dg::file::comments::are_forbidden);
    //we only need some parameters from p, not all
    const feltor::Parameters p(js);
    std::cout << js.asJson() <<  std::endl;
    dg::file::WrappedJsonValue config( dg::file::error::is_warning);
    try{
        dg::file::file2Json( argv[1], config.asJson(),
                dg::file::comments::are_discarded, dg::file::error::is_warning);
    } catch( std::exception& e) {
        DG_RANK0 std::cerr << "ERROR in input file "<<argv[1]<<std::endl;
        DG_RANK0 std::cerr << e.what()<<std::endl;
        return -1;
    }
    std::string configfile = config.asJson().toStyledString();
    std::cout << configfile <<  std::endl;

    //-------------------Construct grids-------------------------------------//

    dg::geo::CylindricalFunctor wall, transition;
    dg::geo::TokamakMagneticField mag, mod_mag;
    try{
        mag = dg::geo::createMagneticField(js["magnetic_field"]["params"]);
        mod_mag = dg::geo::createModifiedField(js["magnetic_field"]["params"],
                js["boundary"]["wall"], wall, transition);
    }catch(std::runtime_error& e)
    {
        std::cerr << "ERROR in input file "<<argv[2]<<std::endl;
        std::cerr <<e.what()<<std::endl;
        return -1;
    }

    const double Rmin=mag.R0()-p.boxscaleRm*mag.params().a();
    const double Zmin=-p.boxscaleZm*mag.params().a();
    const double Rmax=mag.R0()+p.boxscaleRp*mag.params().a();
    const double Zmax=p.boxscaleZp*mag.params().a();
    const unsigned FACTOR=config.get( "Kphi", 10).asUInt();

    unsigned cx = js["output"]["compression"].get(0u,1).asUInt();
    unsigned cy = js["output"]["compression"].get(1u,1).asUInt();
    unsigned n_out = p.n, Nx_out = p.Nx/cx, Ny_out = p.Ny/cy;
    dg::Grid2d g2d_out( Rmin,Rmax, Zmin,Zmax,
        n_out, Nx_out, Ny_out, p.bcxN, p.bcyN);
    /////////////////////////////////////////////////////////////////////////
    dg::CylindricalGrid3d g3d( Rmin, Rmax, Zmin, Zmax, 0., 2.*M_PI,
        n_out, Nx_out, Ny_out, p.Nz, p.bcxN, p.bcyN, dg::PER);
    dg::CylindricalGrid3d g3d_fine( Rmin, Rmax, Zmin, Zmax, 0., 2.*M_PI,
        n_out, Nx_out, Ny_out, FACTOR*p.Nz, p.bcxN, p.bcyN, dg::PER);

    //create RHS
    if( p.periodify)
        mod_mag = dg::geo::periodify( mod_mag, Rmin, Rmax, Zmin, Zmax, dg::NEU, dg::NEU);
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
    att["configfile"] = configfile;
    for( auto pair : att)
        err = nc_put_att_text( ncid_out, NC_GLOBAL,
            pair.first.data(), pair.second.size(), pair.second.data());

    int dim_id1d = 0;
    err = dg::file::define_dimension( ncid_out, &dim_id1d, g1d_out, {"psi"} );
    int dim_idX[2]= { 0, dim_id1d};
    err = dg::file::define_dimension( ncid_out, &dim_idX[0], g1d_out_eta, {"eta"} );
    //write 1d static vectors (psi, q-profile, ...) into file
    for( auto tp : map1d)
    {
        int vid;
        err = nc_def_var( ncid_out, std::get<0>(tp).data(), NC_DOUBLE, 1, &dim_id1d, &vid);
        err = nc_put_att_text( ncid_out, vid, "long_name",
            std::get<2>(tp).size(), std::get<2>(tp).data());
        err = nc_enddef(ncid_out);
        err = nc_put_var_double( ncid_out, vid, std::get<1>(tp).data());
        err = nc_redef(ncid_out);
    }
    for( auto tp : map2d)
    {
        int vid;
        err = nc_def_var( ncid_out, std::get<0>(tp).data(), NC_DOUBLE, 2, dim_idX, &vid);
        err = nc_put_att_text( ncid_out, vid, "long_name",
            std::get<2>(tp).size(), std::get<2>(tp).data());
        err = nc_enddef(ncid_out);
        err = nc_put_var_double( ncid_out, vid, std::get<1>(tp).data());
        err = nc_redef(ncid_out);
    }
    if( p.calibrate )
    {
        err = nc_close( ncid_out);
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
    // define 2d and 1d and 0d dimensions and variables
    int dim_ids[3], tvarID;
    err = dg::file::define_dimensions( ncid_out, dim_ids, &tvarID, g2d_out);

    int dim_ids2d[2] = {dim_ids[0], dim_id1d}; //time,  psi
    int dim_ids2dX[3]= {dim_ids[0], dim_idX[0], dim_idX[1]};
    //Write long description
    std::string long_name = "Time at which 2d fields are written";
    err = nc_put_att_text( ncid_out, tvarID, "long_name", long_name.size(),
            long_name.data());

    size_t count1d[2] = {1, g1d_out.n()*g1d_out.N()};
    size_t count2d[3] = {1, g2d_out.n()*g2d_out.Ny(), g2d_out.n()*g2d_out.Nx()};
    size_t count2dX[3] = {1, g1d_out_eta.n()*g1d_out_eta.N(), g1d_out.n()*g1d_out.N()};//NEW: Definition of count2dX
    size_t start2d[3] = {0, 0, 0};


    std::vector<std::vector<feltor::Record>> equation_list;
    bool equation_list_exists = js["output"].asJson().isMember("equations");
    if( equation_list_exists)
    {
        for( unsigned i=0; i<js["output"]["equations"].size(); i++)
        {
            std::string eqn = js["output"]["equations"][i].asString();
            if( eqn == "Basic")
                equation_list.push_back(feltor::basicDiagnostics2d_list);
            else if( eqn == "Mass-conserv")
                equation_list.push_back(feltor::MassConsDiagnostics2d_list);
            else if( eqn == "Energy-theorem")
                equation_list.push_back(feltor::EnergyDiagnostics2d_list);
            else if( eqn == "Toroidal-momentum")
                equation_list.push_back(feltor::ToroidalExBDiagnostics2d_list);
            else if( eqn == "Parallel-momentum")
                equation_list.push_back(feltor::ParallelMomDiagnostics2d_list);
            else if( eqn == "Zonal-Flow-Energy")
                equation_list.push_back(feltor::RSDiagnostics2d_list);
            else if( eqn == "COCE")
                equation_list.push_back(feltor::COCEDiagnostics2d_list);
            else
                throw std::runtime_error( "output: equations: "+eqn+" not recognized!\n");
        }
    }
    else // default diagnostics
    {
        equation_list.push_back(feltor::basicDiagnostics2d_list);
        equation_list.push_back(feltor::MassConsDiagnostics2d_list);
        equation_list.push_back(feltor::EnergyDiagnostics2d_list);
        equation_list.push_back(feltor::ToroidalExBDiagnostics2d_list);
        equation_list.push_back(feltor::ParallelMomDiagnostics2d_list);
        equation_list.push_back(feltor::RSDiagnostics2d_list);
    }


    std::map<std::string, Entry> diag_list = {
        {"fsa" , {" (Flux surface average.)", 2, dim_ids2d, false}  },
        {"fsa2d" , {" (Flux surface average interpolated to 2d plane.)", 3, dim_ids, false} },
        {"cta2d" , {" (Convoluted toroidal average on 2d plane.)", 3, dim_ids, false}},
        {"cta2dX" , {" (Convoluted toroidal average on magnetic plane.)", 3, dim_ids2dX, false}},
        {"fluc2d" , {" (Fluctuations wrt fsa on phi = 0 plane.)", 3, dim_ids, false}},
        {"ifs", { " (wrt. vol integrated flux surface average)", 2, dim_ids2d, false}},
        {"ifs_lcfs",
            { " (wrt. vol integrated flux surface average evaluated on last closed flux surface)", 1, dim_ids, false}},
        {"ifs_norm", {" (wrt. vol integrated square derivative of the flux surface average from 0 to lcfs)",1, dim_ids, false}},
        {"std_fsa", {" (Flux surface average standard deviation on outboard midplane.)", 2, dim_ids2d, false}}
    };
    bool diagnostics_list_exists = config.asJson().isMember("diagnostics");
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
    std::map<std::string, int> IDS;


    for(auto& m_list : equation_list) //Loop over the output lists (different equations studied).
    for( auto& record : m_list) //Loop over the different variables inside each of the lists of outputs.
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
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, entry.second.dimension_length,
            entry.second.dimensions, &IDS[name]);
        err = nc_put_att_text( ncid_out, IDS[name], "long_name", long_name.size(),
            long_name.data());
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
    size_t counter = 0;
    int ncid;
    std::cout << "Using flux-surface-average mode: "<<fsa_mode << "\n";
    for( int j=2; j<argc-1; j++)
    {
        int timeID;
        size_t steps;
        std::cout << "Opening file "<<argv[j]<<"\n";
        try{
            err = nc_open( argv[j], NC_NOWRITE, &ncid); //open 3d file
        } catch ( dg::file::NC_Error& error)
        {
            std::cerr << "An error occurded opening file "<<argv[j]<<"\n";
            std::cerr << error.what()<<std::endl;
            std::cerr << "Continue with next file\n";
            continue;
        }
        err = nc_inq_unlimdim( ncid, &timeID);
        err = nc_inq_dimlen( ncid, timeID, &steps);
        err = nc_inq_varid(ncid, "time", &timeID);
        //steps = 3;
        for( unsigned i=0; i<steps; i++)//timestepping
        {
            if( j > 2 && i == 0)
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
            for(auto& m_list : equation_list)
            {
            for( auto& record : m_list)
            {
            std::string record_name = record.name;
            if( record_name[0] == 'j')
                record_name[1] = 'v';
            //1. Read toroidal average
            int dataID =0;
            bool available = true;
            try{
                err = nc_inq_varid(ncid, (record.name+"_ta2d").data(), &dataID);
            } catch (dg::file::NC_Error& error)
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
                err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_fsa"),
                    start1d_out, count1d, fsa1d.data());
            }
            if(diag_list[ "fsa2d"].exists)
            {
                err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_fsa2d"),
                    start2d_out, count2d, transferH2d.data() );
            }
            if(diag_list["cta2d"].exists)
            {
                if( record_name[0] == 'j')
                    dg::blas1::pointwiseDot( cta2d_mp, dvdpsip2d, cta2d_mp );//make it jv
                err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_cta2d"),
                    start2d_out, count2d, cta2d_mp.data() );
            }
            if(diag_list["cta2dX"].exists)
            {
                if( record_name[0] == 'j')
                    record_name[1] = 's';
                err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_cta2dX"),
                    start2d_out, count2dX, cta2dX.data() ); //NEW: saving de X_grid data
                if( record_name[0] == 'j')
                    record_name[1] = 'v';
            }
            //4. Read 2d variable and compute fluctuations
            available = true;
            try{
                err = nc_inq_varid(ncid, (record.name+"_2d").data(), &dataID);
            } catch ( dg::file::NC_Error& error)
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
                    cta2d_mp.data());
                if( record_name[0] == 'j')
                    dg::blas1::pointwiseDot( cta2d_mp, dvdpsip2d, cta2d_mp );
                dg::blas1::axpby( 1.0, cta2d_mp, -1.0, transferH2d);
                if(diag_list["fluc2d"].exists)
                {
                    err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_fluc2d"),
                        start2d_out, count2d, transferH2d.data() );
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
                    err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_ifs"),
                        start1d_out, count1d, transfer1d.data());
                }
                //flux surface integral/derivative on last closed flux surface
                if(diag_list[ "ifs_lcfs"].exists)
                {
                    err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_ifs_lcfs"),
                        start2d_out, count2d, &result );
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
                    err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_ifs_norm"),
                        start2d_out, count2d, &result );
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
                    err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_std_fsa"),
                        start1d_out, count1d, fsa1d.data());
                }
            }
            else // make everything zero
            {
                dg::blas1::scal( transferH2d, 0.);
                dg::blas1::scal( transfer1d, 0.);
                double result = 0.;
                if(diag_list["fluc2d"].exists)
                {
                    err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_fluc2d"),
                        start2d_out, count2d, transferH2d.data() );
                }
                if(diag_list["ifs"].exists)
                {
                    err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_ifs"),
                        start1d_out, count1d, transfer1d.data());
                }
                if(diag_list["ifs_lcfs"].exists)
                {
                    err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_ifs_lcfs"),
                        start2d_out, count2d, &result );
                }
                if(diag_list["ifs_norm"].exists)
                {
                    err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_ifs_norm"),
                        start2d_out, count2d, &result );
                }
                if(diag_list["std_fsa"].exists)
                {
                    err = nc_put_vara_double( ncid_out, IDS.at(record_name+"_std_fsa"),
                        start1d_out, count1d, transfer1d.data());
                }
            }
            }
        }
        } //end timestepping
        err = nc_close(ncid);
    }
    err = nc_close(ncid_out);
    return 0;
}
