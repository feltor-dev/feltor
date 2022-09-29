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
    dg::HVec t2d_mp = dg::evaluate(dg::zero,g2d_out);


    ///--------------- Construct X-point grid ---------------------//


    //we use so many Neta so that we get close to the X-point
    unsigned npsi = config.get("n",3).asUInt();
    unsigned Npsi = config.get("Npsi", 64).asUInt();
    unsigned Neta = config.get("Neta", 640).asUInt();
    std::cout << "Using X-point grid resolution (n("<<npsi<<"), Npsi("<<Npsi<<"), Neta("<<Neta<<"))\n";
    double RO = mag.R0(), ZO = 0;
    int point = dg::geo::findOpoint( mag.get_psip(), RO, ZO);
    double psipO = mag.psip()(RO, ZO);
    std::cout << "O-point found at "<<RO<<" "<<ZO
              <<" with Psip "<<psipO<<std::endl;
    if( point == 1 )
        std::cout << " (minimum)"<<std::endl;
    if( point == 2 )
        std::cout << " (maximum)"<<std::endl;
    double fx_0 = config.get( "fx_0", 1./8.).asDouble(); //must evenly divide Npsi
    double psipmax = -fx_0/(1.-fx_0)*psipO;
    std::cout << "psi outer in g1d_out is "<<psipmax<<"\n";
    std::cout << "Generate orthogonal flux-aligned grid ... \n";
    dg::geo::SimpleOrthogonal generator(mag.get_psip(),
            psipO<psipmax ? psipO : psipmax,
            psipO<psipmax ? psipmax : psipO,
            mag.R0() + 0.1*mag.params().a(), 0., 0.1*psipO, 1);
    dg::geo::CurvilinearGrid2d gridX2d (generator,
            npsi, Npsi, Neta, dg::DIR, dg::PER);
    std::cout << "DONE!\n";
    dg::Grid1d g1d_out(psipO<psipmax ? psipO : psipmax,
                       psipO<psipmax ? psipmax : psipO,
                       npsi, Npsi, psipO < psipmax ? dg::DIR_NEU : dg::NEU_DIR);
    dg::Grid1d g1d_out_eta(gridX2d.y0(), gridX2d.y1(), npsi, Neta, dg::DIR_NEU); /// 1D grid for the eta (poloidal) directions instead of psi for the radial cut
    //O-point fsa value is always 0 (hence the DIR boundary condition)
    //f0 makes a - sign if psipmax < psipO
    const double f0 = ( gridX2d.x1() - gridX2d.x0() ) / ( psipmax - psipO );
    dg::HVec t1d = dg::evaluate( dg::zero, g1d_out), fsa1d( t1d);
    dg::HVec transfer1d = dg::evaluate(dg::zero,g1d_out);

    /// ------------------- Compute 1d flux labels ---------------------//

    std::vector<std::tuple<std::string, dg::HVec, std::string> > map1d;
    /// Compute flux volume label
    dg::Average<dg::HVec > poloidal_average( gridX2d, dg::coo2d::y);
    dg::HVec dvdpsip;
    //metric and map
    dg::SparseTensor<dg::HVec> metricX = gridX2d.metric();
    std::vector<dg::HVec > coordsX = gridX2d.map();
    dg::HVec volX2d = dg::tensor::volume2d( metricX);
    dg::HVec transferH2dX(volX2d), realtransferH2dX(volX2d); //NEW: definitions
    dg::blas1::pointwiseDot( coordsX[0], volX2d, volX2d); //R\sqrt{g}
    poloidal_average( volX2d, dvdpsip, false);
    dg::blas1::scal( dvdpsip, 4.*M_PI*M_PI*f0);
    map1d.emplace_back( "dvdpsi", dvdpsip,
        "Derivative of flux volume with respect to flux label psi");
    dg::direction integration_dir = psipO < psipmax ? dg::forward : dg::backward;
    dg::HVec X_psi_vol = dg::integrate( dvdpsip, g1d_out, integration_dir);
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
    dg::geo::SafetyFactor qprof( mod_mag);
    dg::HVec psi_vals = dg::evaluate( dg::cooX1d, g1d_out);
    // we need to avoid calling SafetyFactor outside closed fieldlines
    dg::blas1::subroutine( [psipO]( double& psi){
           if( (psipO < 0 && psi > 0) || (psipO>0 && psi <0))
               psi = psipO/2.; // just use a random value
        }, psi_vals);
    dg::HVec qprofile( psi_vals);
    dg::blas1::evaluate( qprofile, dg::equals(), qprof, psi_vals);
    map1d.emplace_back("q-profile", qprofile,
        "q-profile (Safety factor) using direct integration");
    map1d.emplace_back("psi_psi",    dg::evaluate( dg::cooX1d, g1d_out),
        "Poloidal flux label psi (same as coordinate)");
    dg::HVec psit = dg::integrate( qprofile, g1d_out, integration_dir);
    std::cout << "q-pfo "<<qprofile[10]<<"\n";
    std::cout << "Psi_t "<<psit[10]<<"\n";
    map1d.emplace_back("psit1d", psit,
        "Toroidal flux label psi_t integrated using q-profile");
    //we need to avoid integrating >=0 for total psi_t
    dg::Grid1d g1d_fine(psipO<0. ? psipO : 0., psipO<0. ? 0. : psipO, npsi
            ,Npsi,dg::DIR_NEU);
    qprofile = dg::evaluate( qprof, g1d_fine);
    dg::HVec w1d = dg::create::weights( g1d_fine);
    double psit_tot = dg::blas1::dot( w1d, qprofile);
    if( integration_dir == dg::backward)
        psit_tot *= -1;
    std::cout << "q-pfo "<<qprofile[10]<<"\n";
    std::cout << "Psi_t "<<psit[10]<<"\n";
    std::cout << "total "<<psit_tot<<"\n";
    dg::blas1::scal ( psit, 1./psit_tot);
    dg::blas1::transform( psit, psit, dg::SQRT<double>());
    map1d.emplace_back("rho_t", psit,
        "Toroidal flux label rho_t = sqrt( psit/psit_tot)");

    //-----------------Create Netcdf output file with attributes----------//
    //-----------------And 1d static output                     ----------//
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
    for( auto pair : att)
        err = nc_put_att_text( ncid_out, NC_GLOBAL,
            pair.first.data(), pair.second.size(), pair.second.data());

    int dim_id1d = 0;
    err = dg::file::define_dimension( ncid_out, &dim_id1d, g1d_out, {"psi"} );
    //write 1d static vectors (psi, q-profile, ...) into file
    if(config["DIAG_OUTPUTS"].get( "1dStatics", false).asBool())
    {
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
    if( p.calibrate )
    {
        err = nc_close( ncid_out);
        return 0;
    }
    }
    //
    //---------------------END OF CALIBRATION-----------------------------//
    //
    // interpolate from 2d grid to X-point points
    dg::IHMatrix grid2gridX2d  = dg::create::interpolation(
        coordsX[0], coordsX[1], g2d_out);
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
    int dim_ids2dX[3]= {dim_ids[0], 0, dim_id1d}; //NEW: time,  eta, psip
    err = dg::file::define_dimension( ncid_out, &dim_ids2dX[1], g1d_out_eta, {"eta"} ); //NEW: Name of the new 1d DIRECTION
    //Write long description
    std::string long_name = "Time at which 2d fields are written";
    err = nc_put_att_text( ncid_out, tvarID, "long_name", long_name.size(),
            long_name.data());
    std::map<std::string, int> id0d, id1d, id2d, id2dX; // NEW: Added poloidal and 2dX grids

    //int dimensions[5]= {dim_ids[0], dim_ids[1], dim_ids[2], dim_id1d, dim_ids2dX[1]}; //array of dimensions time, R, Z, psi, eta
    //std::vector<std::map<std::string, int>> IDS;


    size_t count1d[2] = {1, g1d_out.n()*g1d_out.N()};
    size_t count2d[3] = {1, g2d_out.n()*g2d_out.Ny(), g2d_out.n()*g2d_out.Nx()};
    size_t count2dX[3] = {1, g1d_out_eta.n()*g1d_out_eta.N(), g1d_out.n()*g1d_out.N()};//NEW: Definition of count2dX
    size_t start2d[3] = {0, 0, 0};


    std::string name;
    std::vector<std::vector<feltor::Record>> LISTS;
    if(js["output"]["equations"].get( "Basic", false).asBool())
        LISTS.push_back(feltor::basicDiagnostics2d_list);
    if(js["output"]["equations"].get( "Mass-conserv", false).asBool())
        LISTS.push_back(feltor::MassConsDiagnostics2d_list);
    if(js["output"]["equations"].get( "Energy-theorem", false).asBool())
        LISTS.push_back(feltor::EnergyDiagnostics2d_list);
    if(js["output"]["equations"].get( "Toroidal-momentum", false).asBool())
        LISTS.push_back(feltor::ToroidalExBDiagnostics2d_list);
    if(js["output"]["equations"].get( "Parallel-momentum", false).asBool())
        LISTS.push_back(feltor::ParallelMomDiagnostics2d_list);
    if(js["output"]["equations"].get( "Zonal-Flow-Energy", false).asBool())
        LISTS.push_back(feltor::RSDiagnostics2d_list);
    if(js["output"]["equations"].get( "COCE", false).asBool())
        LISTS.push_back(feltor::COCEDiagnostics2d_list);


    std::vector<std::string> endings;
    std::vector<std::string> description;
    std::vector<int> dimension_length;
    std::vector<int *> dimensions;
    std::vector<std::map<std::string, int>> IDS;

    if(config["DIAG_OUTPUTS"].get( "fsa1d", false).asBool())
    {endings.push_back("_fsa");
    description.push_back(" (Flux surface average.)");
    dimension_length.push_back(2);
    dimensions.push_back(dim_ids2d);
    IDS.push_back(id1d);
    }
    if(config["DIAG_OUTPUTS"].get( "fsa2d", false).asBool())
    {endings.push_back("_fsa2d");
    description.push_back(" (Flux surface average interpolated to 2d plane.)");
    dimension_length.push_back(3);
    dimensions.push_back(dim_ids);
    IDS.push_back(id2d);
    }
    if(config["DIAG_OUTPUTS"].get( "cta2d", false).asBool())
    {endings.push_back("_cta2d");
    description.push_back(" (Convoluted toroidal average on 2d plane.)");
    dimension_length.push_back(3);
    dimensions.push_back(dim_ids);
    IDS.push_back(id2d);
    }
    if(config["DIAG_OUTPUTS"].get( "cta2dX", false).asBool())
    {endings.push_back("_cta2dX");
    description.push_back("  (Convoluted toroidal average on magnetic plane.)");
    dimension_length.push_back(3);
    dimensions.push_back(dim_ids2dX);
    IDS.push_back(id2dX);
    }
    if(config["DIAG_OUTPUTS"].get( "fluc2d", false).asBool())
    {endings.push_back("_fluc2d");
    description.push_back(" (Fluctuations wrt fsa on phi = 0 plane.)");
    dimension_length.push_back(3);
    dimensions.push_back(dim_ids);
    IDS.push_back(id2d);
    }
    if(config["DIAG_OUTPUTS"].get( "ifs", false).asBool())
    {endings.push_back("_ifs");
    description.push_back(" (wrt. vol integrated flux surface average)");
    dimension_length.push_back(2);
    dimensions.push_back(dim_ids2d);
    IDS.push_back(id1d);
    }
    if(config["DIAG_OUTPUTS"].get( "ifs_lcfs", false).asBool())
    {endings.push_back("_ifs_lcfs");
    description.push_back(" (wrt. vol integrated flux surface average evaluated on last closed flux surface)");
    dimension_length.push_back(1);
    dimensions.push_back(dim_ids);
    IDS.push_back(id0d);
    }
    if(config["DIAG_OUTPUTS"].get( "ifs_norm", false).asBool())
    {endings.push_back("_ifs_norm");
    description.push_back(" (wrt. vol integrated square derivative of the flux surface average from 0 to lcfs)");
    dimension_length.push_back(1);
    dimensions.push_back(dim_ids);
    IDS.push_back(id0d);
    }
    if(config["DIAG_OUTPUTS"].get( "std_fsa", false).asBool())
    {endings.push_back("_std_fsa");
    description.push_back(" (Flux surface average standard deviation on outboard midplane.)");
    dimension_length.push_back(2);
    dimensions.push_back(dim_ids2d);
    IDS.push_back(id1d);
    }



    int diag_counter=0;
    for(auto& m_list : LISTS) //Loop over the output lists (different equations studied).
    {
    for( auto& record : m_list) //Loop over the different variables inside each of the lists of outputs.
    {diag_counter=0;
    for (auto& end : endings) //Loop over the different diagnostics to be saved.
    { std::string record_name = record.name;
        if( record_name[0] == 'j')
            record_name[1] = 'v';
        name = record_name + end;
        long_name = record.long_name + description[diag_counter];
        if((end=="_ifs") && (record_name[0] == 'j'))
            long_name = record.long_name + " (wrt. vol derivative of the flux surface average)";
        if((end=="_ifs_lcfs") && (record_name[0] == 'j'))
            long_name = record.long_name + " (flux surface average evaluated on the last closed flux surface)";
        if((end=="_ifs_norm") && (record_name[0] == 'j'))
            long_name = record.long_name + " (wrt. vol integrated square derivative of the flux surface average from 0 to lcfs)";
        if((end=="_cta2dX") && (record_name[0] == 'j'))
           {record_name[1] = 's';
            name = record_name + end;
           }
        err = nc_def_var( ncid_out, name.data(), NC_DOUBLE, dimension_length[diag_counter], dimensions[diag_counter],
            &IDS[diag_counter][name]);
        err = nc_put_att_text( ncid_out, IDS[diag_counter][name], "long_name", long_name.size(),
            long_name.data());
        diag_counter++;
        }
        }
    }

    std::cout << "Construct Fieldaligned derivative ... \n";

    auto bhat = dg::geo::createBHat( mod_mag);
    dg::geo::Fieldaligned<dg::CylindricalGrid3d, dg::IDMatrix, dg::DVec> fieldaligned(
        bhat, g3d_fine, dg::NEU, dg::NEU, dg::geo::NoLimiter(), //let's take NEU bc because N is not homogeneous
        p.rk4eps, 5, 5, -1, "dg");
    /////////////////////////////////////////////////////////////////////////
    size_t counter = 0;
    int ncid;
    std::string fsa_mode = config.get( "fsa", "convoluted-toroidal-average").asString();
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
                for(auto& m_list : LISTS)
                {
                for( auto& record : m_list)
                {
                std::string record_name = record.name;
                diag_counter=0;
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
                    dg::DVec transferD2d = transferH2d;
                    fieldaligned.integrate_between_coarse_grid( g3d, transferD2d, transferD2d);
                    t2d_mp = transferD2d; //save toroidal average
                    //2. Compute fsa and output fsa
                    if( fsa_mode == "convoluted-toroidal-average")
                        dg::blas2::symv( grid2gridX2d, t2d_mp, transferH2dX); //interpolate convoluted average onto X-point grid
                    else
                        dg::blas2::symv( grid2gridX2d, transferH2d, transferH2dX); //interpolate simple average onto X-point grid
                    realtransferH2dX=transferH2dX; //NEW: Define the 2dX grid data
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
                    dg::blas1::scal( t2d_mp, 0.);
                }
                if(config["DIAG_OUTPUTS"].get( "fsa1d", false).asBool())
                {
                err = nc_put_vara_double( ncid_out, IDS[diag_counter].at(record_name+"_fsa"),
                    start1d_out, count1d, fsa1d.data());
                    diag_counter++;
                }
                if(config["DIAG_OUTPUTS"].get( "fsa2d", false).asBool())
                {
                err = nc_put_vara_double( ncid_out, IDS[diag_counter].at(record_name+"_fsa2d"),
                    start2d_out, count2d, transferH2d.data() );
                    diag_counter++;
                }
                if( record_name[0] == 'j')
                    dg::blas1::pointwiseDot( t2d_mp, dvdpsip2d, t2d_mp );//make it jv
                if(config["DIAG_OUTPUTS"].get( "cta2d", false).asBool())
                {
                err = nc_put_vara_double( ncid_out, IDS[diag_counter].at(record_name+"_cta2d"),
                    start2d_out, count2d, t2d_mp.data() );
                    diag_counter++;
                }
                if(config["DIAG_OUTPUTS"].get( "cta2dX", false).asBool())
                    {
                        if( record_name[0] == 'j')
                            {record_name[1] = 's';
                                err = nc_put_vara_double( ncid_out, IDS[diag_counter].at(record_name+"_cta2dX"),
                                                         start2d_out, count2dX, realtransferH2dX.data() ); //NEW: saving de X_grid data
                            record_name[1] = 'v';
                            }
                        else
                            {
                            err = nc_put_vara_double( ncid_out, IDS[diag_counter].at(record_name+"_cta2dX"),
                                                      start2d_out, count2dX, realtransferH2dX.data() ); //NEW: saving de X_grid data
                            }
                    diag_counter++;
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
                { err = nc_get_vara_double( ncid, dataID, start2d, count2d,
                        t2d_mp.data());
                    if( record_name[0] == 'j')
                        dg::blas1::pointwiseDot( t2d_mp, dvdpsip2d, t2d_mp );
                    dg::blas1::axpby( 1.0, t2d_mp, -1.0, transferH2d);
                    if(config["DIAG_OUTPUTS"].get("fluc2d", false).asBool())
                    {
                    err = nc_put_vara_double( ncid_out, IDS[diag_counter].at(record_name+"_fluc2d"),
                        start2d_out, count2d, transferH2d.data() );
                        diag_counter++;
                    }
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
                        transfer1d = dg::integrate( t1d, g1d_out, integration_dir);
                        result = dg::interpolate( dg::xspace, transfer1d, -1e-12, g1d_out); //make sure to take inner cell for interpolation
                    }
                    if(config["DIAG_OUTPUTS"].get( "ifs", false).asBool())
                    {
                    err = nc_put_vara_double( ncid_out, IDS[diag_counter].at(record_name+"_ifs"),
                        start1d_out, count1d, transfer1d.data());
                        diag_counter++;
                    }
                    //flux surface integral/derivative on last closed flux surface
                    if(config["DIAG_OUTPUTS"].get( "ifs_lcfs", false).asBool())
                    {
                    err = nc_put_vara_double( ncid_out, IDS[diag_counter].at(record_name+"_ifs_lcfs"),
                        start2d_out, count2d, &result );
                        diag_counter++;
                    }
                    //6. Compute norm of time-integral terms to get relative importance
                    if( record_name[0] == 'j') //j indicates a flux
                    {
                        dg::blas2::symv( dpsi, fsa1d, t1d);
                        dg::blas1::pointwiseDivide( t1d, dvdpsip, t1d); //dvjv
                        dg::blas1::pointwiseDot( t1d, t1d, t1d);//dvjv2
                        dg::blas1::pointwiseDot( t1d, dvdpsip, t1d);//dvjv2
                        transfer1d = dg::integrate( t1d, g1d_out, integration_dir);
                        result = dg::interpolate( dg::xspace, transfer1d, -1e-12, g1d_out);
                        result = sqrt(result);
                    }
                    else
                    {
                        dg::blas1::pointwiseDot( fsa1d, fsa1d, t1d);
                        dg::blas1::pointwiseDot( t1d, dvdpsip, t1d);
                        transfer1d = dg::integrate( t1d, g1d_out, integration_dir);
                        result = dg::interpolate( dg::xspace, transfer1d, -1e-12, g1d_out);
                        result = sqrt(result);
                    }
                    if(config["DIAG_OUTPUTS"].get( "ifs_norm", false).asBool())
                    {
                    err = nc_put_vara_double( ncid_out, IDS[diag_counter].at(record_name+"_ifs_norm"),
                        start2d_out, count2d, &result );
                        diag_counter++;
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
                    if(config["DIAG_OUTPUTS"].get( "std_fsa", false).asBool())
                    {
                    err = nc_put_vara_double( ncid_out, IDS[diag_counter].at(record_name+"_std_fsa"),
                        start1d_out, count1d, fsa1d.data());
                        diag_counter++;
                    }
                }
                else
                {
                    dg::blas1::scal( transferH2d, 0.);
                    dg::blas1::scal( transfer1d, 0.);
                    double result = 0.;
                    if(config["DIAG_OUTPUTS"].get( "fluc2d", false).asBool())
                    {
                    err = nc_put_vara_double( ncid_out, id2d.at(record_name+"_fluc2d"),
                        start2d_out, count2d, transferH2d.data() );
                    }
                    if(config["DIAG_OUTPUTS"].get( "ifs", false).asBool())
                    {
                    err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_ifs"),
                        start1d_out, count1d, transfer1d.data());
                    }
                    if(config["DIAG_OUTPUTS"].get( "ifs_lcfs", false).asBool())
                    {
                    err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_lcfs"),
                        start2d_out, count2d, &result );
                    }
                    if(config["DIAG_OUTPUTS"].get( "ifs_norm", false).asBool())
                    {
                    err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_norm"),
                        start2d_out, count2d, &result );
                    }
                    if(config["DIAG_OUTPUTS"].get( "std_fsa", false).asBool())
                    {
                    err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_std_fsa"),
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
