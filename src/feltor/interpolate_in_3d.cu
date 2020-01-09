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

//convert all 3d variables of the last timestep to float and interpolate to a 3 times finer grid in phi
//we need a time variable when available
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
    std::vector<std::string> names_static_input{
        "xc", "yc", "zc", "BR", "BZ", "BP", "Nprof", "Psip", "Source"
    }
    std::vector<std::string> names_input{
        "electrons", "ions", "Ue", "Ui", "potential", "induction"
    };

    //-----------------Create Netcdf output file with attributes----------//
    int ncid_out;
    err = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid_out);

    /// Set global attributes
    std::map<std::string, std::string> att;
    att["title"] = "Output file of feltor/src/feltor/interpolate_in_3d.cu";
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
        err = nc_put_att_text( ncid_out, NC_GLOBAL,
            pair.first.data(), pair.second.size(), pair.second.data());

    //-------------------Construct grids-------------------------------------//

    const float Rmin=gp.R_0-p.boxscaleRm*gp.a;
    const float Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    const float Rmax=gp.R_0+p.boxscaleRp*gp.a;
    const float Zmax=p.boxscaleZp*gp.a*gp.elongation;

    dg::RealCylindricalGrid3d<float> g3d_out( Rmin,Rmax, Zmin,Zmax, 0, 2*M_PI,
        p.n_out, p.Nx_out, p.Ny_out, 3*p.Nz_out, p.bcxN, p.bcyN, dg::PER);

    // Construct weights and temporaries
    dg::fHVec transferH3d = dg::evaluate(dg::zero,g3d_out);

    // define 2d and 1d and 0d dimensions and variables
    int dim_ids[3];
    err = file::define_dimensions( ncid_out, dim_ids, g3d_out, {"z", "y", "x"});
    std::map<std::string, int> id3d;

    //write 1d static vectors (psi, q-profile, ...) into file
    for( auto tp : map1d)
    {
        int vid;
        err = nc_def_var( ncid_out, std::get<0>(tp).data(), NC_DOUBLE, 1,
            &dim_ids1d[1], &vid);
        err = nc_put_att_text( ncid_out, vid, "long_name",
            std::get<2>(tp).size(), std::get<2>(tp).data());
        err = nc_enddef( ncid);
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
    int timeID;
    double time=0.;

    size_t steps;
    err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
    err = nc_inq_unlimdim( ncid, &timeID); //Attention: Finds first unlimited dim, which hopefully is time and not energy_time
    err = nc_inq_dimlen( ncid, timeID, &steps);
    //steps = 3;
    for( unsigned i=0; i<steps; i++)//timestepping
    {
        start2d[0] = i;
        start1d[0] = i;
        // read and write time
        err = nc_get_vara_double( ncid, timeID, start2d, count2d, &time);
        std::cout << "Timestep = " << i << "  time = " << time << std::endl;
        err = nc_put_vara_double( ncid_out, tvarID, start2d, count2d, &time);
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
            } catch ( file::NC_Error error)
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
                //2. Compute fsa and output fsa
                dg::blas2::symv( grid2gridX2d, transferH2d, transferH2dX); //interpolate onto X-point grid
                dg::blas1::pointwiseDot( transferH2dX, volX2d, transferH2dX); //multiply by sqrt(g)
                poloidal_average( transferH2dX, t1d, false); //average over eta
                dg::blas1::scal( t1d, 4*M_PI*M_PI*f0); //
                dg::blas1::pointwiseDivide( t1d, dvdpsip, fsa1d );
                if( record_name[0] == 'j')
                    dg::blas1::pointwiseDot( fsa1d, dvdpsip, fsa1d );
                //3. Interpolate fsa on 2d plane : <f>
                dg::blas2::gemv(fsa2rzmatrix, fsa1d, transferH2d); //fsa on RZ grid
            }
            else
            {
                dg::blas1::scal( fsa1d, 0.);
                dg::blas1::scal( transferH2d, 0.);
            }
            err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_fsa"),
                start1d, count1d, fsa1d.data());
            err = nc_put_vara_double( ncid_out, id2d.at(record_name+"_fsa2d"),
                start2d, count2d, transferH2d.data() );
            //4. Read 2d variable and compute fluctuations
            available = true;
            try{
                err = nc_inq_varid(ncid, (record.name+"_2d").data(), &dataID);
            } catch ( file::NC_Error error)
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
                    dg::blas1::pointwiseDot( fsa1d, dvdpsip, t1d);
                    transfer1d = dg::integrate( t1d, g1d_out);

                    result = dg::interpolate( transfer1d, 0., g1d_out);
                }
                err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_ifs"),
                    start1d, count1d, transfer1d.data());
                //flux surface integral/derivative on last closed flux surface
                err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_lcfs"),
                    start2d, count2d, &result );
                //6. Compute norm of time-integral terms to get relative importance
                if( record_name[0] == 'j') //j indicates a flux
                {
                    dg::blas2::symv( dpsi, fsa1d, t1d);
                    dg::blas1::pointwiseDivide( t1d, dvdpsip, t1d); //dvjv
                    dg::blas1::pointwiseDot( t1d, t1d, t1d);//dvjv2
                    dg::blas1::pointwiseDot( t1d, dvdpsip, t1d);//dvjv2
                    transfer1d = dg::integrate( t1d, g1d_out);
                    result = dg::interpolate( transfer1d, 0., g1d_out);
                    result = sqrt(result);
                }
                else
                {
                    dg::blas1::pointwiseDot( fsa1d, fsa1d, t1d);
                    dg::blas1::pointwiseDot( t1d, dvdpsip, t1d);
                    transfer1d = dg::integrate( t1d, g1d_out);

                    result = dg::interpolate( transfer1d, 0., g1d_out);
                    result = sqrt(result);
                }
                err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_norm"),
                    start2d, count2d, &result );
            }
            else
            {
                dg::blas1::scal( transferH2d, 0.);
                dg::blas1::scal( transfer1d, 0.);
                double result = 0.;
                err = nc_put_vara_double( ncid_out, id2d.at(record_name+"_fluc2d"),
                    start2d, count2d, transferH2d.data() );
                err = nc_put_vara_double( ncid_out, id1d.at(record_name+"_ifs"),
                    start1d, count1d, transfer1d.data());
                err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_lcfs"),
                    start2d, count2d, &result );
                err = nc_put_vara_double( ncid_out, id0d.at(record_name+"_ifs_norm"),
                    start2d, count2d, &result );
            }

        }


    } //end timestepping
    err = nc_close(ncid);
    err = nc_close(ncid_out);

    return 0;
}
