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

thrust::host_vector<float> append( const thrust::host_vector<float>& in, const dg::aRealTopology3d<double>& g)
{
    unsigned size2d = g.n()*g.n()*g.Nx()*g.Ny();
    thrust::host_vector<float> out(g.size()+size2d);
    for( unsigned i=0; i<g.size(); i++)
        out[i] = in[i];
    for( unsigned i=0; i<size2d; i++)
        out[g.size()+i] = in[i];
    return out;
}
//convert all 3d variables of every TIME_FACTOR-th timestep to float
//and interpolate to a FACTOR times finer grid in phi
//also periodify in 3d and equidistant in RZ
// Also plot in a fieldaligned coordinate system
//input should probably better come from another json file
// depends on diagnostics3d_list and diagnostics3d_static_list
int main( int argc, char* argv[])
{
    if( argc != 4)
    {
        std::cerr << "Usage: "<<argv[0]<<" [config.json] [input.nc] [output.nc]\n";
        return -1;
    }
    std::cout << argv[1]<< " "<<argv[2]<< " -> "<<argv[3]<<std::endl;

    //------------------------open input nc file--------------------------------//
    dg::file::NC_Error_Handle err;
    int ncid_in;
    err = nc_open( argv[2], NC_NOWRITE, &ncid_in); //open 3d file
    dg::file::WrappedJsonValue jsin = dg::file::nc_attrs2json( ncid_in, NC_GLOBAL);
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json(jsin["inputfile"].asString(),
        dg::file::comments::are_forbidden);
    const feltor::Parameters p(js);
    std::cout << js.toStyledString() << std::endl;
    dg::file::WrappedJsonValue config( dg::file::error::is_warning);
    try{
        config.asJson() = dg::file::file2Json( argv[1],
                dg::file::comments::are_discarded, dg::file::error::is_warning);
    } catch( std::exception& e) {
        DG_RANK0 std::cerr << "ERROR in input file "<<argv[1]<<std::endl;
        DG_RANK0 std::cerr << e.what()<<std::endl;
        return -1;
    }
    const unsigned INTERPOLATE = config.get( "fine-grid-factor", 2).asUInt();
    const unsigned TIME_FACTOR = config.get( "time-reduction-factor", 1).asUInt();
    dg::geo::TokamakMagneticField mag;
    try{
        mag = dg::geo::createMagneticField(js["magnetic_field"]["params"]);
    }catch(std::runtime_error& e)
    {
        std::cerr << "ERROR in geometry file "<<argv[2]<<std::endl;
        std::cerr <<e.what()<<std::endl;
        return -1;
    }

    //-----------------Create Netcdf output file with attributes----------//
    int ncid_out;
    err = nc_create(argv[3],NC_NETCDF4|NC_NOCLOBBER, &ncid_out);

    /// Set global attributes
    dg::file::JsonType att;
    att["title"] = "Output file of feltor/src/feltor/interpolate_in_3d.cpp";
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
    att["inputfile"] = jsin["inputfile"].asString();
    dg::file::json2nc_attrs( att, ncid_out, NC_GLOBAL);

    //-------------------Construct grids-------------------------------------//

    auto box = common::box( js);

    unsigned cx = js["output"]["compression"].get(0u,1).asUInt();
    unsigned cy = js["output"]["compression"].get(1u,1).asUInt();
    unsigned n_out = p.n, Nx_out = p.Nx/cx, Ny_out = p.Ny/cy, Nz_out = p.Nz;
    dg::RealCylindricalGrid3d<double> g3d_in( box.at("Rmin"),box.at("Rmax"), box.at("Zmin"),box.at("Zmax"), 0, 2*M_PI,
        n_out, Nx_out, Ny_out, Nz_out, p.bcxN, p.bcyN, dg::PER);
    dg::RealCylindricalGrid3d<double> g3d_out( box.at("Rmin"),box.at("Rmax"), box.at("Zmin"),box.at("Zmax"), 0, 2*M_PI,
        n_out, Nx_out, Ny_out, INTERPOLATE*Nz_out, p.bcxN, p.bcyN, dg::PER);
    dg::RealCylindricalGrid3d<double> g3d_out_equidistant( box.at("Rmin"),box.at("Rmax"),
            box.at("Zmin"),box.at("Zmax"), 0, 2*M_PI, 1, n_out*Nx_out, n_out*Ny_out,
            INTERPOLATE*Nz_out, p.bcxN, p.bcyN, dg::PER);
    dg::RealCylindricalGrid3d<float> g3d_out_periodic( box.at("Rmin"),box.at("Rmax"), box.at("Zmin"),box.at("Zmax"), 0,
            2*M_PI+g3d_out.hz(), n_out, Nx_out, Ny_out, INTERPOLATE*Nz_out+1,
            p.bcxN, p.bcyN, dg::PER);
    dg::RealCylindricalGrid3d<float> g3d_out_periodic_equidistant( box.at("Rmin"),box.at("Rmax"),
            box.at("Zmin"),box.at("Zmax"), 0, 2*M_PI+g3d_out.hz(), 1, n_out*Nx_out, n_out*Ny_out,
            INTERPOLATE*Nz_out+1, p.bcxN, p.bcyN, dg::PER);
    // For field-aligned output
    dg::RealCylindricalGrid3d<double> g3d_out_fieldaligned( box.at("Rmin"),box.at("Rmax"),
            box.at("Zmin"),box.at("Zmax"), -2*M_PI, 2*M_PI, 1, n_out*Nx_out, n_out*Ny_out, 2*Nz_out,
            p.bcxN, p.bcyN, dg::PER);
    dg::RealGrid2d<double> g2d( box.at("Rmin"),box.at("Rmax"),box.at("Zmin"),box.at("Zmax"), 1, n_out*Nx_out,
            n_out*Ny_out, p.bcxN, p.bcyN);


    // Construct weights and temporaries
    dg::HVec transferH_in = dg::evaluate(dg::zero,g3d_in);
    dg::HVec transferH_out = dg::evaluate(dg::zero,g3d_out);
    dg::HVec transferH = dg::evaluate(dg::zero,g3d_out_equidistant);
    dg::fHVec transferH_out_float = dg::construct<dg::fHVec>( transferH);
    // Construct for field-aligned output
    dg::HVec transferH_aligned_out = dg::evaluate( dg::zero, g3d_out_fieldaligned);
    dg::HVec RR = dg::evaluate( dg::zero, g3d_out_fieldaligned), ZZ(RR), PP(RR);
    std::array<thrust::host_vector<double>,3> yy0{
        dg::evaluate( dg::cooX2d, g2d),
        dg::evaluate( dg::cooY2d, g2d),
        dg::evaluate( dg::zero, g2d)}, yy1(yy0); //s
    auto bhat = dg::geo::createBHat( mag);
    dg::geo::detail::DSFieldCylindrical3 cyl_field(bhat);
    dg::Adaptive<dg::ERKStep<std::array<double,3>>> adapt(
            "Dormand-Prince-7-4-5", std::array<double,3>{0,0,0});
    double eps = 1e-5;
    dg::AdaptiveTimeloop< std::array<double,3>> odeint( adapt,
            cyl_field, dg::pid_control, dg::fast_l2norm, eps, 1e-10);
    double deltaPhi = g3d_out_fieldaligned.hz();
    double phi0 = deltaPhi/2.;
    for( unsigned i=0; i<g2d.size(); i++)
    {
        RR[ Nz_out*g2d.size() + i] = yy0[0][i];
        ZZ[ Nz_out*g2d.size() + i] = yy0[1][i];
        PP[ Nz_out*g2d.size() + i] = phi0;
    }
    std::cout << "# Integrate Fieldlines in + direction ...\n";
    for( unsigned k=1; k<Nz_out; k++)
    {
        std::cout << "# Plane "<<k<<" / "<<Nz_out<<"\n";
        for( unsigned i=0; i<g2d.size(); i++)
        {
            double phi1 = phi0 + deltaPhi;
            std::array<double,3> coords0{yy0[0][i],yy0[1][i],yy0[2][i]}, coords1;
            odeint.integrate_in_domain( phi0, coords0, phi1, coords1, 0., g2d, eps);
            yy1[0][i] = coords1[0], yy1[1][i] = coords1[1], yy1[2][i] = coords1[2];
            //now write into right place in RR ...
            RR[ (Nz_out + k)*g2d.size() + i] = yy1[0][i];
            ZZ[ (Nz_out + k)*g2d.size() + i] = yy1[1][i];
            // Note that phi1 can be different from phi0 + deltaPhi
            PP[ (Nz_out + k)*g2d.size() + i] = phi1;
        }
        std::swap( yy0, yy1);
        phi0 += deltaPhi;
    }
    phi0 = deltaPhi/2.;
    yy0 = std::array<thrust::host_vector<double>,3> {
        dg::evaluate( dg::cooX2d, g2d),
        dg::evaluate( dg::cooY2d, g2d),
        dg::evaluate( dg::zero, g2d)};
    std::cout << "# Integrate Fieldlines in - direction ...\n";
    for( unsigned k=0; k<Nz_out; k++)
    {
        std::cout << "# Plane "<<k<<" / "<<Nz_out<<"\n";
        for( unsigned i=0; i<g2d.size(); i++)
        {
            double phi1 = phi0 - deltaPhi;
            std::array<double,3> coords0{yy0[0][i],yy0[1][i],yy0[2][i]}, coords1;
            odeint.integrate_in_domain( phi0, coords0, phi1, coords1, 0., g2d, eps);
            yy1[0][i] = coords1[0], yy1[1][i] = coords1[1], yy1[2][i] = coords1[2];
            //now write into right place in RR ...
            RR[ (Nz_out - k - 1)*g2d.size() + i] = yy1[0][i];
            ZZ[ (Nz_out - k - 1)*g2d.size() + i] = yy1[1][i];
            // Note that phi1 can be different from phi0 - deltaPhi
            PP[ (Nz_out - k - 1)*g2d.size() + i] = phi1 + 2*M_PI;
        }
        std::swap( yy0, yy1);
        phi0 -= deltaPhi;
    }
    dg::IHMatrix big_matrix
        = dg::create::interpolation( RR, ZZ, PP, g3d_in, p.bcxN, p.bcyN, dg::PER, "linear");

    /////////////////////////////////////////////////////////////////////////
    dg::geo::Fieldaligned<dg::CylindricalGrid3d, dg::IHMatrix, dg::HVec> fieldaligned(
        bhat, g3d_out, dg::NEU, dg::NEU, dg::geo::NoLimiter(),
        //let's take NEU bc because N is not homogeneous
        p.rk4eps, 5, 5, -1, "dg" );
    dg::IHMatrix interpolate_in_2d = dg::create::interpolation(
            g3d_out_equidistant, g3d_out);


    {
    dg::file::Writer<dg::RealCylindricalGrid3d<float>> write_periodic_stat(
        ncid_out, g3d_out_periodic_equidistant, {"z", "y", "x"});
    dg::file::Reader<dg::RealCylindricalGrid3d<double>> read( ncid_in, g3d_in, {"z", "y", "x"});

    for( auto& record : feltor::diagnostics3d_static_list)
    {
        if( record.name != "xc" && record.name != "yc" && record.name != "zc" )
        {
            read.get( record.name, transferH_in);

            transferH_out = fieldaligned.interpolate_from_coarse_grid(
                g3d_in, transferH_in);
            dg::blas2::symv( interpolate_in_2d, transferH_out, transferH);
            dg::assign( transferH, transferH_out_float);
            write_periodic_stat.def_and_put( record.name, dg::file::long_name(
                record.long_name), append( transferH_out_float,
                g3d_out_equidistant));
        }
        else
        {
            record.function ( transferH_out, mag, g3d_out_equidistant);
            dg::assign( transferH_out, transferH_out_float);
            write_periodic_stat.def_and_put( record.name, dg::file::long_name(
                record.long_name), append( transferH_out_float,
                g3d_out_equidistant));
        }
    }
    }
    // for fieldaligned output (transform to Cartesian coords)
    {
    dg::HVec XXc(RR), YYc(RR), ZZc(ZZ);
    dg::blas1::evaluate( XXc, dg::equals(),[] DG_DEVICE (double R, double P){
            return R*sin(P);}, RR, PP);
    dg::blas1::evaluate( YYc, dg::equals(),[] DG_DEVICE (double R, double P){
            return R*cos(P);}, RR, PP);
    std::vector<dg::file::Record<void( dg::HVec&) >> list = {
        { "xfc", "xf-coordinate in Cartesian coordinate system",
            [&]( dg::HVec& result) { result = XXc; }},
        { "yfc", "yf-coordinate in Cartesian coordinate system",
            [&]( dg::HVec& result) { result = YYc; }},
        { "zfc", "zf-coordinate in Cartesian coordinate system",
            [&]( dg::HVec& result) { result = ZZc; }},
    };
    dg::file::WriteRecordsList<dg::RealCylindricalGrid3d<double>> ( ncid_out,
        g3d_out_fieldaligned, {"zf", "yf", "xf"}).write( list);
    }

    // define 4d dimension
    dg::file::Writer<dg::RealCylindricalGrid3d<float>> write_periodic(
        ncid_out, g3d_out_periodic_equidistant, {"time", "z", "y", "x"});
    dg::file::Writer<dg::RealGrid0d<float>> write0d( ncid_out, {}, {"time"});

    dg::file::Writer<dg::RealCylindricalGrid3d<double>> write_fieldaligned(
        ncid_out, g3d_out_fieldaligned, {"timef", "zf", "yf", "xf"});
    dg::file::Writer<dg::RealGrid0d<double>> write0dF( ncid_out, {}, {"timef"});

    for( auto& record : feltor::diagnostics3d_list)
    {
        write_periodic.def( record.name, dg::file::long_name( record.long_name));
        write_fieldaligned.def( record.name+"FF", dg::file::long_name( record.long_name));
    }

    double time=0.;

    dg::file::Reader<dg::Grid0d> read0d( ncid_in, {}, {"time"});
    dg::file::Reader<dg::CylindricalGrid3d> read3d( ncid_in, g3d_in, {"time",
        "z", "y", "x"});
    size_t steps = read0d.size();
    auto names = read3d.names();

    ///////////////////////////////////////////////////////////////////////
    for( unsigned i=0; i<steps; i+=TIME_FACTOR)//timestepping
    {
        std::cout << "Timestep = "<<i<< "/"<<steps;
        // read and write time
        read0d.get( "time", time, i);
        std::cout << "  time = " << time << std::endl;
        write0d.put(  "time", (float)time,  i/TIME_FACTOR);
        write0dF.put( "timef", time,  i/TIME_FACTOR);
        for( auto& record : feltor::diagnostics3d_list)
        {
            std::string record_name = record.name;
            bool available = true;

            if( std::find( names.begin(), names.end(), record.name) == names.end())
            {
                if(  i == 0)
                {
                    std::cerr << "Variable "<<record.name<<" not found!" <<std::endl;
                    std::cerr << "Writing zeros ... \n";
                }
                available = false;
            }
            if( available)
            {
                read3d.get( record.name, transferH_in);
                transferH_out = fieldaligned.interpolate_from_coarse_grid(
                    g3d_in, transferH_in);
                dg::blas2::symv( interpolate_in_2d, transferH_out, transferH);
                dg::assign( transferH, transferH_out_float);
                // and the fieldaligned variables
                dg::blas2::symv( big_matrix, transferH_in, transferH_aligned_out);
            }
            else
            {
                dg::blas1::scal( transferH_out_float, (float)0);
                dg::blas1::scal( transferH_aligned_out, (double)0);
            }
            write_periodic.put( record.name, append(transferH_out_float,
                    g3d_out_equidistant), i/TIME_FACTOR);
            write_fieldaligned.put( record.name+"FF",
                    transferH_aligned_out, i/TIME_FACTOR);
        }

    } //end timestepping
    err = nc_close(ncid_in);
    err = nc_close(ncid_out);

    return 0;
}
