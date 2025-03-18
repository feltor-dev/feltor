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
    dg::file::NcFile file_in( argv[2], dg::file::nc_nowrite);
    std::string inputfile = file_in.get_att_as<std::string>( "inputfile");
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() =
        dg::file::string2Json(inputfile, dg::file::comments::are_forbidden);
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
    dg::file::NcFile file_out ( argv[3], dg::file::nc_noclobber);

    /// Set global attributes
    std::map<std::string, dg::file::nc_att_t> att;
    att["title"] = "Output file of feltor/src/feltor/interpolate_in_3d.cpp";
    att["Conventions"] = "CF-1.7";
    att["history"] = dg::file::timestamp(argc, argv);
    att["comment"] = "Find more info in feltor/src/feltor.tex";
    att["source"] = "FELTOR";
    att["references"] = "https://github.com/feltor-dev/feltor";
    att["inputfile"] = inputfile;
    file_out.put_atts( att);

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

    { // Static write
    file_out.defput_dim( "x", {{"axis", "X"}},
            g3d_out_periodic_equidistant.abscissas(0));
    file_out.defput_dim( "y", {{"axis", "Y"}},
            g3d_out_periodic_equidistant.abscissas(1));
    file_out.defput_dim( "z", {{"axis", "Z"}},
            g3d_out_periodic_equidistant.abscissas(2));

    for( auto& record : feltor::diagnostics3d_static_list)
    {
        if( record.name != "xc" && record.name != "yc" && record.name != "zc" )
        {
            file_in.get_var( record.name, {g3d_in}, transferH_in);

            transferH_out = fieldaligned.interpolate_from_coarse_grid(
                g3d_in, transferH_in);
            dg::blas2::symv( interpolate_in_2d, transferH_out, transferH);
            dg::assign( transferH, transferH_out_float);
            file_out.defput_var( record.name, {"z", "y", "x"}, record.atts,
                    {g3d_out_periodic_equidistant},
                    append( transferH_out_float, g3d_out_equidistant));
        }
        else
        {
            record.function ( transferH_out, mag, g3d_out_equidistant);
            dg::assign( transferH_out, transferH_out_float);
            file_out.defput_var( record.name, {"z", "y", "x"}, record.atts,
                    {g3d_out_periodic_equidistant},
                    append( transferH_out_float, g3d_out_equidistant));
        }
    }
    }
    // for fieldaligned output (transform to Cartesian coords)
    {
    file_out.defput_dim( "xf", {{"axis", "X"}},
            g3d_out_fieldaligned.abscissas(0));
    file_out.defput_dim( "yf", {{"axis", "Y"}},
            g3d_out_fieldaligned.abscissas(1));
    file_out.defput_dim( "zf", {{"axis", "Z"}},
            g3d_out_fieldaligned.abscissas(2));
    dg::HVec XXc(RR), YYc(RR), ZZc(ZZ);
    dg::blas1::evaluate( XXc, dg::equals(),[] DG_DEVICE (double R, double P){
            return R*sin(P);}, RR, PP);
    dg::blas1::evaluate( YYc, dg::equals(),[] DG_DEVICE (double R, double P){
            return R*cos(P);}, RR, PP);
    file_out.defput_var( "xfc", {"zf", "yf", "xf"},
            {{"long_name", "xf-coordinate in Cartesian coordinate system"}},
            {g3d_out_fieldaligned}, XXc);
    file_out.defput_var( "yfc", {"zf", "yf", "xf"},
            {{"long_name", "yf-coordinate in Cartesian coordinate system"}},
            {g3d_out_fieldaligned}, YYc);
    file_out.defput_var( "zfc", {"zf", "yf", "xf"},
            {{"long_name", "zf-coordinate in Cartesian coordinate system"}},
            {g3d_out_fieldaligned}, ZZc);
    }

    // define 4d dimension
    file_out.def_dimvar_as<float>( "time", NC_UNLIMITED, {{"axis", "T"}});
    file_out.def_dimvar_as<double>( "timef", NC_UNLIMITED, {{"axis", "T"}});

    for( auto& record : feltor::diagnostics3d_list)
    {
        file_out.def_var_as<float>( record.name, {"time", "z", "y", "x"},
                record.atts);
        file_out.def_var_as<double>( record.name+"FF", {"timef", "zf", "yf",
                "xf"}, record.atts);
    }

    double time=0.;

    size_t steps = file_in.get_dim_size( "time");
    auto names = file_in.get_var_names();
    // actually we only need names with certain dimensions

    ///////////////////////////////////////////////////////////////////////
    for( unsigned i=0; i<steps; i+=TIME_FACTOR)//timestepping
    {
        std::cout << "Timestep = "<<i<< "/"<<steps;
        // read and write time
        file_in.get_var("time", {i}, time);
        std::cout << "  time = " << time << std::endl;
        file_out.put_var( "time", {i/TIME_FACTOR}, (float)time);
        file_out.put_var( "timef", {i/TIME_FACTOR}, time);
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
                file_in.get_var( record.name, {i,g3d_in}, transferH_in);
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
            file_out.put_var( record.name, {i/TIME_FACTOR,
                g3d_out_periodic_equidistant}, append( transferH_out_float,
                        g3d_out_equidistant));
            file_out.put_var( record.name+"FF", {i/TIME_FACTOR,
                g3d_out_fieldaligned}, transferH_aligned_out);
        }

    } //end timestepping
    file_in.close();
    file_out.close();

    return 0;
}
