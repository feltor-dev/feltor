#pragma once

#include "dg/file/json_utilities.h"
#include "init_from_file.h"
#include "../feltor/init.h"

namespace thermal
{

/* The purpose of this file is to provide an interface for custom initial conditions and
 * source profiles.  Just add your own to the relevant map below.
 * @note y0[3,4,5] havec to be staggered velocity, qperp and qpara
 */

std::array<std::vector<dg::x::DVec>,6> initial_conditions(
    Explicit<dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec>& thermal, // for fieldaligned
    const dg::x::CylindricalGrid3d& grid, const thermal::Parameters& p,
    const dg::geo::TokamakMagneticField& mag,
    const dg::geo::TokamakMagneticField& unmod_mag,
    dg::file::WrappedJsonValue jsspecies, // input["species"]
    double & time, dg::geo::CylindricalFunctor& sheath_coordinate )
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    // First, let's scan for restart
    for( unsigned s=0; s<p.num_species; s++)
    {
        dg::file::WrappedJsonValue js = jsspecies[s]["init"];
        std::string type = js["type"].asString();
        if( "fields" == type)
            continue;
        else if( "restart" == type)
        {
            std::string file = js["file"].asString();
            return init_from_file( file, grid, p, time);
        }
        else
            throw dg::Error(dg::Message()<< "Invalid initial condition "<<type<<"\n");
    }
    time = 0;
    std::array<std::vector<dg::x::DVec>,6> y0;
    for( unsigned i=0; i<6; i++)
    {
        y0[i].resize( p.num_species);
        for( unsigned s=0; s<p.num_species; s++)
            y0[i][s] = dg::construct<dg::x::DVec>( dg::evaluate( dg::zero, grid));
    }
    unsigned num_quasineutral_n = 0, idx_quasineutral_n = 0;
    // Init physical n, pperp, ppara
    for( unsigned s=0; s<p.num_species; s++)
    {
        dg::file::WrappedJsonValue js = jsspecies[s]["init"];
        DG_RANK0  std::cout << "# Initializing species "<<p.name[s]<<"\n";

        const std::array<std::string,3> eqs = { "density", "pperp", "ppara"};
        // Think of this as initialising the **physical** quantities
        for( unsigned u=0; u<3; u++)
        {
            std::string field = eqs[u];
            std::string ntype = js[field].get("type", "zero").asString();
            DG_RANK0 std::cout << "# Initialize "<<field<<" with "<<ntype << std::endl;
            if ( "const" == ntype)
            {
                double nbg = js[field].get("background", 0.1).asDouble();
                y0[u][s] = dg::construct<dg::x::DVec>(
                        dg::evaluate( dg::CONSTANT( nbg), grid));
            }
            else if( "profile" == ntype )
            {
                double nbg = 0.;
                dg::x::HVec ntilde  = feltor::detail::make_ntilde(  thermal, grid, mag,
                        js[field]["ntilde"]);
                dg::x::HVec profile = feltor::detail::make_profile( grid, mag,
                        js[field]["profile"], nbg);
                dg::x::HVec damping = feltor::detail::make_damping( grid, unmod_mag,
                        js[field]["damping"]);
                dg::x::HVec density = profile;
                dg::blas1::subroutine( [nbg]( double profile, double
                    ntilde, double damping, double& density)
                        { density = (profile+ntilde-nbg)*damping+nbg;},
                        profile, ntilde, damping, density);
                dg::assign( density, y0[u][s]);
            }
            else if ( "density" == field and "quasineutral" == ntype)
            {
                num_quasineutral_n ++ ;
                idx_quasineutral_n = s;
            }
            else
                throw dg::Error(dg::Message()<< "Invalid "<<field<<" initial condition "<<ntype<<"\n");
        }
    }
    // Check quasineutrality
    if( num_quasineutral_n > 1)
        throw std::runtime_error( "There cannot be more than one quasineutral density");
    if( num_quasineutral_n == 0)
        ;//TODO Check that species indeed add up to 0
    else
    {
        for( unsigned s=0; s<p.num_species; s++)
        {
            if( s == idx_quasineutral_n) continue;
            dg::blas1::axpby( -p.z[s]/ p.z[idx_quasineutral_n], y0[0][s], 1.,
                y0[0][idx_quasineutral_n]);
        }
    }
    if( p.isothermal)
    {
        y0[1] = y0[2] = y0[0];
    }
    // Now transform to gyro-centre density
    for( unsigned s=0; s<p.num_species; s++)
    {
        dg::file::WrappedJsonValue js = jsspecies[s]["init"];
        std::string invert = js["density"].get("invert", "none").asString();
        if( "none" == invert)
            continue; // N_s = n_s
        else if( "lwl" == invert)
        {
            // Thermal trafo runs on device so we need to move data back and forth
            dg::x::DVec density = y0[0][s], pperp = y0[1][s], phi = dg::evaluate( dg::zero, grid);
            thermal.sources().transform_density_pperp( s, density, pperp, phi, density, pperp);
            y0[0][s] = density;
            double minimalni = dg::blas1::reduce( y0[0][s], 1e10,
                    thrust::minimum<double>());
            DG_RANK0 std::cerr << "# Minimum Ns "<<s<<" value "<<minimalni<<std::endl;
            //actually we should always invert according to Markus
            //because the dG direct application is supraconvergent
        }
        else
            throw dg::Error(dg::Message()<< "Invalid density invert type "<<invert<<"\n");

    }

    unsigned num_quasineutral_u = 0, idx_quasineutral_u = 0;
    // Init physical u, qperp, qpara
    for( unsigned s=0; s<p.num_species; s++)
    {
        // init (staggered) velocity and thus canonical W
        dg::file::WrappedJsonValue js = jsspecies[s]["init"];
        std::string utype = js["velocity"].get("type", "zero").asString();
        DG_RANK0 std::cout << "# Initialize velocity with "<<utype << std::endl;
        if( "zero" == utype)
            ; // velocity already is zero
        else if( "linear_cs" == utype )
        {
            if( s == 0)
                throw std::runtime_error( "Linear_cs velocity not available for electrons!");
            // Assume toroidal alignment ot T in sheath
            double zs = p.z[s], ms = p.mu[s];
            dg::blas1::evaluate( y0[3][s], dg::equals(),
                [ zs , ms ] DG_DEVICE( double ns, double ps, double ne, double pe)
                {
                    return sqrt( ( ps/ns + zs * pe/ne )/ms);
                }, y0[0][s], y0[2][s], y0[0][0], y0[2][0]); // n, ppara and ppara_e initialized previously
            // sheath coordinate
            std::unique_ptr<dg::x::aGeometry2d> perp_grid_ptr( grid.perp_grid());
            dg::x::HVec coord2d = dg::pullback( sheath_coordinate,
                    *perp_grid_ptr);
            dg::x::HVec ui;
            dg::assign3dfrom2d( coord2d, ui, grid);
            dg::x::DVec ui_device = ui;
            dg::blas1::pointwiseDot( ui_device, y0[3][s], y0[3][s]);
        }
        else if( "quasineutral" == utype)
        {
            num_quasineutral_u ++;
            idx_quasineutral_u = s;
        }
        else
            throw dg::Error(dg::Message()<< "Invalid velocity initial condition "<<utype<<"\n");

        std::string qperptype = js["qperp"].get("type", "zero").asString();
        DG_RANK0 std::cout << "# Initialize qperp with "<<utype << std::endl;
        if( "zero" == qperptype)
            ; // qperp already is zero
        else
            throw dg::Error(dg::Message()<< "Invalid qperp initial condition "<<qperptype<<"\n");

        std::string qparatype = js["qpara"].get("type", "zero").asString();
        DG_RANK0 std::cout << "# Initialize qperp with "<<utype << std::endl;
        if( "zero" == qparatype)
            ; // qpara
        else
            throw dg::Error(dg::Message()<< "Invalid qpara initial condition "<<qparatype<<"\n");
    }
    // Init quasineutral species
    if( num_quasineutral_u > 1)
        throw std::runtime_error( "There cannot be more than one quasineutral velocity");
    if( num_quasineutral_u == 0)
        ;//TODO Check that species indeed add up to 0
    else
    {
        // Assume densities are toroidally aligned (so nST = n)
        for( unsigned s=0; s<p.num_species ; s++)
        {
            if( s == idx_quasineutral_u)
                continue;
            double zk = p.z[idx_quasineutral_u], zs = p.z[s];
            dg::blas1::evaluate( y0[3][idx_quasineutral_u], dg::plus_equals(),
                [ zk, zs] DG_DEVICE(
                    double nk, double ns, double us)
                {
                    return -zs * ns * us / (zk * nk);
                }, y0[0][idx_quasineutral_u], y0[0][s], y0[3][s]);
        }
    }
    // Since total current is 0, we have A_\parallel = 0

    return y0;
};

// init physical influx / forced profiles for n, pperp, ppara
// For "fixed profile" the returned source represents the damping profile, else it is the source without source rate (i.e. source profile/region)
std::array<std::vector<dg::x::HVec>,3> source_profiles(
    Explicit<dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec>& thermal,
    std::array<std::vector<bool>,3>& fixed_profile,     //indicate whether a profile should be forced (yes or no)
    std::array<std::vector<double>,3>& source_rate,     // source rates
    std::array<std::vector<dg::x::HVec>,3>& profile,    // if fixed_profile is yes you need to construct something here, if no then you can ignore the parameter; if you construct something it will show in the output file
    const dg::x::CylindricalGrid3d& grid,
    const dg::geo::TokamakMagneticField& mag,
    const dg::geo::TokamakMagneticField& unmod_mag,
    dg::file::WrappedJsonValue jsspecies, // input["species"]
    std::vector<double>& minn,
    double& mint,
    double& minrate,
    double& minbeta
    )
{
    unsigned num_species = jsspecies.size();

    // Minimum density
    minn.resize( num_species);
    for( unsigned s=0; s<num_species; s++)
    {
        dg::file::WrappedJsonValue js = jsspecies[s]["source"];
        minn[s] = js.get("minn", 0.).asDouble();
        if( s == 0)
        {
            mint = js.get("mint", 0.).asDouble();
            minrate = js.get("minrate", 1.).asDouble();
            minbeta = js.get("minbeta", 0.25).asDouble();
        }
        if( s != 0)
        {
            double mints = js.get("mint", 0.).asDouble();
            double minrates = js.get("minrate", 1.).asDouble();
            double minbetas = js.get("minbeta", 0.25).asDouble();
            if( mints != mint)
                throw dg::Error(dg::Message()<< "Mint must be equal among species "<<mints<<" vs "<<mint<<"!\n");
            if( minrates != minrate)
                throw dg::Error(dg::Message()<< "Minrate must be equal among species "<<minrates<<" vs "<<minrate<<"!\n");
            if( minbetas != minbeta)
                throw dg::Error(dg::Message()<< "Minbeta must be equal among species "<<minbetas<<" vs "<<minbeta<<"!\n");
        }
    }

    // Source profiles
    std::array<std::vector<dg::x::HVec>,3> source;
    for( unsigned u=0; u<3; u++)
    {
        profile[u].resize( num_species);
        fixed_profile[u].resize( num_species);
        source[u].resize( num_species);
        source_rate[u].resize( num_species);
    }
    unsigned num_quasineutral_n = 0, idx_quasineutral_n = 0;
    for( unsigned s=0; s<num_species; s++)
    {
        dg::file::WrappedJsonValue js = jsspecies[s]["source"];
        const std::array<std::string,3> eqs = { "density", "pperp", "ppara"};
        // Think of this as initialising the **physical** quantities
        for( unsigned u=0; u<3; u++)
        {
            std::string field = eqs[u];
            fixed_profile[u][s] = false;
            profile[u][s] = dg::evaluate( dg::zero, grid);
            source[u][s] = dg::evaluate( dg::zero, grid);
            source_rate[u][s] = 0.;

            std::string type  = js[field].get( "type", "zero").asString();
            if( "zero" == type)
            {
                ;
            }
            else if( "fixed_profile" == type)
            {
                source_rate[u][s] = js[field].get( "rate", 1e-3).asDouble();
                fixed_profile[u][s] = true;
                double nbg = 0;
                profile[u][s] = feltor::detail::make_profile(grid, mag, js[field]["profile"], nbg);
                source[u][s] = feltor::detail::make_damping( grid, unmod_mag, js[field]["damping"]);
            }
            else if("influx" == type)
            {
                double nbg = 0.;
                source_rate[u][s] = js[field].get( "rate", 1e-3).asDouble();
                source[u][s]  = feltor::detail::make_ntilde( thermal, grid, mag, js[field]["ntilde"]);
                profile[u][s] = feltor::detail::make_profile( grid, mag, js[field]["profile"], nbg);
                dg::x::HVec damping = feltor::detail::make_damping( grid, unmod_mag, js[field]["damping"]);
                dg::blas1::subroutine( [nbg]( double& profile, double& ntilde, double
                            damping) {
                            ntilde  = (profile+ntilde-nbg)*damping+nbg;
                            profile = (profile-nbg)*damping +nbg;
                        },
                        profile[u][s], source[u][s], damping);
            }
            else if ( "density" == field and "fixed_profile-quasineutral" == type)
            {
                source_rate[u][s] = js[field].get( "rate", 1e-3).asDouble();
                fixed_profile[u][s] = true;
                num_quasineutral_n ++ ;
                idx_quasineutral_n = s;
                source[u][s] = feltor::detail::make_damping( grid, unmod_mag, js[field]["damping"]);
            }
            else if ( "density" == field and "influx-quasineutral" == type)
            {
                source_rate[u][s] = js[field].get( "rate", 1e-3).asDouble();
                num_quasineutral_n ++ ;
                idx_quasineutral_n = s;
            }
            //else if( "torpex" == type)
            //{
            //    double rhosinm = 0.98 / mag.R0();
            //    double rhosinm2 = rhosinm*rhosinm;
            //    source[u][s] = dg::pullback( feltor::detail::TorpexSource(
            //        0.98/rhosinm, -0.02/rhosinm, 0.0335/rhosinm, 0.05/rhosinm, 565*rhosinm2),
            //        grid);
            //}
            //else if( "tcv" == type)
            //{
            //    fixed_profile = false;
            //    const double R_0 = 1075., Z_0 = -10.;
            //    const double psip0 = mag.psip()( R_0, Z_0);
            //    const double sigma = 9.3e-3*psip0/0.4;
            //    source = dg::pullback(
            //        dg::compose( dg::GaussianX( psip0, sigma, 1.),  mag.psip() ), grid);
            //    dg::blas1::pointwiseDot( feltor::detail::xpoint_damping(grid,mag),
            //           source, source);
            //}
            else
                throw dg::Error(dg::Message()<< "Invalid source type "<<type<<"\n");
            if( fixed_profile[u][s] != fixed_profile[0][0])
                throw dg::Error(dg::Message()<< "Profile types cannot be mixed among species or equations!");
            if( source_rate[0][s] != source_rate[0][0])
                throw dg::Error(dg::Message()<< "Density source rates must be equal among species "<<source_rate[0][s]<<" vs "<<source_rate[0][0]<<"!\n");
        }
        // Check quasineutrality
        if( num_quasineutral_n > 1)
            throw std::runtime_error( "There cannot be more than one quasineutral density");
        if( num_quasineutral_n == 0)
            ;//TODO Check that species indeed add up to 0
        else
        {
            for( unsigned s=0; s<num_species; s++)
            {
                if( s == idx_quasineutral_n) continue;
                double zs = jsspecies[s]["z"].asDouble();
                double zidx = jsspecies[idx_quasineutral_n]["z"].asDouble();
                dg::blas1::axpby( -zs/ zidx, profile[0][s], 1.,
                    profile[0][idx_quasineutral_n]);
                if( not fixed_profile[0][0])
                {
                    dg::blas1::axpby( -zs/ zidx, source[0][s], 1.,
                        source[0][idx_quasineutral_n]);
                }
            }
        }
    }
    return source;
};

} //namespace thermal
