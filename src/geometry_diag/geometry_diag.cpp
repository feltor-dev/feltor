#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <functional>
#include <sstream>
#include <ctime>
#include <cmath>

#include "dg/algorithm.h"
#include "dg/file/file.h"
#include "dg/geometries/geometries.h"


int main( int argc, char* argv[])
{
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    std::string inputfile = argc==1 ? "geometry_diag.json" : argv[1];
    js.asJson() = dg::file::file2Json( inputfile);

    std::string geometry_params = js["magnetic_field"]["input"].asString();
    if( geometry_params == "file")
    {
        std::string path = js["magnetic_field"]["file"].asString();
        js.asJson()["magnetic_field"]["file"] = dg::file::file2Json( path,
                dg::file::comments::are_discarded);
    }
    //Test coefficients
    dg::geo::CylindricalFunctor wall, transition, sheath;
    dg::geo::TokamakMagneticField mag = dg::geo::createMagneticField(
            js["magnetic_field"][geometry_params]);
    dg::geo::TokamakMagneticField mod_mag =
        dg::geo::createModifiedField(js["magnetic_field"][geometry_params],
                js["boundary"]["wall"], wall, transition);
    unsigned n = js["grid"].get("n",3).asUInt();
    unsigned Nx = js["grid"].get("Nx",100).asUInt();
    unsigned Ny = js["grid"].get("Ny",100).asUInt();
    unsigned Nz = js["grid"].get("Nz", 1).asUInt();
    double boxscaleRm = js["grid"]["scaleR"].get(0u, 1.1).asDouble();
    double boxscaleRp = js["grid"]["scaleR"].get(1u, 1.1).asDouble();
    double boxscaleZm = js["grid"]["scaleZ"].get(0u, 1.2).asDouble();
    double boxscaleZp = js["grid"]["scaleZ"].get(1u, 1.1).asDouble();
    double Rmin=mag.R0()-boxscaleRm*mag.params().a();
    double Zmin=-boxscaleZm*mag.params().a();
    double Rmax=mag.R0()+boxscaleRp*mag.params().a();
    double Zmax=boxscaleZp*mag.params().a();
    dg::Grid2d sheath_walls( Rmin, Rmax, Zmin, Zmax, 1,1,1);
    bool compute_fsa = false, compute_sheath = false, compute_q = false;
    for( unsigned i=0; i<js["diagnostics"].size(); i++)
    {
        std::string flag = js["diagnostics"].get(i,"fsa").asString();
        if( flag  == "fsa")
            compute_fsa = true;
        else if( flag == "sheath" )
            compute_sheath = true;
        else if( flag == "q-profile" )
            compute_q = true;
        else
            throw std::runtime_error( "diagnostics "+flag+" not recognized!\n");
    }
    if( compute_sheath)
        dg::geo::createSheathRegion( js["boundary"]["sheath"],
            mag, wall, sheath_walls, sheath);

    dg::geo::description mag_description = mag.params().getDescription();

    dg::Grid2d grid2d(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny);
    dg::DVec psipog2d   = dg::evaluate( mag.psip(), grid2d);
    double RO = mag.R0(), ZO = 0.;
    double psipO = dg::blas1::reduce( psipog2d, +1e308, thrust::minimum<double>());
    double psipmax = dg::blas1::reduce( psipog2d, -1e308, thrust::maximum<double>());
    if ( psipmax == psipO) // toroidal field
        psipmax += 1;
    // find O-point
    if( mag_description == dg::geo::description::standardX ||
        mag_description == dg::geo::description::standardO ||
        mag_description == dg::geo::description::square ||
        mag_description == dg::geo::description::doubleX
        )
    {
        try
        {
            int point = dg::geo::findOpoint( mag.get_psip(), RO, ZO);
            psipO = mag.psip()( RO, ZO);
            std::cout << "O-point found at "<<RO<<" "<<ZO
                      <<" with Psip "<<psipO<<std::endl;
            if( point == 1 )
                std::cout << " (minimum)"<<std::endl;
            if( point == 2 )
                std::cout << " (maximum)"<<std::endl;
            double psip0 = mag.psip()(mag.R0(), 0);
            std::cout << "psip( R_0, 0) = "<<psip0<<"\n";
            double fx_0 = 0.125; // must evenly divide Npsi
            psipmax = -fx_0/(1.-fx_0)*psipO;
        } catch( dg::Error& e) { std::cerr << e.what()<<"\n"; }
    }
    double RX = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
    double ZX = -1.1*mag.params().elongation()*mag.params().a();
    double Z2X = Zmax, R2X = RX;
    // find X-point
    if( mag_description == dg::geo::description::standardX
        || mag_description == dg::geo::description::doubleX)
    {
        try{
            dg::geo::findXpoint( mag.get_psip(), RX, ZX);
            std::cout << "X-point found at "<<RX<<" "<<ZX<<std::endl;
            if( mag_description == dg::geo::description::doubleX)
            {
                R2X = RX;
                Z2X = -ZX;
                dg::geo::findXpoint( mag.get_psip(), R2X, Z2X);
                std::cout << "2nd X-point found at "<<R2X<<" "<<Z2X<<std::endl;
            }
        } catch( dg::Error& e) { std::cerr << e.what()<<"\n"; }
    }
    double width_factor = js.get("width-factor",1.0).asDouble();
    dg::geo::FluxSurfaceIntegral<dg::HVec> fsi( grid2d, mag, width_factor);
    double deltaPsi = fsi.get_deltapsi();
    if( deltaPsi < 1e-14) // protect against toroidal
        deltaPsi = 0.1;

    std::vector<std::tuple<std::string, dg::HVec, std::string, double> > map1d;
    //Generate list of functions to evaluate
    std::vector< std::tuple<std::string, std::string, dg::geo::CylindricalFunctor >> map{
        {"Psip", "Flux function", mag.psip()},
        {"modPsip", "Modified Flux function", mod_mag.psip()},
        {"PsipR", "Flux function derivative in R", mag.psipR()},
        {"PsipZ", "Flux function derivative in Z", mag.psipZ()},
        {"PsipRR", "Flux function derivative in RR", mag.psipRR()},
        {"PsipRZ", "Flux function derivative in RZ", mag.psipRZ()},
        {"PsipZZ", "Flux function derivative in ZZ", mag.psipZZ()},
        {"Ipol", "Poloidal current", mag.ipol()},
        {"IpolR", "Poloidal current derivative in R", mag.ipolR()},
        {"IpolZ", "Poloidal current derivative in Z", mag.ipolZ()},
        {"Rho_p", "Normalized Poloidal flux label", dg::geo::RhoP(mag)},
        {"LaplacePsip", "Laplace of flux function", dg::geo::LaplacePsip(mag)},
        {"Bmodule", "Magnetic field strength", dg::geo::Bmodule(mag)},
        {"InvB", "Inverse of Bmodule", dg::geo::InvB(mag)},
        {"LnB", "Natural logarithm of Bmodule", dg::geo::LnB(mag)},
        {"GradLnB", "The parallel derivative of LnB", dg::geo::GradLnB(mag)},
        {"Divb", "The divergence of the magnetic unit vector", dg::geo::Divb(mag)},
        {"B_R", "Derivative of Bmodule in R", dg::geo::BR(mag)},
        {"B_Z", "Derivative of Bmodule in Z", dg::geo::BZ(mag)},
        {"CurvatureNablaBR",  "R-component of the (toroidal) Nabla B curvature vector", dg::geo::CurvatureNablaBR(mag,+1)},
        {"CurvatureNablaBZ",  "Z-component of the (toroidal) Nabla B curvature vector", dg::geo::CurvatureNablaBZ(mag,+1)},
        {"CurvatureKappaR",   "R-component of the (toroidal) Kappa B curvature vector", dg::geo::CurvatureKappaR(mag,+1)},
        {"CurvatureKappaZ",   "Z-component of the (toroidal) Kappa B curvature vector", dg::geo::CurvatureKappaZ(mag,+1)},
        {"DivCurvatureKappa", "Divergence of the (toroidal) Kappa B curvature vector", dg::geo::DivCurvatureKappa(mag,+1)},
        {"DivCurvatureNablaB","Divergence of the (toroidal) Nabla B curvature vector", dg::geo::DivCurvatureNablaB(mag,+1)},
        {"TrueCurvatureNablaBR", "R-component of the (true) Nabla B curvature vector", dg::geo::TrueCurvatureNablaBR(mag)},
        {"TrueCurvatureNablaBZ", "Z-component of the (true) Nabla B curvature vector", dg::geo::TrueCurvatureNablaBZ(mag)},
        {"TrueCurvatureNablaBP", "Contravariant Phi-component of the (true) Nabla B curvature vector", dg::geo::TrueCurvatureNablaBP(mag)},
        {"TrueCurvatureKappaR", "R-component of the (true) Kappa B curvature vector", dg::geo::TrueCurvatureKappaR(mag)},
        {"TrueCurvatureKappaZ", "Z-component of the (true) Kappa B curvature vector", dg::geo::TrueCurvatureKappaZ(mag)},
        {"TrueCurvatureKappaP", "Contravariant Phi-component of the (true) Kappa B curvature vector", dg::geo::TrueCurvatureKappaP(mag)},
        {"TrueDivCurvatureKappa", "Divergence of the (true) Kappa B curvature vector", dg::geo::TrueDivCurvatureKappa(mag)},
        {"TrueDivCurvatureNablaB","Divergence of the (true) Nabla B curvature vector",  dg::geo::TrueDivCurvatureNablaB(mag)},
        {"BFieldR", "R-component of the magnetic field vector", dg::geo::BFieldR(mag)},
        {"BFieldZ", "Z-component of the magnetic field vector", dg::geo::BFieldZ(mag)},
        {"BFieldP", "Contravariant Phi-component of the magnetic field vector", dg::geo::BFieldP(mag)},
        {"BHatR", "R-component of the magnetic field unit vector", dg::geo::BHatR(mag)},
        {"BHatZ", "Z-component of the magnetic field unit vector", dg::geo::BHatZ(mag)},
        {"BHatP", "Contravariant Phi-component of the magnetic field unit vector", dg::geo::BHatP(mag)},
        {"BHatRR", "R derivative of BHatR", dg::geo::BHatRR(mag)},
        {"BHatZR", "R derivative of BHatZ", dg::geo::BHatZR(mag)},
        {"BHatPR", "R derivative of BHatP", dg::geo::BHatPR(mag)},
        {"BHatRZ", "Z derivative of BHatR", dg::geo::BHatRZ(mag)},
        {"BHatZZ", "Z derivative of BHatZ", dg::geo::BHatZZ(mag)},
        {"BHatPZ", "Z derivative of BHatP", dg::geo::BHatPZ(mag)},
        {"NormGradPsip", "Norm of gradient of Psip", dg::geo::SquareNorm( dg::geo::createGradPsip(mag), dg::geo::createGradPsip(mag))},
        {"SquareGradPsip", "Norm of gradient of Psip", dg::geo::ScalarProduct( dg::geo::createGradPsip(mag), dg::geo::createGradPsip(mag))},
        {"CurvatureNablaBGradPsip", "(Toroidal) Nabla B curvature dot the gradient of Psip", dg::geo::ScalarProduct( dg::geo::createCurvatureNablaB(mag, +1), dg::geo::createGradPsip(mag))},
        {"CurvatureKappaGradPsip", "(Toroidal) Kappa curvature dot the gradient of Psip", dg::geo::ScalarProduct( dg::geo::createCurvatureKappa(mag, +1), dg::geo::createGradPsip(mag))},
        {"TrueCurvatureNablaBGradPsip", "True Nabla B curvature dot the gradient of Psip", dg::geo::ScalarProduct( dg::geo::createTrueCurvatureNablaB(mag), dg::geo::createGradPsip(mag))},
        {"TrueCurvatureKappaGradPsip", "True Kappa curvature dot the gradient of Psip", dg::geo::ScalarProduct( dg::geo::createTrueCurvatureKappa(mag), dg::geo::createGradPsip(mag))},
        //////////////////////////////////
        {"Iris", "A flux aligned Iris", dg::compose( dg::Iris( 0.5, 0.7), dg::geo::RhoP(mag))},
        {"Pupil", "A flux aligned Pupil", dg::compose( dg::Pupil(0.7), dg::geo::RhoP(mag)) },
        {"PsiLimiter", "A flux aligned Heaviside", dg::compose( dg::Heaviside( 1.03), dg::geo::RhoP(mag) )},
        {"MagneticTransition", "The region where the magnetic field is modified", transition},
        {"Delta", "A flux aligned Gaussian peak", dg::compose( dg::GaussianX( psipO*0.2, deltaPsi, 1./(sqrt(2.*M_PI)*deltaPsi)), mag.psip())},
        {"ClosedFieldlineRegion", "Region of closed fieldlines", dg::compose( [](bool x){ return double(x);}, dg::geo::mod::ClosedFieldlineRegion( mag) )},
        {"SOL", "The scrape off layer region", dg::compose( [](bool x){ return double(x);}, dg::geo::mod::SOLRegion( mag, wall) )},
        ////
        { "Hoo", "The novel h02 factor", dg::geo::Hoo( mag) },
        {"Wall", "Penalization region that acts as the wall", wall },
        {"WallDistance", "Distance to closest wall", dg::geo::CylindricalFunctor( dg::WallDistance( sheath_walls)) }
    };
    double maxPhi = 0;
    if ( compute_sheath)
        maxPhi = 2.*M_PI*js["boundary"]["sheath"].get("max_angle", 0.1).asDouble();
    std::vector< std::tuple<std::string, std::string, dg::geo::CylindricalFunctor >> sheath_map{
        /////////////////////////////////////
        {"WallFieldlineAnglePDistance", "Distance to wall along fieldline",
            dg::geo::WallFieldlineDistance( dg::geo::createBHat(mod_mag),
                    sheath_walls, maxPhi, 1e-6, "phi", dg::geo::mod::SOLRegion( mag, wall)) },
        {"WallFieldlineAngleMDistance", "Distance to wall along fieldline",
            dg::geo::WallFieldlineDistance( dg::geo::createBHat(mod_mag),
                    sheath_walls, -maxPhi, 1e-6, "phi", dg::geo::mod::SOLRegion( mag, wall)) },
        {"WallFieldlineSPDistance", "Distance to wall along fieldline",
            dg::geo::WallFieldlineDistance( dg::geo::createBHat(mod_mag),
                    sheath_walls, maxPhi, 1e-6, "s", dg::geo::mod::SOLRegion( mag, wall)) },
        {"WallFieldlineSMDistance", "Distance to wall along fieldline",
            dg::geo::WallFieldlineDistance( dg::geo::createBHat(mod_mag),
                    sheath_walls, -maxPhi, 1e-6, "s", dg::geo::mod::SOLRegion( mag, wall)) },
        {"Sheath", "Sheath region", sheath},
        {"SheathDirection", "Direction of magnetic field relative to sheath", dg::geo::WallDirection(mag, sheath_walls) },
        {"SheathCoordinate", "Coordinate from -1 to 1 of magnetic field relative to sheath", dg::geo::WallFieldlineCoordinate( dg::geo::createBHat( mod_mag), sheath_walls, maxPhi, 1e-4, "s")}
    };

    ///////////TEST CURVILINEAR GRID TO COMPUTE FSA QUANTITIES
    unsigned npsi = js["grid"].get("npsi", 3).asUInt();
    //set number of psivalues (NPsi % 8 == 0)
    unsigned Npsi = js["grid"].get("Npsi", 32).asUInt();
    unsigned Neta = js["grid"].get("Neta", 640).asUInt();
    /// -------  Elements for fsa on X-point grid ----------------
    std::unique_ptr<dg::geo::CurvilinearGrid2d> gX2d;
    dg::direction integration_dir = psipO<psipmax ? dg::forward : dg::backward;
    dg::Timer t;
    if( compute_fsa &&
        (
        mag_description == dg::geo::description::standardX ||
        mag_description == dg::geo::description::standardO ||
        mag_description == dg::geo::description::square ||
        mag_description == dg::geo::description::doubleX
        )
        )
    {
        t.tic();
        std::cout << "Generate orthogonal flux-aligned grid ... \n";
        double fx_0 = js["grid"].get( "fx_0", 1./8.).asDouble(); // must evenly divide Npsi
        psipmax = -fx_0/(1.-fx_0)*psipO;
        std::cout << "psi 1 is          "<<psipmax<<"\n";
        // this one is actually slightly better than the X-point grid
        dg::geo::SimpleOrthogonal generator(mag.get_psip(),
                psipO<psipmax ? psipO : psipmax,
                psipO<psipmax ? psipmax : psipO,
                mag.R0() + 0.1*mag.params().a(), 0., 0.1*psipO, 1);
        gX2d = std::make_unique<dg::geo::CurvilinearGrid2d>(generator,
                npsi, Npsi, Neta, dg::DIR, dg::PER);
        t.toc();
        map1d.emplace_back( "gridX", dg::HVec(npsi*Npsi,0.0), "Time to generated X-grid", t.diff());
        std::cout << "DONE! \n";
        t.tic();
        dg::Average<dg::IHMatrix, dg::HVec > avg_eta( *gX2d, dg::coo2d::y);
        std::vector<dg::HVec> coordsX = gX2d->map();
        dg::SparseTensor<dg::HVec> metricX = gX2d->metric();
        dg::HVec volX2d = dg::tensor::volume2d( metricX);
        dg::blas1::pointwiseDot( coordsX[0], volX2d, volX2d); //R\sqrt{g}

        // f0 makes a - sign if psipmax < psipO
        const double f0 = (gX2d->x1()-gX2d->x0())/ ( psipmax - psipO);
        dg::HVec dvdpsip;
        avg_eta( volX2d, dvdpsip, false);
        dg::blas1::scal( dvdpsip, 4.*M_PI*M_PI*f0);
        dg::Grid1d gX1d(psipO<psipmax ? psipO : psipmax,
                        psipO<psipmax ? psipmax : psipO,
                        npsi, Npsi, psipO < psipmax ? dg::DIR_NEU : dg::NEU_DIR);
        //O-point fsa value is always zero
        t.toc();
        map1d.emplace_back( "dvdpsip", dvdpsip,
            "Derivative of flux volume with respect to flux label psi", t.diff());
        t.tic();
        dg::HVec X_psi_vol = dg::integrate( dvdpsip, gX1d, integration_dir);
        t.toc();
        map1d.emplace_back( "psi_vol", X_psi_vol,
            "Flux volume on X-point grid", t.diff());
        t.tic();
        dg::HMatrix dpsi = dg::create::dx( gX1d, dg::NEU, dg::backward); //we need to avoid involving cells outside LCFS in computation (also avoids right boundary)
        if( psipO > psipmax)
            dpsi = dg::create::dx( gX1d, dg::NEU, dg::forward);

        //NOTE: VOLUME is WITHIN cells while AREA is ON gridpoints
        dg::HVec gradZetaX = metricX.value(0,0), X_psi_area;
        dg::blas1::transform( gradZetaX, gradZetaX, dg::SQRT<double>());
        dg::blas1::pointwiseDot( volX2d, gradZetaX, gradZetaX); //R\sqrt{g}|\nabla\zeta|
        avg_eta( gradZetaX, X_psi_area, false);
        dg::blas1::scal( X_psi_area, 4.*M_PI*M_PI);
        t.toc();
        map1d.emplace_back( "psi_area", X_psi_area,
            "Flux area on X-point grid", t.diff());
        std::cout << "Total volume within separatrix is "
              << dg::interpolate( dg::xspace, X_psi_vol, 0., gX1d)<<std::endl;

        //Compute FSA of cylindrical functions
        dg::HVec transferH, transferH1d, integral1d;
        for( auto tp : map)
        {
            if( std::get<0>(tp).find("Wall") != std::string::npos)
                continue;
            if( std::get<0>(tp).find("Sheath") != std::string::npos)
                continue;
            transferH = dg::pullback( std::get<2>(tp), *gX2d);
            t.tic();
            dg::blas1::pointwiseDot( volX2d, transferH, transferH);
            avg_eta( transferH, transferH1d, false);
            dg::blas1::scal( transferH1d, 4*M_PI*M_PI*f0); //
            dg::blas1::pointwiseDivide( transferH1d, dvdpsip, transferH1d );
            t.toc();
            map1d.emplace_back( std::get<0>(tp)+"_fsa", transferH1d,
                std::get<1>(tp)+" (Flux surface average)", t.diff());
            t.tic();
            dg::blas1::pointwiseDot( transferH1d, dvdpsip, transferH1d );
            integral1d = dg::integrate( transferH1d, gX1d, integration_dir);
            t.toc();
            map1d.emplace_back( std::get<0>(tp)+"_ifs", integral1d,
                std::get<1>(tp)+" (Flux surface integral)", t.diff());
            t.tic();
            dg::blas2::symv( dpsi, transferH1d, integral1d);
            dg::blas1::pointwiseDivide( integral1d, dvdpsip, integral1d);
            t.toc();
            map1d.emplace_back( std::get<0>(tp)+"_dfs", integral1d,
                std::get<1>(tp)+" (Flux current derivative)", t.diff());
        }
    }
    /// --------- More flux labels --------------------------------

    dg::Grid1d grid1d(psipO<psipmax ? psipO : psipmax,
            psipO<psipmax ? psipmax : psipO, npsi, Npsi,dg::DIR_NEU); //inner value is always zero
    if( compute_q &&
        (
        mag_description == dg::geo::description::standardX ||
        mag_description == dg::geo::description::standardO ||
        mag_description == dg::geo::description::square ||
        mag_description == dg::geo::description::doubleX
        )
        )
    {
        t.tic();
        dg::HVec rho = dg::evaluate( dg::cooX1d, grid1d);
        dg::blas1::axpby( -1./psipO, rho, +1., 1., rho); //transform psi to rho
        t.toc();
        map1d.emplace_back("rho", rho,
            "Alternative flux label rho = -psi/psimin + 1", t.diff());
        t.tic();
        dg::blas1::transform( rho, rho, dg::SQRT<double>());
        t.toc();
        map1d.emplace_back("rho_p", rho,
            "Alternative flux label rho_p = Sqrt[-psi/psimin + 1]", t.diff());
        t.tic();
        dg::geo::SafetyFactor qprof( mag);
        dg::HVec psi_vals = dg::evaluate( dg::cooX1d, grid1d);
        // we need to avoid calling SafetyFactor outside closed fieldlines
        dg::blas1::subroutine( [psipO]( double& psi){
               if( (psipO < 0 && psi > 0) || (psipO>0 && psi <0))
                   psi = psipO/2.; // just use a random value
            }, psi_vals);
        dg::HVec qprofile( psi_vals);
        dg::blas1::evaluate( qprofile, dg::equals(), qprof, psi_vals);
        t.toc();
        map1d.emplace_back("q-profile", qprofile,
            "q-profile (Safety factor) using direct integration", t.diff());
        if( mag_description == dg::geo::description::standardX
            || mag_description == dg::geo::description::doubleX)
        {
            try{
                dg::Grid2d grid2d_tmp(Rmin,Rmax,ZX,Z2X, n,Nx,Ny);
                double width_factor = js.get("width-factor",0.03).asDouble();
                dg::geo::SafetyFactorAverage qprof_avg(grid2d_tmp, mag, width_factor);
                dg::HVec qprofile_avg( psi_vals);
                t.tic();
                dg::blas1::evaluate( qprofile_avg, dg::equals(), qprof_avg, psi_vals);
                t.toc();
                map1d.emplace_back("q-profile-avg", qprofile_avg,
                    "q-profile (Safety factor) using average integration", t.diff());
                dg::HVec curvVec = dg::evaluate( dg::geo::ScalarProduct(
                            dg::geo::createCurvatureNablaB(mag, +1),
                            dg::geo::createGradPsip(mag)), grid2d_tmp);
                dg::geo::FluxSurfaceAverage<dg::DVec> fsa_avg( grid2d_tmp, mag,
                        curvVec, dg::evaluate( dg::cooX2d, grid2d_tmp), width_factor);
                t.tic();
                dg::blas1::evaluate( qprofile_avg, dg::equals(), fsa_avg, psi_vals);
                t.toc();
                map1d.emplace_back("CurvatureNablaBGradPsip_fsa-avg", qprofile_avg,
                    "using average integration", t.diff());
                curvVec = dg::evaluate( dg::geo::ScalarProduct(
                            dg::geo::createCurvatureKappa(mag, +1),
                            dg::geo::createGradPsip(mag)), grid2d_tmp);
                fsa_avg.set_container( curvVec);
                t.tic();
                dg::blas1::evaluate( qprofile_avg, dg::equals(), fsa_avg, psi_vals);
                t.toc();
                map1d.emplace_back("CurvatureKappaGradPsip_fsa-avg", qprofile_avg,
                    "using average integration", t.diff());


            } catch( dg::Error& e) { std::cerr << e.what()<<"\n"; }
        }
        t.tic();
        dg::HVec psit = dg::integrate( qprofile, grid1d, integration_dir);
        t.toc();
        map1d.emplace_back("psit1d", psit,
            "Toroidal flux label psi_t integrated  on grid1d using direct q", t.diff());
        //we need to avoid integrating outside closed fieldlines
        t.tic();
        dg::Grid1d g1d_fine(psipO<0. ? psipO : 0.,
                psipO<0. ? 0. : psipO, npsi, Npsi,dg::NEU);
        qprofile = dg::evaluate( qprof, g1d_fine);
        dg::HVec w1d = dg::create::weights( g1d_fine);
        double psit_tot = dg::blas1::dot( w1d, qprofile);
        if( integration_dir == dg::backward)
            psit_tot*=-1;
        //std::cout << "psit tot "<<psit_tot<<"\n";
        dg::blas1::scal ( psit, 1./psit_tot);
        dg::blas1::transform( psit, psit, dg::SQRT<double>());
        t.toc();
        map1d.emplace_back("rho_t", psit,
            "Toroidal flux label rho_t = sqrt( psit/psit_tot) evaluated on grid1d", t.diff());
    }

    /////////////////////////////set up netcdf/////////////////////////////////////
    std::cout << "CREATING/OPENING FILE ... \n";
    std::string newfilename = argc<3 ? "geometry_diag.nc" : argv[2];
    dg::file::NcFile file( newfilename, dg::file::nc_clobber);
    /// Set global attributes
    std::map<std::string, dg::file::nc_att_t> g_atts;
    g_atts["title"] = "Output file of feltor/src/geometry_diag/geometry_diag.cpp";
    g_atts["Conventions"] = "CF-1.7";
    g_atts["history"] = dg::file::timestamp( argc, argv);
    g_atts["comment"] = "Find more info in feltor/src/geometry_diag/geometry_diag.tex";
    g_atts["source"] = "FELTOR";
    g_atts["references"] = "https://github.com/feltor-dev/feltor";
    g_atts["inputfile"] = js.toStyledString();
    file.put_atts( g_atts);
    file.put_atts( dg::file::version_flags);

    if( mag_description == dg::geo::description::standardX ||
        mag_description == dg::geo::description::standardO ||
        mag_description == dg::geo::description::square ||
        mag_description == dg::geo::description::doubleX
        )
    {
        file.put_att( {"opoint", std::vector{RO, ZO}});
        if( mag_description == dg::geo::description::standardX)
            file.put_att( {"xpoint", std::vector{RX, ZX}});
        if( mag_description == dg::geo::description::doubleX)
            file.put_att( {"xpoint", std::vector{RX, ZX, R2X, Z2X}});
    }


    if( compute_fsa &&
        (
        mag_description == dg::geo::description::standardX ||
        mag_description == dg::geo::description::standardO ||
        mag_description == dg::geo::description::square ||
        mag_description == dg::geo::description::doubleX
        )
        )
    {
        file.defput_dim( "zeta", {{"axis", "X"},
            {"long_name", "Radial coordinate in X grid"}},
            gX2d->abscissas(0));
        file.defput_dim( "eta", {{"axis", "Y"},
            {"long_name", "Poloidal coordinate in X grid"}},
            gX2d->abscissas(1));
        file.defput_var( "xcc", {"eta", "zeta"}, {{"long_name",
            "Cartesian x-coordinate"}},*gX2d, gX2d->map()[0]);
        file.defput_var( "ycc", {"eta", "zeta"}, {{"long_name",
            "Cartesian y-coordinate"}},*gX2d, gX2d->map()[1]);
    }
    else
        file.defput_dim( "zeta", {{"axis", "X"},
            {"long_name", "Flux surface label"}},
            grid1d.abscissas(0));

    dg::CylindricalGrid3d grid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, n,Nx,Ny,Nz);

    //write 1d vectors
    std::cout << "WRTING 1D FIELDS ... \n";
    for( auto tp : map1d)
    {
        file.defput_var(std::get<0>(tp), {"zeta"}, {{"long_name",
            std::get<2>(tp)}, {"time", std::get<3>(tp)}}, grid1d, std::get<1>(tp));
    }
    //write 2d vectors
    //allocate mem for visual
    dg::HVec hvisual = dg::evaluate( dg::zero, grid2d);
    dg::HVec hvisual3d = dg::evaluate( dg::zero, grid3d);
    dg::fHVec fvisual, fvisual3d;
    std::cout << "WRTING 2D/3D CYLINDRICAL FIELDS ... \n";
    dg::RealCylindricalGrid3d<float> fgrid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, n,Nx,Ny,Nz);
    file.defput_dim( "R", {{"axis", "X"}, {"long_name", "R coordinate in Cylindrical system"}},
        fgrid3d.abscissas(0));
    file.defput_dim( "Z", {{"axis", "Y"}, {"long_name", "Z coordinate in Cylindrical system"}},
        fgrid3d.abscissas(1));
    file.defput_dim( "P", {{"axis", "Z"}, {"long_name", "Phi coordinate in Cylindrical system"}},
        fgrid3d.abscissas(2));
    dg::RealGrid2d<float> fgrid2d( Rmin, Rmax, Zmin,Zmax, n, Nx, Ny);
    auto prolong = dg::create::prolongation( grid3d, std::array{2u});
    for(auto tp : map)
    {
        hvisual = dg::evaluate( std::get<2>(tp), grid2d);
        dg::apply( prolong, hvisual, hvisual3d);
        dg::assign( hvisual, fvisual);
        dg::assign( hvisual3d, fvisual3d);

        std::map<std::string, dg::file::nc_att_t> atts;
        atts["long_name"] = std::get<1>(tp);
        file.defput_var( std::get<0>(tp), {"Z", "R"}, atts,
            *fgrid3d.perp_grid(), fvisual);
        atts["coordinates"] = "zc yc xc";
        file.defput_var( std::get<0>(tp)+"3d", {"P", "Z", "R"}, atts, fgrid3d,
            fvisual3d);
    }
    if( compute_sheath)
    {
        for(auto tp : sheath_map)
        {
            dg::Timer t;
            t.tic();
            hvisual = dg::evaluate( std::get<2>(tp), grid2d);
            t.toc();
            if((    std::get<0>(tp).find("Wall") != std::string::npos)
                ||( std::get<0>(tp).find("Sheath") != std::string::npos))
                std::cout<< std::get<0>(tp) << " took "<<t.diff()<<"s\n";
            auto prolong = dg::create::prolongation( grid3d, std::array{2u});
            dg::apply( prolong, hvisual, hvisual3d);
            dg::assign( hvisual, fvisual);
            dg::assign( hvisual3d, fvisual3d);

            std::map<std::string, dg::file::nc_att_t> atts;
            atts["long_name"] = std::get<1>(tp);
            file.defput_var( std::get<0>(tp), {"Z", "R"}, atts, *fgrid3d.perp_grid(), fvisual);
            atts["coordinates"] = "zc yc xc";
            file.defput_var( std::get<0>(tp)+"3d", {"P", "Z", "R"}, atts, fgrid3d, fvisual3d);
        }
    }
    std::cout << "WRTING 3D FIELDS ... \n";
    //compute & write 3d vectors
    std::vector< std::tuple<std::string, std::string, std::function< double(double,double,double)> > > map3d{
        {"BR", "R-component of the magnetic field vector (3d version of BFieldR)",
            dg::geo::BFieldR(mag)},
        {"BZ", "Z-component of the magnetic field vector (3d version of BFieldZ)",
            dg::geo::BFieldZ(mag)},
        {"BP", "Contravariant Phi-component of the magnetic field vector (3d version of BFieldP)",
            dg::geo::BFieldP(mag)},
        {"xc", "x-coordinate in Cartesian coordinate system", dg::cooRZP2X},
        {"yc", "y-coordinate in Cartesian coordinate system", dg::cooRZP2Y},
        {"zc", "z-coordinate in Cartesian coordinate system", dg::cooRZP2Z}
    };
    for( auto tp : map3d)
    {
        std::map<std::string, dg::file::nc_att_t> atts;
        atts["long_name"] = std::get<1>(tp);
        if( std::get<1>(tp) != "xc" && std::get<1>(tp) != "yc" &&std::get<1>(tp) != "zc")
        {
            atts["coordinates"] = "zc yc xc";
        }
        hvisual3d = dg::evaluate( std::get<2>(tp), grid3d);
        dg::assign( hvisual3d, fvisual3d);
        file.defput_var( std::get<0>(tp), {"P", "Z", "R"}, atts, fgrid3d, fvisual3d);
    }
    //////////////////////////////Finalize////////////////////////////////////
    file.close();
    return 0;
}
