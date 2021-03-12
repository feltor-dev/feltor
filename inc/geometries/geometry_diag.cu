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

#include "make_field.h"
#include "magnetic_field.h"
#include "testfunctors.h"
#include "curvilinearX.h"
#include "separatrix_orthogonal.h"
#include "average.h"


// - write magnetic functions into file
// - compute Flux - surface averages and write into file
struct Parameters
{
    unsigned n, Nx, Ny, Nz, Npsi;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;
    double amp, k_psi, nprofileamp;
    double sigma, posX, posY;
    double source_alpha, source_boundary;
    double profile_alpha;
    Parameters( const dg::file::WrappedJsonValue& js){
        n = js.get("n",3).asUInt();
        Nx = js.get("Nx",100).asUInt()/js["compression"].get(0u,1).asUInt();
        Ny = js.get("Ny",100).asUInt()/js["compression"].get(1u,1).asUInt();
        Nz = js.get("Nz", 1).asUInt();
        Npsi = js.get("Npsi", 32).asUInt();
        boxscaleRm = js["box"]["scaleR"].get(0u, 1.1).asDouble();
        boxscaleRp = js["box"]["scaleR"].get(1u, 1.1).asDouble();
        boxscaleZm = js["box"]["scaleZ"].get(0u, 1.2).asDouble();
        boxscaleZp = js["box"]["scaleZ"].get(1u, 1.1).asDouble();
        amp = js.get("amplitude", 1.).asDouble();
        k_psi = js.get("k_psi", 1.).asDouble();
        nprofileamp = js["profile"].get("amp", 1.).asDouble();
        profile_alpha = js["profile"].get("alpha", 0.1).asDouble();
        sigma = js.get("sigma", 10).asDouble();
        posX = js.get("posX", 0.5).asDouble();
        posY = js.get("posY", 0.5).asDouble();
        source_alpha = js["source"].get("alpha", 0.5).asDouble();
        source_boundary = js["source"].get("boundary", 0.5).asDouble();
    }
};


int main( int argc, char* argv[])
{
    std::string newfilename;
    Json::Value input_js, geom_js;
    if( argc == 4)
    {
        newfilename = argv[3];
        std::cout << argv[0]<< " "<<argv[1]<<" & "<<argv[2]<<" -> " <<argv[3]<<std::endl;
        dg::file::file2Json( argv[1], input_js, dg::file::comments::are_discarded);
        dg::file::file2Json( argv[2], geom_js, dg::file::comments::are_discarded);
    }
    else if( argc == 3)
    {
        newfilename = argv[2];
        std::cout << argv[0]<< " "<<argv[1]<<" -> " <<argv[2]<<std::endl;
        dg::file::NC_Error_Handle err;
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
        dg::file::string2Json(inputfile, input_js, dg::file::comments::are_discarded);
        dg::file::string2Json(geomfile, geom_js, dg::file::comments::are_discarded);
    }
    else
    {
        std::cerr << "ERROR: Wrong number of arguments!\n";
        std::cerr << " Usage: "<< argv[0]<<" [input.json] [geom.json] [output.nc]\n";
        std::cerr << " ( Minimum input json file is { \"n\" : 3, \"Nx\": 100, \"Ny\":100 })\n";
        std::cerr << "Or \n Usage: "<< argv[0]<<" [file.nc] [output.nc]\n";
        std::cerr << " ( Program searches for string variables 'inputfile' and 'geomfile' in file.nc and tries a json parser)\n";
        return -1;
    }
    std::cout << input_js<<std::endl;
    dg::file::WrappedJsonValue ws ( input_js, dg::file::error::is_warning);
    const Parameters p(ws);
    //Test coefficients
    dg::geo::CylindricalFunctor wall, transition, sheath, direction;
    dg::geo::TokamakMagneticField mag = dg::geo::createMagneticField(geom_js);
    dg::geo::TokamakMagneticField mod_mag =
        dg::geo::createModifiedField(geom_js, ws["wall"], wall, transition);
    std::string input = input_js.toStyledString();
    std::string geom = geom_js.toStyledString();
    unsigned n, Nx, Ny, Nz;
    n = p.n, Nx = p.Nx, Ny = p.Ny, Nz = p.Nz;
    double Rmin=mag.R0()-p.boxscaleRm*mag.params().a();
    double Zmin=-p.boxscaleZm*mag.params().a()*mag.params().elongation();
    double Rmax=mag.R0()+p.boxscaleRp*mag.params().a();
    double Zmax=p.boxscaleZp*mag.params().a()*mag.params().elongation();
    ws.set_mode( dg::file::error::is_warning);
    dg::geo::createSheathRegion( ws["sheath"],
            mag, wall, Rmin, Rmax, Zmin, Zmax, sheath, direction);

    dg::geo::description mag_description = mag.params().getDescription();
    double psipO = -1.;
    if( mag_description == dg::geo::description::standardX || mag_description == dg::geo::description::standardO )
    {
        //Find O-point
        double RO = mag.R0(), ZO = 0.;
        int point = dg::geo::findOpoint( mag.get_psip(), RO, ZO);
        psipO = mag.psip()( RO, ZO);
        std::cout << "O-point found at "<<RO<<" "<<ZO<<" with Psip "<<psipO<<std::endl;
        if( point == 1 )
            std::cout << " (minimum)"<<std::endl;
        if( point == 2 )
            std::cout << " (maximum)"<<std::endl;
        double psip0 = mag.psip()(mag.R0(), 0);
        std::cout << "psip( R_0, 0) = "<<psip0<<"\n";
    }


    dg::Grid2d grid2d(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny);
    dg::DVec psipog2d   = dg::evaluate( mag.psip(), grid2d);
    std::vector<std::tuple<std::string, dg::HVec, std::string> > map1d;
    ///////////TEST CURVILINEAR GRID TO COMPUTE FSA QUANTITIES
    unsigned npsi = 3, Npsi = p.Npsi;//set number of psivalues (NPsi % 8 == 0)
    //Generate list of functions to evaluate
    std::vector< std::tuple<std::string, std::string, dg::geo::CylindricalFunctor >> map{
        {"Psip", "Flux function", mag.psip()},
        {"PsipR", "Flux function derivative in R", mag.psipR()},
        {"PsipZ", "Flux function derivative in Z", mag.psipZ()},
        {"PsipRR", "Flux function derivative in RR", mag.psipRR()},
        {"PsipRZ", "Flux function derivative in RZ", mag.psipRZ()},
        {"PsipZZ", "Flux function derivative in ZZ", mag.psipZZ()},
        {"Ipol", "Poloidal current", mag.ipol()},
        {"IpolR", "Poloidal current derivative in R", mag.ipolR()},
        {"IpolZ", "Poloidal current derivative in Z", mag.ipolZ()},
        {"Rho_p", "Normalized Poloidal flux label", dg::geo::RhoP(mag)},
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
        {"GradBHatR", "Parallel derivative of BHatR", dg::geo::BHatR(mag)},
        {"GradBHatZ", "Parallel derivative of BHatZ", dg::geo::BHatZ(mag)},
        {"GradBHatP", "Parallel derivative of BHatP", dg::geo::BHatP(mag)},
        {"NormGradPsip", "Norm of gradient of Psip", dg::geo::SquareNorm( dg::geo::createGradPsip(mag), dg::geo::createGradPsip(mag))},
        {"CurvatureNablaBGradPsip", "(Toroidal) Nabla B curvature dot the gradient of Psip", dg::geo::ScalarProduct( dg::geo::createCurvatureNablaB(mag, +1), dg::geo::createGradPsip(mag))},
        {"CurvatureKappaGradPsip", "(Toroidal) Kappa curvature dot the gradient of Psip", dg::geo::ScalarProduct( dg::geo::createCurvatureKappa(mag, +1), dg::geo::createGradPsip(mag))},
        {"TrueCurvatureNablaBGradPsip", "True Nabla B curvature dot the gradient of Psip", dg::geo::ScalarProduct( dg::geo::createTrueCurvatureNablaB(mag), dg::geo::createGradPsip(mag))},
        {"TrueCurvatureKappaGradPsip", "True Kappa curvature dot the gradient of Psip", dg::geo::ScalarProduct( dg::geo::createTrueCurvatureKappa(mag), dg::geo::createGradPsip(mag))},
        /////////////////////////////////////
        {"WallDistance", "Distance to closest wall", dg::geo::CylindricalFunctor( dg::WallDistance( {grid2d.x0(), grid2d.x1()}, {grid2d.y0(), grid2d.y1()})) },
        {"WallDirection", "Direction of magnetic field to closest wall", dg::geo::WallDirection(mag, {grid2d.x0(), grid2d.x1()}, {grid2d.y0(), grid2d.y1()}) },
        {"Sheath", "Sheath region", sheath},
        {"SheathDirection", "Direction of magnetic field relative to sheath", direction},
        //////////////////////////////////
        {"Iris", "A flux aligned Iris", dg::compose( dg::Iris( 0.5, 0.7), dg::geo::RhoP(mag))},
        {"Pupil", "A flux aligned Pupil", dg::compose( dg::Pupil(0.7), dg::geo::RhoP(mag)) },
        {"GaussianDamping", "A flux aligned Heaviside with Gaussian damping", dg::compose( dg::GaussianDamping( 0.8, p.source_alpha), dg::geo::RhoP(mag)) },
        {"ZonalFlow",  "Flux aligned Sine function", dg::compose( dg::SinX ( p.amp, 0., 2.*M_PI*p.k_psi ), mag.psip())},
        {"PsiLimiter", "A flux aligned Heaviside", dg::compose( dg::Heaviside( 1.03), dg::geo::RhoP(mag) )},
        {"SourceProfile", "A source profile", dg::compose( dg::PolynomialHeaviside(
                    p.source_boundary-p.source_alpha/2., p.source_alpha/2., -1 ),
                dg::geo::RhoP(mag))},
        {"Wall", "Penalization region that acts as the wall", wall },
        {"MagneticTransition", "The region where the magnetic field is modified", transition},
        {"Nprofile", "A flux aligned profile", dg::compose( dg::LinearX( p.nprofileamp/mag.psip()(mag.R0(),0.), p.nprofileamp ), mag.psip())},
        {"Delta", "A flux aligned Gaussian peak", dg::compose( dg::GaussianX( psipO*0.2, 0.1, 1./(sqrt(2.*M_PI)*0.1)), mag.psip())},
        {"TanhDamping", "A flux aligned Heaviside with Tanh Damping", dg::compose( dg::TanhProfX( -3*p.source_alpha, p.source_alpha, -1), mag.psip())},
        ////
        {"BathRZ", "A randomized field", dg::BathRZ( 16, 16, Rmin,Zmin, 30.,2, p.amp)},
        {"Gaussian3d", "A Gaussian field", dg::Gaussian3d(mag.R0()+p.posX*mag.params().a(), p.posY*mag.params().a(),
            M_PI, p.sigma, p.sigma, p.sigma, p.amp)},
        { "Hoo", "The novel h02 factor", dg::geo::Hoo( mag) }
    };

    /// -------  Elements for fsa on X-point grid ----------------
    double psipmax = dg::blas1::reduce( psipog2d, 0., thrust::maximum<double>()); //DEPENDS ON GRID RESOLUTION!!
    std::unique_ptr<dg::geo::CurvilinearGridX2d> gX2d;
    if( mag_description == dg::geo::description::standardX)
    {
        std::cout << "Generate X-point flux-aligned grid ... \n";
        double RX = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
        double ZX = -1.1*mag.params().elongation()*mag.params().a();
        dg::geo::findXpoint( mag.get_psip(), RX, ZX);
        double psipX = mag.psip()(RX, ZX);
        std::cout << "X-point found at "<<RX<<" "<<ZX<<" with Psip = "<<psipX<<std::endl;
        if( fabs(psipX ) > 1e-10)
        {
            std::cerr << " Psip at X-point is not zero. Unable to construct grid\n";
            return -1;
        }
        dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( mag.get_psip(), RX, ZX) ;
        dg::geo::SeparatrixOrthogonal generator(mag.get_psip(), monitor_chi, psipO, RX, ZX, mag.R0(), 0, 0, false);
        double fx_0 = 1./8.;
        psipmax = -fx_0/(1.-fx_0)*psipO;
        std::cout << "psi 1 is          "<<psipmax<<"\n";
        gX2d = std::make_unique<dg::geo::CurvilinearGridX2d>(generator, fx_0, 0., npsi, Npsi, 640, dg::DIR, dg::NEU);
        std::cout << "DONE! \n";
        dg::Average<dg::HVec > avg_eta( gX2d->grid(), dg::coo2d::y);
        std::vector<dg::HVec> coordsX = gX2d->map();
        dg::SparseTensor<dg::HVec> metricX = gX2d->metric();
        dg::HVec volX2d = dg::tensor::volume2d( metricX);
        dg::blas1::pointwiseDot( coordsX[0], volX2d, volX2d); //R\sqrt{g}
        const double f0 = (gX2d->x1()-gX2d->x0())/ ( psipmax - psipO);
        dg::HVec dvdpsip;
        avg_eta( volX2d, dvdpsip, false);
        dg::blas1::scal( dvdpsip, 4.*M_PI*M_PI*f0);
        dg::Grid1d gX1d(psipO<psipmax ? psipO : psipmax,
            psipO<psipmax ? psipmax : psipO, npsi ,Npsi,dg::DIR_NEU); //inner value is always zero
        dg::HVec X_psi_vol = dg::integrate( dvdpsip, gX1d);
        map1d.emplace_back( "dvdpsip", dvdpsip,
            "Derivative of flux volume with respect to flux label psi");
        map1d.emplace_back( "psi_vol", X_psi_vol,
            "Flux volume on X-point grid");

        //NOTE: VOLUME is WITHIN cells while AREA is ON gridpoints
        dg::HVec gradZetaX = metricX.value(0,0), X_psi_area;
        dg::blas1::transform( gradZetaX, gradZetaX, dg::SQRT<double>());
        dg::blas1::pointwiseDot( volX2d, gradZetaX, gradZetaX); //R\sqrt{g}|\nabla\zeta|
        avg_eta( gradZetaX, X_psi_area, false);
        dg::blas1::scal( X_psi_area, 4.*M_PI*M_PI);
        map1d.emplace_back( "psi_area", X_psi_area,
            "Flux area on X-point grid");
        std::cout << "Total volume within separatrix is "<< dg::interpolate( dg::xspace, X_psi_vol, 0., gX1d)<<std::endl;

        //Compute FSA of cylindrical functions
        dg::HVec transferH, transferH1d;
        for( auto tp : map)
        {
            transferH = dg::pullback( std::get<2>(tp), *gX2d);
            dg::blas1::pointwiseDot( volX2d, transferH, transferH);
            avg_eta( transferH, transferH1d, false);
            dg::blas1::scal( transferH1d, 4*M_PI*M_PI*f0); //
            dg::blas1::pointwiseDivide( transferH1d, dvdpsip, transferH1d );
            map1d.emplace_back( std::get<0>(tp)+"_fsa", transferH1d,
                std::get<1>(tp)+" (Flux surface average)");
            dg::blas1::pointwiseDot( transferH1d, dvdpsip, transferH1d );
            transferH1d = dg::integrate( transferH1d, gX1d);
            map1d.emplace_back( std::get<0>(tp)+"_ifs", transferH1d,
                std::get<1>(tp)+" (Flux surface integral)");

        }
    }
    /// --------- More flux labels --------------------------------
    dg::Grid1d grid1d(psipO<psipmax ? psipO : psipmax,
            psipO<psipmax ? psipmax : psipO, npsi ,Npsi,dg::DIR_NEU); //inner value is always zero
    if( mag_description != dg::geo::description::none && mag_description != dg::geo::description::centeredX)
    {
        dg::HVec rho = dg::evaluate( dg::cooX1d, grid1d);
        dg::blas1::axpby( -1./psipO, rho, +1., 1., rho); //transform psi to rho
        map1d.emplace_back("rho", rho,
            "Alternative flux label rho = -psi/psimin + 1");
        dg::blas1::transform( rho, rho, dg::SQRT<double>());
        map1d.emplace_back("rho_p", rho,
            "Alternative flux label rho_p = Sqrt[-psi/psimin + 1]");
        if( mag_description == dg::geo::description::standardX || mag_description == dg::geo::description::standardO)
        {
            dg::geo::SafetyFactor qprof( mod_mag); //mag can lead to dead-lock!?
            dg::HVec qprofile = dg::evaluate( qprof, grid1d);
            map1d.emplace_back("q-profile", qprofile,
                "q-profile (Safety factor) using direct integration");
            dg::HVec psit = dg::integrate( qprofile, grid1d);
            map1d.emplace_back("psit1d", psit,
                "Toroidal flux label psi_t integrated  on grid1d using direct q");
            //we need to avoid integrating >=0
            dg::Grid1d g1d_fine(psipO<0. ? psipO : 0.,
                    psipO<0. ? 0. : psipO, npsi, Npsi,dg::NEU);
            qprofile = dg::evaluate( qprof, g1d_fine);
            dg::HVec w1d = dg::create::weights( g1d_fine);
            double psit_tot = dg::blas1::dot( w1d, qprofile);
            //std::cout << "psit tot "<<psit_tot<<"\n";
            dg::blas1::scal ( psit, 1./psit_tot);
            dg::blas1::transform( psit, psit, dg::SQRT<double>());
            map1d.emplace_back("rho_t", psit,
                "Toroidal flux label rho_t = sqrt( psit/psit_tot) evaluated on grid1d");
        }
    }

    /////////////////////////////set up netcdf/////////////////////////////////////
    std::cout << "CREATING/OPENING FILE AND WRITING ... \n";
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_create( newfilename.data(), NC_NETCDF4|NC_CLOBBER, &ncid);
    /// Set global attributes
    std::map<std::string, std::string> att;
    att["title"] = "Output file of feltor/inc/geometries/geometry_diag.cu";
    att["Conventions"] = "CF-1.7";
    ///Get local time and begin file history
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

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

    int dim1d_ids[1], dim2d_ids[2], dim3d_ids[3] ;
    if( mag_description == dg::geo::description::standardX)
    {
        int dim_idsX[2] = {0,0};
        err = dg::file::define_dimensions( ncid, dim_idsX, gX2d->grid(), {"eta", "zeta"} );
        std::string long_name = "Flux surface label";
        err = nc_put_att_text( ncid, dim_idsX[0], "long_name",
            long_name.size(), long_name.data());
        long_name = "Flux angle";
        err = nc_put_att_text( ncid, dim_idsX[1], "long_name",
            long_name.size(), long_name.data());
        int xccID, yccID;
        err = nc_def_var( ncid, "xcc", NC_DOUBLE, 2, dim_idsX, &xccID);
        err = nc_def_var( ncid, "ycc", NC_DOUBLE, 2, dim_idsX, &yccID);
        long_name="Cartesian x-coordinate";
        err = nc_put_att_text( ncid, xccID, "long_name",
            long_name.size(), long_name.data());
        long_name="Cartesian y-coordinate";
        err = nc_put_att_text( ncid, yccID, "long_name",
            long_name.size(), long_name.data());
        err = nc_enddef( ncid);
        err = nc_put_var_double( ncid, xccID, gX2d->map()[0].data());
        err = nc_put_var_double( ncid, yccID, gX2d->map()[1].data());
        err = nc_redef(ncid);
        dim1d_ids[0] = dim_idsX[1];
    }
    else
    {
        err = dg::file::define_dimension( ncid, &dim1d_ids[0], grid1d, "zeta");
        std::string psi_long_name = "Flux surface label";
        err = nc_put_att_text( ncid, dim1d_ids[0], "long_name",
            psi_long_name.size(), psi_long_name.data());
    }
    dg::CylindricalGrid3d grid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, n,Nx,Ny,Nz);
    dg::RealCylindricalGrid3d<float> fgrid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, n,Nx,Ny,Nz);

    err = dg::file::define_dimensions( ncid, &dim3d_ids[0], fgrid3d);
    dim2d_ids[0] = dim3d_ids[1], dim2d_ids[1] = dim3d_ids[2];

    //write 1d vectors
    std::cout << "WRTING 1D FIELDS ... \n";
    for( auto tp : map1d)
    {
        int vid;
        err = nc_def_var( ncid, std::get<0>(tp).data(), NC_DOUBLE, 1,
            &dim1d_ids[0], &vid);
        err = nc_put_att_text( ncid, vid, "long_name",
            std::get<2>(tp).size(), std::get<2>(tp).data());
        err = nc_enddef( ncid);
        err = nc_put_var_double( ncid, vid, std::get<1>(tp).data());
        err = nc_redef(ncid);
    }
    //write 2d vectors
    //allocate mem for visual
    dg::HVec hvisual = dg::evaluate( dg::zero, grid2d);
    dg::HVec hvisual3d = dg::evaluate( dg::zero, grid3d);
    dg::fHVec fvisual, fvisual3d;
    std::cout << "WRTING 2D/3D CYLINDRICAL FIELDS ... \n";
    for(auto tp : map)
    {
        int vectorID, vectorID3d;
        err = nc_def_var( ncid, std::get<0>(tp).data(), NC_FLOAT, 2,
            &dim2d_ids[0], &vectorID);
        err = nc_def_var( ncid, (std::get<0>(tp)+"3d").data(), NC_FLOAT, 3,
            &dim3d_ids[0], &vectorID3d);
        err = nc_put_att_text( ncid, vectorID, "long_name",
            std::get<1>(tp).size(), std::get<1>(tp).data());
        err = nc_put_att_text( ncid, vectorID3d, "long_name",
            std::get<1>(tp).size(), std::get<1>(tp).data());
        std::string coordinates = "zc yc xc";
        err = nc_put_att_text( ncid, vectorID3d, "coordinates", coordinates.size(), coordinates.data());
        err = nc_enddef( ncid);
        hvisual = dg::evaluate( std::get<2>(tp), grid2d);
        dg::extend_line( grid2d.size(), grid3d.Nz(), hvisual, hvisual3d);
        dg::assign( hvisual, fvisual);
        dg::assign( hvisual3d, fvisual3d);
        err = nc_put_var_float( ncid, vectorID, fvisual.data());
        err = nc_put_var_float( ncid, vectorID3d, fvisual3d.data());
        err = nc_redef(ncid);
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
        int vectorID;
        err = nc_def_var( ncid, std::get<0>(tp).data(), NC_FLOAT, 3,
            &dim3d_ids[0], &vectorID);
        err = nc_put_att_text( ncid, vectorID, "long_name",
            std::get<1>(tp).size(), std::get<1>(tp).data());
        if( std::get<1>(tp) != "xc" && std::get<1>(tp) != "yc" &&std::get<1>(tp) != "zc")
        {
            std::string coordinates = "zc yc xc";
            err = nc_put_att_text( ncid, vectorID, "coordinates", coordinates.size(), coordinates.data());
        }
        err = nc_enddef( ncid);
        hvisual3d = dg::evaluate( std::get<2>(tp), grid3d);
        dg::assign( hvisual3d, fvisual3d);
        err = nc_put_var_float( ncid, vectorID, fvisual3d.data());
        err = nc_redef(ncid);
    }
    //////////////////////////////Finalize////////////////////////////////////
    err = nc_close(ncid);
    return 0;
}
