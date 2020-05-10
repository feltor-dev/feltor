#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <functional>
#include <sstream>
#include <ctime>
#include <cmath>

#include "json/json.h"

#include "dg/algorithm.h"
#include "dg/file/nc_utilities.h"

#include "solovev.h"
//#include "taylor.h"
#include "magnetic_field.h"
#include "testfunctors.h"
#include "curvilinearX.h"
#include "separatrix_orthogonal.h"
#include "average.h"


// - write magnetic functions into file
// - test performance and accuracy of Flux surface averages and integrals
struct Parameters
{
    unsigned n, Nx, Ny, Nz, Npsi;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;
    double amp, k_psi, nprofileamp;
    double sigma, posX, posY;
    double damping_boundary, source_alpha, damping_alpha, source_boundary;
    double profile_alpha;
    Parameters( const Json::Value& js){
        n = js.get("n",3).asUInt();
        Nx = js.get("Nx",100).asUInt();
        Ny = js.get("Ny",100).asUInt();
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
        damping_boundary = js["damping"].get("boundary", 1.2).asDouble();
        damping_alpha = js["damping"].get("alpha", 0.1).asDouble();
        source_alpha = js["source"].get("alpha", 0.5).asDouble();
        source_boundary = js["source"].get("boundary", 0.5).asDouble();
    }
    void display( std::ostream& os = std::cout ) const
    {
        os << "Input parameters are: \n";
        os  <<" n             = "<<n<<"\n"
            <<" Nx            = "<<Nx<<"\n"
            <<" Ny            = "<<Ny<<"\n"
            <<" Nz            = "<<Nz<<"\n"
            <<" Npsi          = "<<Npsi<<"\n"
            <<" boxscaleRm    = "<<boxscaleRm<<"\n"
            <<" boxscaleRp    = "<<boxscaleRp<<"\n"
            <<" boxscaleZm    = "<<boxscaleZm<<"\n"
            <<" boxscaleZp    = "<<boxscaleZp<<"\n"
            <<" source bound  = "<<source_boundary<<"\n"
            <<" source alpha  = "<<source_alpha<<"\n"
            <<" damping bound = "<<damping_boundary<<"\n"
            <<" damping alpha = "<<damping_alpha<<"\n"
            <<" amp           = "<<amp<<"\n"
            <<" k_psi         = "<<k_psi<<"\n"
            <<" nprofileamp   = "<<nprofileamp<<"\n"
            <<" sigma         = "<<sigma<<"\n"
            <<" posX          = "<<posX<<"\n"
            <<" posY          = "<<posY<<"\n";
        os << std::flush;
    }
};


int main( int argc, char* argv[])
{
    if( !(argc == 4 || argc == 3))
    {
        std::cerr << "ERROR: Wrong number of arguments!\n";
        std::cerr << " Usage: "<< argv[0]<<" [input.json] [geom.json] [output.nc]\n";
        std::cerr << " ( Minimum input json file is { \"n\" : 3, \"Nx\": 100, \"Ny\":100 })\n";
        std::cerr << "Or \n Usage: "<< argv[0]<<" [file.nc] [output.nc]\n";
        std::cerr << " ( Program searches for string variables 'inputfile' and 'geomfile' in file.nc and tries a json parser)\n";
        return -1;
    }
    std::string newfilename;
    Json::Value input_js, geom_js;
    Json::CharReaderBuilder parser;
    parser["collectComments"] = false;
    std::string errs;
    if( argc == 4)
    {
        newfilename = argv[3];
        std::cout << argv[0]<< " "<<argv[1]<<" & "<<argv[2]<<" -> " <<argv[3]<<std::endl;
        std::ifstream isI( argv[1]);
        std::ifstream isG( argv[2]);
        parseFromStream( parser, isI, &input_js, &errs); //read input without comments
        parseFromStream( parser, isG, &geom_js, &errs); //read input without comments
    }
    else
    {
        newfilename = argv[2];
        std::cout << argv[0]<< " "<<argv[1]<<" -> " <<argv[2]<<std::endl;
        //////////////////////////open nc file//////////////////////////////////
        file::NC_Error_Handle err;
        int ncid;
        err = nc_open( argv[1], NC_NOWRITE, &ncid);
        ///////////////read in and show inputfile und geomfile//////////////////
        std::string input, geom;
        size_t length;
        err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
        input.resize( length, 'x');
        err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
        err = nc_inq_attlen( ncid, NC_GLOBAL, "geomfile", &length);
        geom.resize( length, 'x');
        err = nc_get_att_text( ncid, NC_GLOBAL, "geomfile", &geom[0]);
        nc_close( ncid);
        std::stringstream ss( input);
        parseFromStream( parser, ss, &input_js, &errs); //read input without comments
        ss.str( geom);
        parseFromStream( parser, ss, &geom_js, &errs); //read input without comments
    }
    const Parameters p(input_js);
    const dg::geo::solovev::Parameters gp(geom_js);
    p.display( std::cout);
    gp.display( std::cout);
    std::string input = input_js.toStyledString();
    std::string geom = geom_js.toStyledString();
    unsigned n, Nx, Ny, Nz;
    n = p.n, Nx = p.Nx, Ny = p.Ny, Nz = p.Nz;
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;

    //Test coefficients
    dg::geo::TokamakMagneticField mag_origin = dg::geo::createSolovevField(gp);
    dg::geo::TokamakMagneticField mag = mag_origin;
    if( p.damping_alpha > 0.)
    {
        double RO=mag.R0(), ZO=0.;
        dg::geo::findOpoint( mag.get_psip(), RO, ZO);
        double psipO = mag.psip()( RO, ZO);
        double damping_psi0p = (1.-p.damping_boundary*p.damping_boundary)*psipO;
        double damping_alphap = -(2.*p.damping_boundary+p.damping_alpha)*p.damping_alpha*psipO;
        std::cout<< " damping "<< damping_psi0p << " "<<damping_alphap<<"\n";
        mag = dg::geo::createModifiedSolovevField(gp, damping_psi0p+damping_alphap/2., fabs(damping_alphap/2.), ((psipO>0)-(psipO<0)));
    }
    //Find O-point
    double R_O = gp.R_0, Z_O = 0.;
    if( !gp.isToroidal() )
        dg::geo::findOpoint( mag.get_psip(), R_O, Z_O);
    const double psipmin = mag.psip()(R_O, Z_O);
    std::cout << "O-point found at "<<R_O<<" "<<Z_O<<" with Psip "<<psipmin<<std::endl;
    const double psip0 = mag.psip()(gp.R_0, 0);
    std::cout << "psip( R_0, 0) = "<<psip0<<"\n";


    dg::Grid2d grid2d(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny);
    dg::DVec psipog2d   = dg::evaluate( mag.psip(), grid2d);
    std::vector<std::tuple<std::string, dg::HVec, std::string> > map1d;
    ///////////TEST CURVILINEAR GRID TO COMPUTE FSA QUANTITIES
    unsigned npsi = 3, Npsi = p.Npsi;//set number of psivalues (NPsi % 8 == 0)
    double psipmax = dg::blas1::reduce( psipog2d, 0. ,thrust::maximum<double>()); //DEPENDS ON GRID RESOLUTION!!
    double volumeXGrid;
    /// -------  Elements for fsa of curvature operators ----------------
    // This is a numerical test because the fsa of the curvature operators exactly vanishes (at least for I = cte and almost for everything else)!!
    dg::geo::CylindricalVectorLvl0 bhat_ = dg::geo::createBHat( mag);
    dg::geo::CylindricalVectorLvl0 curvB_ = dg::geo::createTrueCurvatureNablaB( mag);
    dg::geo::CylindricalVectorLvl0 curvK_ = dg::geo::createTrueCurvatureKappa( mag);
    dg::HVec curvNablaBR = dg::evaluate( curvB_.x(), grid2d);
    dg::HVec curvNablaBZ = dg::evaluate( curvB_.y(), grid2d);
    dg::HVec curvKappaKR = dg::evaluate( curvK_.x(), grid2d);
    dg::HVec curvKappaKZ = dg::evaluate( curvK_.y(), grid2d);
    dg::HVec gradPsipR = dg::evaluate( mag.psipR(), grid2d);
    dg::HVec gradPsipZ = dg::evaluate( mag.psipZ(), grid2d);
    dg::HVec curvNablaBGradPsip(curvNablaBR), curvKappaKGradPsip( curvNablaBR), gradPsip(curvNablaBR);
    dg::blas1::pointwiseDot( 1., curvNablaBR, gradPsipR, 1., curvNablaBZ, gradPsipZ, 0., curvNablaBGradPsip);
    dg::blas1::pointwiseDot( 1., curvKappaKR, gradPsipR, 1., curvKappaKZ, gradPsipZ, 0., curvKappaKGradPsip);
    dg::blas1::pointwiseDot( 1., gradPsipR, gradPsipR, 1., gradPsipZ, gradPsipZ, 0., gradPsip);
    dg::blas1::transform( gradPsip, gradPsip, dg::SQRT<double>());
    if( gp.hasXpoint())
    {
        std::cout << "Generate X-point flux-aligned grid ... \n";
        double RX = gp.R_0-1.1*gp.triangularity*gp.a;
        double ZX = -1.1*gp.elongation*gp.a;
        dg::geo::findXpoint( mag.get_psip(), RX, ZX);
        dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( mag.get_psip(), RX, ZX) ;
        dg::geo::SeparatrixOrthogonal generator(mag.get_psip(), monitor_chi, psipmin, RX, ZX, mag.R0(), 0, 0, false);
        double fx_0 = 1./8.;
        psipmax = -fx_0/(1.-fx_0)*psipmin;
        //std::cout << "psi 1 is          "<<psipmax<<"\n";
        dg::geo::CurvilinearGridX2d gX2d( generator, fx_0, 0., npsi, Npsi, 640, dg::DIR, dg::NEU);
        std::cout << "DONE! \n";
        dg::Average<dg::HVec > avg_eta( gX2d.grid(), dg::coo2d::y);
        std::vector<dg::HVec> coordsX = gX2d.map();
        dg::SparseTensor<dg::HVec> metricX = gX2d.metric();
        dg::HVec volX2d = dg::tensor::volume2d( metricX);
        dg::blas1::pointwiseDot( coordsX[0], volX2d, volX2d); //R\sqrt{g}
        dg::IHMatrix grid2gX2d = dg::create::interpolation( coordsX[0], coordsX[1], grid2d);
        dg::HVec psip_X = dg::evaluate( dg::one, gX2d), dvdzeta, X_psip1d;
        dg::blas2::symv( grid2gX2d, (dg::HVec)psipog2d, psip_X);
        dg::blas1::pointwiseDot( volX2d, psip_X, psip_X);
        avg_eta( psip_X, X_psip1d, false);
        dg::blas1::scal( X_psip1d, 4.*M_PI*M_PI);
        avg_eta( volX2d, dvdzeta, false);
        dg::blas1::scal( dvdzeta, 4.*M_PI*M_PI);
        dg::Grid1d gX1d( gX2d.x0(), gX2d.x1(), npsi, Npsi, dg::NEU);
        dg::HVec X_psi_vol = dg::integrate( dvdzeta, gX1d);
        map1d.emplace_back( "X_dvdzeta", dvdzeta,
            "dvdzeta on X-point grid");
        map1d.emplace_back( "X_psi_vol", X_psi_vol,
            "Flux volume on X-point grid");
        dg::blas1::pointwiseDivide( X_psip1d, dvdzeta, X_psip1d);
        map1d.emplace_back( "X_psip_fsa", X_psip1d,
            "Flux surface average of psi on X-point grid");

        //NOTE: VOLUME is WITHIN cells while AREA is ON gridpoints
        dg::HVec gradZetaX = metricX.value(0,0), X_psi_area;
        dg::blas1::transform( gradZetaX, gradZetaX, dg::SQRT<double>());
        dg::blas1::pointwiseDot( volX2d, gradZetaX, gradZetaX); //R\sqrt{g}|\nabla\zeta|
        avg_eta( gradZetaX, X_psi_area, false);
        dg::blas1::scal( X_psi_area, 4.*M_PI*M_PI);
        map1d.emplace_back( "X_psi_area", X_psi_area,
            "Flux area on X-point grid");
        volumeXGrid = dg::interpolate( dg::xspace, X_psi_vol, gX1d.x1(), gX1d);

        //Compute FSA of curvature operators
        dg::HVec X_curvNablaBGradPsip( psip_X), X_curvKappaKGradPsip( psip_X), X_gradPsip(psip_X);
        dg::HVec X_curvNablaB_fsa, X_curvKappaK_fsa, X_gradPsip_fsa;
        dg::blas2::symv( grid2gX2d, curvNablaBGradPsip, X_curvNablaBGradPsip);
        dg::blas2::symv( grid2gX2d, curvKappaKGradPsip, X_curvKappaKGradPsip);
        dg::blas2::symv( grid2gX2d, gradPsip, X_gradPsip);
        dg::blas1::pointwiseDot( volX2d, X_curvNablaBGradPsip, X_curvNablaBGradPsip);
        dg::blas1::pointwiseDot( volX2d, X_curvKappaKGradPsip, X_curvKappaKGradPsip);
        dg::blas1::pointwiseDot( volX2d, X_gradPsip, X_gradPsip);
        avg_eta( X_curvNablaBGradPsip, X_curvNablaB_fsa, false);
        avg_eta( X_curvKappaKGradPsip, X_curvKappaK_fsa, false);
        avg_eta( X_gradPsip, X_gradPsip_fsa, false);
        dg::blas1::scal( X_curvNablaB_fsa, 4*M_PI*M_PI); //
        dg::blas1::scal( X_curvKappaK_fsa, 4*M_PI*M_PI); //
        dg::blas1::scal( X_gradPsip_fsa, 4*M_PI*M_PI); //
        dg::blas1::pointwiseDivide( X_curvNablaB_fsa, dvdzeta, X_curvNablaB_fsa );
        dg::blas1::pointwiseDivide( X_curvKappaK_fsa, dvdzeta, X_curvKappaK_fsa );
        dg::blas1::pointwiseDivide( X_gradPsip_fsa, dvdzeta, X_gradPsip_fsa );
        map1d.emplace_back( "X_curvNablaB_fsa", X_curvNablaB_fsa,
            "Flux surface average of true Nabla B curvature Dot Grad Psip");
        map1d.emplace_back( "X_curvKappaK_fsa", X_curvKappaK_fsa,
            "Flux surface average of true Kappa curvature Dot Grad Psip");
        map1d.emplace_back( "X_gradPsip_fsa", X_gradPsip_fsa,
            "Flux surface average of |Grad Psip|");
        // h02 factor
        dg::HVec h02 = dg::pullback( dg::geo::Hoo(mag), gX2d), X_h02_fsa;
        dg::blas1::pointwiseDot( volX2d, h02, h02);
        avg_eta( h02, X_h02_fsa, false);
        dg::blas1::scal( X_h02_fsa, 4*M_PI*M_PI); //
        dg::blas1::pointwiseDivide( X_h02_fsa, dvdzeta, X_h02_fsa );
        map1d.emplace_back( "X_hoo_fsa", X_h02_fsa,
            "Flux surface average of novel h02 factor");
        // total source term
        h02 = dg::pullback( dg::compose( dg::PolynomialHeaviside(
                    p.source_boundary-p.source_alpha/2., p.source_alpha/2., -1 ),
                dg::geo::RhoP(mag)), gX2d);
        dg::blas1::pointwiseDot( volX2d, h02, h02);
        avg_eta( h02, X_h02_fsa, false);
        dg::blas1::scal( X_h02_fsa, 4*M_PI*M_PI); //
        dg::blas1::pointwiseDivide( X_h02_fsa, dvdzeta, X_h02_fsa );
        map1d.emplace_back( "X_sne_fsa", X_h02_fsa,
            "Flux surface average over source term");
        dg::blas1::pointwiseDot( X_h02_fsa, dvdzeta, X_h02_fsa );
        X_h02_fsa = dg::integrate( X_h02_fsa, gX1d);
        map1d.emplace_back( "X_sne_ifs", X_h02_fsa,
            "Flux surface integral over source term");
        //divb
        h02 = dg::pullback( dg::geo::Divb(mag), gX2d);
        dg::blas1::pointwiseDot( volX2d, h02, h02);
        avg_eta( h02, X_h02_fsa, false);
        dg::blas1::scal( X_h02_fsa, 4*M_PI*M_PI); //
        dg::blas1::pointwiseDivide( X_h02_fsa, dvdzeta, X_h02_fsa );
        map1d.emplace_back( "X_divb_fsa", X_h02_fsa,
            "Flux surface average of divb");
    }

    ///////////////////Compute flux average////////////////////
    dg::Grid1d grid1d( 0,1,npsi,Npsi, dg::NEU);
    double deltapsi = 0.1;
    if( !gp.isToroidal())
    {
        std::cout << "Compute flux averages\n";
        dg::HVec xpoint_weights = dg::evaluate( dg::cooX2d, grid2d);
        if( gp.hasXpoint() )
        {
            double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
            double Z_X = -1.1*gp.elongation*gp.a;
            dg::geo::findXpoint( mag.get_psip(), R_X, Z_X);
            dg::blas1::pointwiseDot( xpoint_weights,
                    dg::evaluate( dg::geo::ZCutter(Z_X), grid2d), xpoint_weights);
        }
        dg::geo::FluxSurfaceAverage<dg::DVec>  fsa( grid2d, mag, psipog2d, xpoint_weights);
        grid1d = dg::Grid1d (psipmin<psipmax ? psipmin : psipmax, psipmin<psipmax ? psipmax : psipmin, npsi ,Npsi,dg::NEU);
        map1d.emplace_back("psi_fsa",   dg::evaluate( fsa,      grid1d),
            "Flux surface average of psi with delta function");
        dg::HVec qprofile;
        if( gp.equilibrium == "solovev")
        {
            dg::geo::SafetyFactor qprof( mag);
            qprofile = dg::evaluate( qprof, grid1d);
            map1d.emplace_back("q-profile", qprofile,
                "q-profile (Safety factor) using direct integration");
            dg::HVec psit = dg::integrate( qprofile, grid1d);
            map1d.emplace_back("psit1d", psit,
                "Toroidal flux label psi_t integrated  on grid1d using direct q");
            //we need to avoid integrating >=0
            dg::Grid1d g1d_fine(psipmin<0. ? psipmin : 0., psipmin<0. ? 0. : psipmin, npsi ,Npsi,dg::NEU);
            qprofile = dg::evaluate( qprof, g1d_fine);
            dg::HVec w1d = dg::create::weights( g1d_fine);
            double psit_tot = dg::blas1::dot( w1d, qprofile);
            //std::cout << "psit tot "<<psit_tot<<"\n";
            dg::blas1::scal ( psit, 1./psit_tot);
            dg::blas1::transform( psit, psit, dg::SQRT<double>());
            map1d.emplace_back("rho_t", psit,
                "Toroidal flux label rho_t = sqrt( psit/psit_tot) evaluated on grid1d");
        }
        dg::geo::SafetyFactorAverage qprof( grid2d, mag);
        qprofile = dg::evaluate( qprof, grid1d);
        map1d.emplace_back("q-profile_fsa", qprofile,
            "q-profile (Safety factor) using average integration");
        dg::HVec psit = dg::integrate( qprofile, grid1d);
        map1d.emplace_back("psit1d_fsa", psit,
            "Toroidal flux label psi_t integrated on grid1d using average q");
        map1d.emplace_back("psip1d",    dg::evaluate( dg::cooX1d, grid1d),
            "Poloidal flux label psi_p evaluated on grid1d");
        dg::HVec rho = dg::evaluate( dg::cooX1d, grid1d);
        dg::blas1::axpby( -1./psipmin, rho, +1., 1., rho); //transform psi to rho
        map1d.emplace_back("rho", rho,
            "Alternative flux label rho = -psi/psimin + 1");
        dg::blas1::transform( rho, rho, dg::SQRT<double>());
        map1d.emplace_back("rho_p", rho,
            "Alternative flux label rho_p = Sqrt[-psi/psimin + 1]");
        fsa.set_container( (dg::DVec)curvNablaBGradPsip);
        map1d.emplace_back("curvNablaB_fsa",   dg::evaluate( fsa,      grid1d),
            "Flux surface average of true Nabla B curvature Dot Grad Psip with delta function");
        fsa.set_container( (dg::DVec)curvKappaKGradPsip);
        map1d.emplace_back("curvKappaK_fsa",   dg::evaluate( fsa,      grid1d),
            "Flux surface average of true Kappa curvature Dot Grad Psip with delta function");
        fsa.set_container( (dg::DVec)gradPsip);
        map1d.emplace_back("gradPsip_fsa",   dg::evaluate( fsa,      grid1d),
            "Flux surface average of |Grad Psip| with delta function");

        //other flux labels
        dg::geo::FluxSurfaceIntegral<dg::HVec> fsi( grid2d, mag);
        fsi.set_right( xpoint_weights);
        deltapsi = fsi.get_deltapsi();

        dg::HVec areaT_psip = dg::evaluate( fsi, grid1d);
        dg::HVec w1d = dg::create::weights( grid1d);
        double volumeCoarea = 2.*M_PI*dg::blas1::dot( areaT_psip, w1d);

        //area
        fsi.set_left( dg::evaluate( dg::geo::GradPsip(mag), grid2d));
        dg::HVec psi_area = dg::evaluate( fsi, grid1d);
        dg::blas1::scal(psi_area, 2.*M_PI);
        map1d.emplace_back( "psi_area", psi_area,
            "Flux area with delta function");
        // h02 factor
        dg::HVec h02 = dg::evaluate( dg::geo::Hoo(mag), grid2d);
        fsa.set_container( h02);
        map1d.emplace_back( "hoo_fsa", dg::evaluate( fsa, grid1d),
           "Flux surface average of novel h02 factor with delta function");



        dg::geo::FluxVolumeIntegral<dg::HVec> fvi( (dg::CartesianGrid2d)grid2d, mag);
        //std::cout << "Delta Rho for Flux surface integrals = "<<-deltapsi/psipmin<<"\n";

        fvi.set_right( xpoint_weights);
        dg::HVec psi_vol = dg::evaluate( fvi, grid1d);
        dg::blas1::scal(psi_vol, 2.*M_PI);
        map1d.emplace_back( "psi_vol", psi_vol,
            "Flux volume with delta function");
        double volumeFVI = 2.*M_PI*fvi(psipmax);
        double volumeSep = 2.*M_PI*fvi(0.);
        std::cout << "volume enclosed by separatrix: "<<volumeSep<<"\n";
        std::cout << "volume test with coarea formula: "<<volumeCoarea<<" "<<volumeFVI
                  <<" rel error = "<<fabs(volumeCoarea-volumeFVI)/volumeFVI<<"\n";
        if(gp.hasXpoint()){
            std::cout << "volume test with x grid formula: "<<volumeXGrid<<" "<<volumeFVI
                      <<" rel error = "<<fabs(volumeXGrid-volumeFVI)/volumeFVI<<"\n";
        };
    }

    /////////////////////////////set up netcdf/////////////////////////////////////
    std::cout << "CREATING/OPENING FILE AND WRITING ... \n";
    file::NC_Error_Handle err;
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
    err = file::define_dimension( ncid, &dim1d_ids[0], grid1d, "psi");
    std::string psi_long_name = "Flux surface label";
    err = nc_put_att_text( ncid, dim1d_ids[0], "long_name",
        psi_long_name.size(), psi_long_name.data());
    dg::CylindricalGrid3d grid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, n,Nx,Ny,Nz);
    dg::RealCylindricalGrid3d<float> fgrid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, n,Nx,Ny,Nz);

    err = file::define_dimensions( ncid, &dim3d_ids[0], fgrid3d);
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
    std::vector< std::tuple<std::string, std::string, dg::geo::CylindricalFunctor >> map2d{
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
        {"GradPsip", "Module of gradient of Psip", dg::geo::GradPsip(mag)},
        //////////////////////////////////
        {"Iris", "A flux aligned Iris", dg::compose( dg::Iris( gp.psipmin, gp.psipmax), mag.psip())},
        {"Pupil", "A flux aligned Pupil", dg::compose( dg::Pupil(gp.psipmaxcut), mag.psip()) },
        {"GaussianDamping", "A flux aligned Heaviside with Gaussian damping", dg::compose( dg::GaussianDamping( gp.psipmaxcut, p.source_alpha), mag.psip()) },
        {"ZonalFlow",  "Flux aligned Sine function", dg::compose( dg::SinX ( p.amp, 0., 2.*M_PI*p.k_psi ), mag.psip())},
        {"PsiLimiter", "A flux aligned Heaviside", dg::compose( dg::Heaviside( gp.psipmaxlim), mag.psip() )},
        {"SourceProfile", "A source profile", dg::compose( dg::PolynomialHeaviside(
                    p.source_boundary-p.source_alpha/2., p.source_alpha/2., -1 ),
                dg::geo::RhoP(mag))},
        {"ProfileDamping", "Density profile damping", dg::compose(dg::PolynomialHeaviside(
            1.-p.profile_alpha/2., p.profile_alpha/2., -1), dg::geo::RhoP(mag)) },
        {"MagneticTransition", "The region where the magnetic field is modified", dg::compose(dg::DPolynomialHeaviside(
            p.damping_boundary+p.damping_alpha/2.,
            p.damping_alpha/2., +1 ), dg::geo::RhoP(mag_origin))},
        {"Nprofile", "A flux aligned profile", dg::compose( dg::LinearX( p.nprofileamp/mag.psip()(mag.R0(),0.), p.nprofileamp ), mag.psip())},
        {"Delta", "A flux aligned Gaussian peak", dg::compose( dg::GaussianX( psipmin*0.2, deltapsi, 1./(sqrt(2.*M_PI)*deltapsi)), mag.psip())},
        {"TanhDamping", "A flux aligned Heaviside with Tanh Damping", dg::compose( dg::TanhProfX( -3*p.source_alpha, p.source_alpha, -1), mag.psip())},
        ////
        {"BathRZ", "A randomized field", dg::BathRZ( 16, 16, Rmin,Zmin, 30.,2, p.amp)},
        {"Gaussian3d", "A Gaussian field", dg::Gaussian3d(gp.R_0+p.posX*gp.a, p.posY*gp.a,
            M_PI, p.sigma, p.sigma, p.sigma, p.amp)},
        { "Hoo", "The novel h02 factor", dg::geo::Hoo( mag) }

    };
    //allocate mem for visual
    dg::HVec hvisual = dg::evaluate( dg::zero, grid2d);
    dg::HVec hvisual3d = dg::evaluate( dg::zero, grid3d);
    dg::fHVec fvisual, fvisual3d;
    std::cout << "WRTING 2D/3D CYLINDRICAL FIELDS ... \n";
    for(auto tp : map2d)
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
