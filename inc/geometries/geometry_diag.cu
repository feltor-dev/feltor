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
#include "init.h"
#include "magnetic_field.h"
#include "testfunctors.h"
#include "curvilinearX.h"
#include "separatrix_orthogonal.h"
#include "average.h"

struct Parameters
{
    unsigned n, Nx, Ny, Nz, Npsi;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;
    double amp, k_psi, bgprofamp, nprofileamp;
    double sigma, posX, posY;
    double rho_damping, alpha_mag;
    Parameters( const Json::Value& js){
        n = js.get("n",3).asUInt();
        Nx = js.get("Nx",100).asUInt();
        Ny = js.get("Ny",100).asUInt();
        Nz = js.get("Nz", 1).asUInt();
        Npsi = js.get("Npsi", 16).asUInt();
        boxscaleRm = js.get("boxscaleRm", 1.1).asDouble();
        boxscaleRp = js.get("boxscaleRp", 1.1).asDouble();
        boxscaleZm = js.get("boxscaleZm", 1.2).asDouble();
        boxscaleZp = js.get("boxscaleZp", 1.1).asDouble();
        amp = js.get("amplitude", 1.).asDouble();
        k_psi = js.get("k_psi", 1.).asDouble();
        bgprofamp = js.get("bgprofamp", 1.).asDouble();
        nprofileamp = js.get("nprofileamp", 1.).asDouble();
        sigma = js.get("sigma", 10).asDouble();
        posX = js.get("posX", 0.5).asDouble();
        posY = js.get("posY", 0.5).asDouble();
        rho_damping = js.get("rho_damping", 1.2).asDouble();
        alpha_mag = js.get("alpha_mag", 0.05).asDouble();
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
            <<" amp           = "<<amp<<"\n"
            <<" k_psi         = "<<k_psi<<"\n"
            <<" bgprofamp     = "<<bgprofamp<<"\n"
            <<" nprofileamp   = "<<nprofileamp<<"\n"
            <<" sigma         = "<<sigma<<"\n"
            <<" posX          = "<<posX<<"\n"
            <<" posY          = "<<posY<<"\n";
        os << std::flush;
    }
};

struct IPhi
{
    IPhi( dg::geo::solovev::Parameters gp): R_0(gp.R_0), A(gp.A){}
    double operator()(double R, double Z, double phi)const
    {
        return ((A-1.)*R - A*R_0*R_0/R)/R_0/R_0/R_0;
    }
    private:
    double R_0, A;
};

int main( int argc, char* argv[])
{
    if( !(argc == 4 || argc == 3))
    {
        std::cerr << "ERROR: Wrong number of arguments!\n";
        std::cerr << " Usage: "<< argv[0]<<" [input.js] [geom.js] [output.nc]\n";
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
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    //mag = dg::geo::createModifiedSolovevField(gp, (1.-p.rho_damping)*mag.psip()(mag.R0(), 0.), p.alpha_mag);
    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    if( gp.hasXpoint())
    {
        dg::geo::findXpoint( mag.get_psip(), R_X, Z_X);
        std::cout <<  "X-point found at "<<R_X << " "<<Z_X<<"\n";
    }
    const double R_H = gp.R_0-gp.triangularity*gp.a;
    const double Z_H = gp.elongation*gp.a;
    const double alpha_ = asin(gp.triangularity);
    const double N1 = -(1.+alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.+alpha_);
    const double N2 =  (1.-alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.-alpha_);
    const double N3 = -gp.elongation/(gp.a*cos(alpha_)*cos(alpha_));
    const double psip0 = mag.psip()(gp.R_0, 0);
    std::cout << "psip( 1, 0) "<<psip0<<"\n";
    //Find O-point
    double R_O = gp.R_0, Z_O = 0.;
    dg::geo::findXpoint( mag.get_psip(), R_O, Z_O);
    const double psipmin = mag.psip()(R_O, Z_O);

    std::cout << "O-point "<<R_O<<" "<<Z_O<<" with Psip = "<<psipmin<<std::endl;

    std::cout << "TEST ACCURACY OF PSI (values must be close to 0!)\n";
    if( gp.hasXpoint())
        std::cout << "    Equilibrium with X-point!\n";
    else
        std::cout << "    NO X-point in flux function!\n";
    std::cout << "psip( 1+e,0)           "<<mag.psip()(gp.R_0 + gp.a, 0.)<<"\n";
    std::cout << "psip( 1-e,0)           "<<mag.psip()(gp.R_0 - gp.a, 0.)<<"\n";
    std::cout << "psip( 1-de,ke)         "<<mag.psip()(R_H, Z_H)<<"\n";
    if( !gp.hasXpoint())
        std::cout << "psipR( 1-de,ke)        "<<mag.psipR()(R_H, Z_H)<<"\n";
    else
    {
        std::cout << "psip( 1-1.1de,-1.1ke)  "<<mag.psip()(R_X, Z_X)<<"\n";
        std::cout << "psipZ( 1+e,0)          "<<mag.psipZ()(gp.R_0 + gp.a, 0.)<<"\n";
        std::cout << "psipZ( 1-e,0)          "<<mag.psipZ()(gp.R_0 - gp.a, 0.)<<"\n";
        std::cout << "psipR( 1-de,ke)        "<<mag.psipR()(R_H,Z_H)<<"\n";
        std::cout << "psipR( 1-1.1de,-1.1ke) "<<mag.psipR()(R_X,Z_X)<<"\n";
        std::cout << "psipZ( 1-1.1de,-1.1ke) "<<mag.psipZ()(R_X,Z_X)<<"\n";
    }
    std::cout << "psipZZ( 1+e,0)         "<<mag.psipZZ()(gp.R_0+gp.a,0.)+N1*mag.psipR()(gp.R_0+gp.a,0)<<"\n";
    std::cout << "psipZZ( 1-e,0)         "<<mag.psipZZ()(gp.R_0-gp.a,0.)+N2*mag.psipR()(gp.R_0-gp.a,0)<<"\n";
    std::cout << "psipRR( 1-de,ke)       "<<mag.psipRR()(R_H,Z_H)+N3*mag.psipZ()(R_H,Z_H)<<"\n";


    dg::HVec hvisual;
    //allocate mem for visual
    dg::HVec visual;
    std::vector< std::tuple<std::string, std::string, std::function<double(double,double)> > > map{
        {"Psip", "Flux function", mag.psip()},
        {"PsipR", "Flux function derivative in R", mag.psipR()},
        {"PsipZ", "Flux function derivative in Z", mag.psipZ()},
        {"Ipol", "Poloidal current", mag.ipol()},
        {"Bmodule", "Magnetic field strength", dg::geo::Bmodule(mag)},
        {"InvB", "Inverse of Bmodule", dg::geo::InvB(mag)},
        {"LnB", "Natural logarithm of Bmodule", dg::geo::LnB(mag)},
        {"GradLnB", "The parallel derivative of LnB", dg::geo::GradLnB(mag)},
        {"Divb", "The divergence of the magnetic unit vector", dg::geo::Divb(mag)},
        {"B_R", "Derivative of Bmodule in R", dg::geo::BR(mag)},
        {"B_Z", "Derivative of Bmodule in Z", dg::geo::BZ(mag)},
        {"CurvatureNablaBR",  "R-component of the (toroidal) Nabla B curvature vector", dg::geo::CurvatureNablaBR(mag)},
        {"CurvatureNablaBZ",  "Z-component of the (toroidal) Nabla B curvature vector", dg::geo::CurvatureNablaBZ(mag)},
        {"CurvatureKappaR",   "R-component of the (toroidal) Kappa B curvature vector", dg::geo::CurvatureKappaR(mag)},
        {"CurvatureKappaZ",   "Z-component of the (toroidal) Kappa B curvature vector", dg::geo::CurvatureKappaZ(mag)},
        {"DivCurvatureKappa", "Divergence of the (toroidal) Kappa B curvature vector", dg::geo::DivCurvatureKappa(mag)},
        {"DivCurvatureNablaB","Divergence of the (toroidal) Nabla B curvature vector", dg::geo::DivCurvatureNablaB(mag)},
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
        //////////////////////////////////
        {"Iris", "A flux aligned Heaviside", dg::geo::Iris(mag.psip(), gp.psipmin, gp.psipmax)},
        {"Pupil", "A flux aligned Heaviside", dg::geo::Pupil(mag.psip(), gp.psipmaxcut)},
        {"GaussianDamping", "A flux aligned Heaviside with Gaussian damping", dg::geo::GaussianDamping(mag.psip(), gp.psipmaxcut, gp.alpha)},
        {"ZonalFlow",  "Flux aligned zonal flows", dg::geo::ZonalFlow(mag.psip(), p.amp, 0., 2.*M_PI*p.k_psi )},
        {"PsiLimiter", "A flux aligned Heaviside", dg::geo::PsiLimiter(mag.psip(), gp.psipmaxlim)},
        {"Nprofile", "A flux aligned profile", dg::geo::Nprofile(mag.psip(), p.nprofileamp/mag.psip()(mag.R0(),0.), p.bgprofamp )},
        {"Delta", "A flux aligned delta function", dg::geo::DeltaFunction( mag, gp.alpha*gp.alpha, psip0*0.2)},
        {"TanhDamping", "A flux aligned Heaviside with Tanh Damping", dg::geo::TanhDamping(mag.psip(), -3*gp.alpha, gp.alpha, -1)},
        ////
        {"BathRZ", "A randomized field", dg::BathRZ( 16, 16, Rmin,Zmin, 30.,2, p.amp)},
        {"Gaussian3d", "A Gaussian field", dg::Gaussian3d(gp.R_0+p.posX*gp.a, p.posY*gp.a,
            M_PI, p.sigma, p.sigma, p.sigma, p.amp)}
    };
    dg::Grid2d grid2d(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny);
    dg::DVec psipog2d   = dg::evaluate( mag.psip(), grid2d);
    std::vector<std::tuple<std::string, dg::HVec, std::string> > map1d;
    ///////////TEST CURVILINEAR GRID TO COMPUTE FSA QUANTITIES
    unsigned npsi = 3, Npsi = p.Npsi;//set number of psivalues (NPsi % 8 == 0)
    double psipmax = dg::blas1::reduce( psipog2d, 0. ,thrust::maximum<double>()); //DEPENDS ON GRID RESOLUTION!!
    double volumeXGrid;
    /// -------  Elements for fsa of curvature operators ----------------
    // This is a numerical test because the fsa of the curvature operators exactly vanishes (at least for I = cte)
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
        std::cout << "Generate X-point flux-aligned grid!\n";
        double RX = gp.R_0-1.1*gp.triangularity*gp.a;
        double ZX = -1.1*gp.elongation*gp.a;
        dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( mag.get_psip(), RX, ZX) ;
        dg::geo::SeparatrixOrthogonal generator(mag.get_psip(), monitor_chi, psipmin, R_X, Z_X, mag.R0(), 0, 0, true);
        double fx_0 = 1./8.;
        psipmax = -fx_0/(1.-fx_0)*psipmin;
        std::cout << "psi 1 is          "<<psipmax<<"\n";
        dg::geo::CurvilinearGridX2d gX2d( generator, fx_0, 0., npsi, Npsi, 360, dg::DIR, dg::NEU);
        std::cout << "DONE! Generate X-point flux-aligned grid!\n";
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
        volumeXGrid = dg::interpolate( X_psi_vol, gX1d.x1(), gX1d);

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
    }

    ///////////////////Compute flux average////////////////////
    std::cout << "Compute flux averages\n";
    dg::HVec xpoint_weights = dg::evaluate( dg::cooX2d, grid2d);
    if( gp.hasXpoint() )
        dg::blas1::pointwiseDot( xpoint_weights , dg::evaluate( dg::geo::ZCutter(Z_X), grid2d), xpoint_weights);
    dg::geo::FluxSurfaceAverage<dg::DVec>  fsa( grid2d, mag, psipog2d, xpoint_weights);
    dg::Grid1d grid1d(psipmin, psipmax, npsi ,Npsi,dg::NEU);
    map1d.emplace_back("psi_fsa",   dg::evaluate( fsa,      grid1d),
        "Flux surface average of psi with delta function");
    if( gp.equilibrium == "solovev")
    {
        dg::geo::SafetyFactor     qprof( mag);
        map1d.emplace_back("q-profile", dg::evaluate( qprof,    grid1d),
            "q-profile (Safety factor) using direct integration");
    }
    else
    {
        dg::geo::SafetyFactorAverage     qprof( grid2d, mag);
        map1d.emplace_back("q-profile", dg::evaluate( qprof,    grid1d),
            "q-profile (Safety factor) using average integration");
    }
    map1d.emplace_back("psip1d",    dg::evaluate( dg::cooX1d, grid1d),
        "Flux label psi evaluated on grid1d");
    dg::HVec rho = dg::evaluate( dg::cooX1d, grid1d);
    dg::blas1::axpby( -1./psipmin, rho, +1., 1., rho); //transform psi to rho
    map1d.emplace_back("rho", rho,
        "Alternative flux label rho = -psi/psimin + 1");
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

    dg::HVec areaT_psip = dg::evaluate( fsi, grid1d);
    dg::HVec w1d = dg::create::weights( grid1d);
    double volumeCoarea = 2.*M_PI*dg::blas1::dot( areaT_psip, w1d);

    //area
    fsi.set_left( dg::evaluate( dg::geo::GradPsip(mag), grid2d));
    dg::HVec psi_area = dg::evaluate( fsi, grid1d);
    dg::blas1::scal(psi_area, 2.*M_PI);
    map1d.emplace_back( "psi_area", psi_area,
        "Flux area with delta function");

    dg::geo::FluxVolumeIntegral<dg::HVec> fvi( (dg::CartesianGrid2d)grid2d, mag);
    std::cout << "Delta Rho for Flux surface integrals = "<<-fsi.get_deltapsi()/psipmin<<"\n";

    fvi.set_right( xpoint_weights);
    dg::HVec psi_vol = dg::evaluate( fvi, grid1d);
    dg::blas1::scal(psi_vol, 2.*M_PI);
    map1d.emplace_back( "psi_vol", psi_vol,
        "Flux volume with delta function");
    double volumeFVI = 2.*M_PI*fvi(psipmax);
    std::cout << "VOLUME TEST WITH COAREA FORMULA: "<<volumeCoarea<<" "<<volumeFVI
              <<" rel error = "<<fabs(volumeCoarea-volumeFVI)/volumeFVI<<"\n";
    if(gp.hasXpoint()){
        std::cout << "VOLUME TEST WITH X Grid FORMULA: "<<volumeXGrid<<" "<<volumeFVI
                  <<" rel error = "<<fabs(volumeXGrid-volumeFVI)/volumeFVI<<"\n";
    };

    /////////////////////////////set up netcdf/////////////////////////////////////
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
    err = file::define_dimension( ncid, "psi", &dim1d_ids[0], grid1d);
    std::string psi_long_name = "Flux surface label";
    err = nc_put_att_text( ncid, dim1d_ids[0], "long_name",
        psi_long_name.size(), psi_long_name.data());
    dg::CylindricalGrid3d grid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, n,Nx,Ny,Nz);
    err = file::define_dimensions( ncid, &dim3d_ids[0], grid3d);
    dim2d_ids[0] = dim3d_ids[1], dim2d_ids[1] = dim3d_ids[2];

    //write 1d vectors
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
    for(auto tp : map)
    {
        int vectorID;
        err = nc_def_var( ncid, std::get<0>(tp).data(), NC_DOUBLE, 2,
            &dim2d_ids[0], &vectorID);
        err = nc_put_att_text( ncid, vectorID, "long_name",
            std::get<1>(tp).size(), std::get<1>(tp).data());
        err = nc_enddef( ncid);
        hvisual = dg::evaluate( std::get<2>(tp), grid2d);
        err = nc_put_var_double( ncid, vectorID, hvisual.data());
        err = nc_redef(ncid);

    }
    //compute & write 3d vectors
    dg::HVec vecR = dg::evaluate( dg::geo::BFieldR(mag), grid3d);
    dg::HVec vecZ = dg::evaluate( dg::geo::BFieldZ(mag), grid3d);
    dg::HVec vecP = dg::evaluate( dg::geo::BFieldP(mag), grid3d);
    std::string vec_long[3] = {"R-component of the magnetic field vector (3d version of BFieldR)",
        "Z-component of the magnetic field vector (3d version of BFieldZ)",
        "Contravariant Phi-component of the magnetic field vector (3d version of BFieldP)"};
    int vecID[3];
    err = nc_def_var( ncid, "BR", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[0]);
    err = nc_def_var( ncid, "BZ", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[1]);
    err = nc_def_var( ncid, "BP", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[2]);
    for(int i=0; i<3; i++)
    {
        err = nc_put_att_text( ncid, vecID[i], "long_name", vec_long[i].size(), vec_long[i].data());
        std::string coordinates = "zc yc xc";
        err = nc_put_att_text( ncid, vecID[i], "coordinates", coordinates.size(), coordinates.data());
    }
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, vecID[0], vecR.data());
    err = nc_put_var_double( ncid, vecID[1], vecZ.data());
    err = nc_put_var_double( ncid, vecID[2], vecP.data());
    err = nc_redef(ncid);
    vecR = dg::evaluate( dg::cooX3d, grid3d);
    vecZ = dg::evaluate( dg::cooZ3d, grid3d);
    vecP = dg::evaluate( dg::cooY3d, grid3d);
    for( unsigned i=0; i<vecR.size(); i++)
    {
        double xc = vecR[i]*sin(vecZ[i]);
        double yc = vecR[i]*cos(vecZ[i]);
        vecR[i] = xc;
        vecZ[i] = yc;
    }
    vec_long[0] = "x-coordinate in Cartesian coordinate system",
    vec_long[1] = "y-coordinate in Cartesian coordinate system",
    vec_long[2] = "z-coordinate in Cartesian coordinate system";
    err = nc_def_var( ncid, "xc", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[0]);
    err = nc_def_var( ncid, "yc", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[1]);
    err = nc_def_var( ncid, "zc", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[2]);
    for(int i=0; i<3; i++)
        err = nc_put_att_text( ncid, vecID[i], "long_name", vec_long[i].size(), vec_long[i].data());
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, vecID[0], vecR.data());
    err = nc_put_var_double( ncid, vecID[1], vecZ.data());
    err = nc_put_var_double( ncid, vecID[2], vecP.data());
    err = nc_redef(ncid);
    //////////////////////////////Finalize////////////////////////////////////
    err = nc_close(ncid);
    std::cout << "TEST ACCURACY OF CURVATURES (values must be close to 0!)\n";
    std::array<dg::HVec, 3> bhat, curvB, curvK;
    dg::pushForward( bhat_.x(), bhat_.y(), bhat_.z(),
            bhat[0], bhat[1], bhat[2], grid3d);
    std::array<dg::HVec, 3> bhat_covariant(bhat);
    dg::tensor::inv_multiply3d( grid3d.metric(), bhat[0], bhat[1], bhat[2],
            bhat_covariant[0], bhat_covariant[1], bhat_covariant[2]);
    dg::HVec normb( bhat[0]), one3d = dg::evaluate( dg::one, grid3d);
    dg::blas1::pointwiseDot( 1., bhat[0], bhat_covariant[0],
                             1., bhat[1], bhat_covariant[1],
                             0., normb);
    dg::blas1::pointwiseDot( 1., bhat[2], bhat_covariant[2],
                             1., normb);
    dg::blas1::axpby( 1., one3d, -1, normb);
    double error = sqrt(dg::blas1::dot( normb, normb));
    std::cout << "Error in norm b == 1 :  "<<error<<std::endl;

    dg::pushForward( curvB_.x(), curvB_.y(), curvB_.z(),
            curvB[0], curvB[1], curvB[2], grid3d);
    dg::pushForward( curvK_.x(), curvK_.y(), curvK_.z(),
            curvK[0], curvK[1], curvK[2], grid3d);
    dg::blas1::axpby( 1., curvK, -1., curvB);
    dg::HVec Bmodule = dg::pullback( dg::geo::Bmodule(mag), grid3d);
    dg::blas1::pointwiseDot( Bmodule, Bmodule, Bmodule);
    for( int i=0; i<3; i++)
        dg::blas1::pointwiseDot( Bmodule, curvB[i], curvB[i]);
    dg::HVec R = dg::pullback( dg::cooX3d, grid3d);
    dg::HVec IR =  dg::pullback( mag.ipolR(), grid3d);
    dg::blas1::pointwiseDivide( gp.R_0, IR, R, 0., IR);
    dg::HVec IZ =  dg::pullback( mag.ipolZ(), grid3d);
    dg::blas1::pointwiseDivide( gp.R_0, IZ, R, 0., IZ);
    dg::HVec IP =  dg::pullback( IPhi( gp), grid3d);
    dg::blas1::pointwiseDivide( gp.R_0, IP, R, 0., IP);
    dg::blas1::axpby( 1., IZ, -1., curvB[0]);
    dg::blas1::axpby( 1., IR, +1., curvB[1]);
    dg::blas1::axpby( 1., IP, -1., curvB[2]);
    for( int i=0; i<3; i++)
    {
        error = sqrt(dg::blas1::dot( curvB[i], curvB[i] ) );
        std::cout << "Error in curv "<<i<<" :   "<<error<<"\n";
    }

    return 0;
}
