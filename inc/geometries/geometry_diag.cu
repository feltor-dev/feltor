#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <functional>
#include <sstream>
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
    unsigned n, Nx, Ny, Nz;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;
    double amp, k_psi, bgprofamp, nprofileamp;
    double sigma, posX, posY;
    double rho_damping, alpha;
    Parameters( const Json::Value& js){
        n = js.get("n",3).asUInt();
        Nx = js.get("Nx",100).asUInt();
        Ny = js.get("Ny",100).asUInt();
        Nz = js.get("Nz", 1).asUInt();
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
        alpha = js.get("alpha", 0.05).asDouble();
    }
    void display( std::ostream& os = std::cout ) const
    {
        os << "Input parameters are: \n";
        os  <<" n             = "<<n<<"\n"
            <<" Nx            = "<<Nx<<"\n"
            <<" Ny            = "<<Ny<<"\n"
            <<" Nz            = "<<Nz<<"\n"
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
    dg::geo::TokamakMagneticField c = dg::geo::createSolovevField(gp);
    c = dg::geo::createModifiedSolovevField(gp, (1.-p.rho_damping)*c.psip()(c.R0(), 0.), p.alpha);
    const double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    const double Z_X = -1.1*gp.elongation*gp.a;
    const double R_H = gp.R_0-gp.triangularity*gp.a;
    const double Z_H = gp.elongation*gp.a;
    const double alpha_ = asin(gp.triangularity);
    const double N1 = -(1.+alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.+alpha_);
    const double N2 =  (1.-alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.-alpha_);
    const double N3 = -gp.elongation/(gp.a*cos(alpha_)*cos(alpha_));
    const double psip0 = c.psip()(gp.R_0, 0);
    std::cout << "psip( 1, 0) "<<psip0<<"\n";
    //Find O-point
    double R_O = gp.R_0, Z_O = 0.;
    dg::geo::findXpoint( c.get_psip(), R_O, Z_O);
    const double psipmin = c.psip()(R_O, Z_O);

    std::cout << "O-point "<<R_O<<" "<<Z_O<<" with Psip = "<<psipmin<<std::endl;

    std::cout << "TEST ACCURACY OF PSI (values must be close to 0!)\n";
    if( gp.hasXpoint())
        std::cout << "    Equilibrium with X-point!\n";
    else
        std::cout << "    NO X-point in flux function!\n";
    std::cout << "psip( 1+e,0)           "<<c.psip()(gp.R_0 + gp.a, 0.)<<"\n";
    std::cout << "psip( 1-e,0)           "<<c.psip()(gp.R_0 - gp.a, 0.)<<"\n";
    std::cout << "psip( 1-de,ke)         "<<c.psip()(R_H, Z_H)<<"\n";
    if( !gp.hasXpoint())
        std::cout << "psipR( 1-de,ke)        "<<c.psipR()(R_H, Z_H)<<"\n";
    else
    {
        std::cout << "psip( 1-1.1de,-1.1ke)  "<<c.psip()(R_X, Z_X)<<"\n";
        std::cout << "psipZ( 1+e,0)          "<<c.psipZ()(gp.R_0 + gp.a, 0.)<<"\n";
        std::cout << "psipZ( 1-e,0)          "<<c.psipZ()(gp.R_0 - gp.a, 0.)<<"\n";
        std::cout << "psipR( 1-de,ke)        "<<c.psipR()(R_H,Z_H)<<"\n";
        std::cout << "psipR( 1-1.1de,-1.1ke) "<<c.psipR()(R_X,Z_X)<<"\n";
        std::cout << "psipZ( 1-1.1de,-1.1ke) "<<c.psipZ()(R_X,Z_X)<<"\n";
    }
    std::cout << "psipZZ( 1+e,0)         "<<c.psipZZ()(gp.R_0+gp.a,0.)+N1*c.psipR()(gp.R_0+gp.a,0)<<"\n";
    std::cout << "psipZZ( 1-e,0)         "<<c.psipZZ()(gp.R_0-gp.a,0.)+N2*c.psipR()(gp.R_0-gp.a,0)<<"\n";
    std::cout << "psipRR( 1-de,ke)       "<<c.psipRR()(R_H,Z_H)+N3*c.psipZ()(R_H,Z_H)<<"\n";


    dg::HVec hvisual;
    //allocate mem for visual
    dg::HVec visual;
    std::map< std::string, std::function<double(double,double)>> map{
        {"Psip", c.psip()},
        {"Ipol", c.ipol()},
        {"Bmodule", dg::geo::Bmodule(c)},
        {"InvB", dg::geo::InvB(c)},
        {"LnB", dg::geo::LnB(c)},
        {"GradLnB", dg::geo::GradLnB(c)},
        {"Divb", dg::geo::Divb(c)},
        {"BR", dg::geo::BR(c)},
        {"BZ", dg::geo::BZ(c)},
        {"CurvatureNablaBR", dg::geo::CurvatureNablaBR(c)},
        {"CurvatureNablaBZ", dg::geo::CurvatureNablaBZ(c)},
        {"CurvatureKappaR", dg::geo::CurvatureKappaR(c)},
        {"CurvatureKappaZ", dg::geo::CurvatureKappaZ(c)},
        {"DivCurvatureKappa", dg::geo::DivCurvatureKappa(c)},
        {"DivCurvatureNablaB", dg::geo::DivCurvatureNablaB(c)},
        {"TrueCurvatureNablaBR", dg::geo::TrueCurvatureNablaBR(c)},
        {"TrueCurvatureNablaBZ", dg::geo::TrueCurvatureNablaBZ(c)},
        {"TrueCurvatureNablaBP", dg::geo::TrueCurvatureNablaBP(c)},
        {"TrueCurvatureKappaR", dg::geo::TrueCurvatureKappaR(c)},
        {"TrueCurvatureKappaZ", dg::geo::TrueCurvatureKappaZ(c)},
        {"TrueCurvatureKappaP", dg::geo::TrueCurvatureKappaP(c)},
        {"TrueDivCurvatureKappa", dg::geo::TrueDivCurvatureKappa(c)},
        {"TrueDivCurvatureNablaB", dg::geo::TrueDivCurvatureNablaB(c)},
        {"BFieldR", dg::geo::BFieldR(c)},
        {"BFieldZ", dg::geo::BFieldZ(c)},
        {"BFieldP", dg::geo::BFieldP(c)},
        {"BHatR", dg::geo::BHatR(c)},
        {"BHatZ", dg::geo::BHatZ(c)},
        {"BHatP", dg::geo::BHatP(c)},
        {"GradBHatR", dg::geo::BHatR(c)},
        {"GradBHatZ", dg::geo::BHatZ(c)},
        {"GradBHatP", dg::geo::BHatP(c)},
        //////////////////////////////////
        {"Iris", dg::geo::Iris(c.psip(), gp.psipmin, gp.psipmax)},
        {"Pupil", dg::geo::Pupil(c.psip(), gp.psipmaxcut)},
        {"GaussianDamping", dg::geo::GaussianDamping(c.psip(), gp.psipmaxcut, gp.alpha)},
        {"ZonalFlow", dg::geo::ZonalFlow(c.psip(), p.amp, 0., 2.*M_PI*p.k_psi )},
        {"PsiLimiter", dg::geo::PsiLimiter(c.psip(), gp.psipmaxlim)},
        {"Nprofile", dg::geo::Nprofile(c.psip(), p.nprofileamp/c.psip()(c.R0(),0.), p.bgprofamp )},
        {"Delta", dg::geo::DeltaFunction( c, gp.alpha*gp.alpha, psip0*0.2)},
        {"TanhDamping", dg::geo::TanhDamping(c.psip(), -3*gp.alpha, gp.alpha, -1)},
        ////
        {"BathRZ", dg::BathRZ( 16, 16, Rmin,Zmin, 30.,5., p.amp)},
        {"Gaussian3d", dg::Gaussian3d(gp.R_0+p.posX*gp.a, p.posY*gp.a,
            M_PI, p.sigma, p.sigma, p.sigma, p.amp)}
    };
    dg::Grid2d grid2d(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny);
    dg::DVec psipog2d   = dg::evaluate( c.psip(), grid2d);
    std::map<std::string, dg::HVec> map1d;
    ///////////TEST CURVILINEAR GRID TO COMPUTE FSA QUANTITIES
    unsigned npsi = 3, Npsi = 64;//set number of psivalues (NPsi % 8 == 0)
    double psipmax = dg::blas1::reduce( psipog2d, 0. ,thrust::maximum<double>()); //DEPENDS ON GRID RESOLUTION!!
    double volumeXGrid;
    if( gp.hasXpoint())
    {
        std::cout << "Generate X-point flux-aligned grid!\n";
        double RX = gp.R_0-1.1*gp.triangularity*gp.a;
        double ZX = -1.1*gp.elongation*gp.a;
        dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( c.get_psip(), RX, ZX) ;
        dg::geo::SeparatrixOrthogonal generator(c.get_psip(), monitor_chi, psipmin, R_X, Z_X, c.R0(), 0, 0, true);
        double fx_0 = 1./8.;
        psipmax = -fx_0/(1.-fx_0)*psipmin;
        std::cout << "psi 1 is          "<<psipmax<<"\n";
        dg::geo::CurvilinearGridX2d gX2d( generator, fx_0, 0., npsi, Npsi, 160, dg::DIR, dg::NEU);
        std::cout << "DONE! Generate X-point flux-aligned grid!\n";
        dg::Average<dg::HVec > avg_eta( gX2d.grid(), dg::coo2d::y);
        std::vector<dg::HVec> coordsX = gX2d.map();
        dg::SparseTensor<dg::HVec> metricX = gX2d.metric();
        dg::HVec volX2d = dg::tensor::volume2d( metricX);
        dg::blas1::pointwiseDot( coordsX[0], volX2d, volX2d); //R\sqrt{g}
        dg::IHMatrix grid2gX2d = dg::create::interpolation( coordsX[0], coordsX[1], grid2d);
        dg::HVec psip_X = dg::evaluate( dg::one, gX2d), area_psip_X;
        dg::blas2::symv( grid2gX2d, (dg::HVec)psipog2d, psip_X);
        dg::blas1::pointwiseDot( volX2d, psip_X, psip_X);
        avg_eta( psip_X, map1d["X_psip1d"], false);
        avg_eta( volX2d, area_psip_X, false);
        dg::Grid1d gX1d( gX2d.x0(), gX2d.x1(), npsi, Npsi, dg::NEU);
        map1d["X_psi_vol"] = dg::integrate( area_psip_X, gX1d);
        dg::blas1::scal( map1d.at("X_psi_vol"), 4.*M_PI*M_PI);
        dg::blas1::pointwiseDivide( map1d.at("X_psip1d"), area_psip_X,
            map1d.at("X_psip1d"));

        //NOTE: VOLUME is WITHIN cells while AREA is ON gridpoints
        dg::HVec gradPsipX = metricX.value(0,0);
        dg::blas1::transform( gradPsipX, gradPsipX, dg::SQRT<double>());
        dg::blas1::pointwiseDot( volX2d, gradPsipX, gradPsipX); //R\sqrt{g}|\nabla\zeta|
        avg_eta( gradPsipX, map1d["X_psi_area"], false);
        dg::blas1::scal( map1d.at("X_psi_area"), 4.*M_PI*M_PI);
        volumeXGrid = dg::interpolate( map1d.at("X_psi_vol"), gX1d.x1(), gX1d);
    }

    ///////////////////Compute flux average////////////////////
    dg::HVec xpoint_weights = dg::evaluate( dg::cooX2d, grid2d);
    if( gp.hasXpoint() )
        dg::blas1::pointwiseDot( xpoint_weights , dg::evaluate( dg::geo::ZCutter(Z_X), grid2d), xpoint_weights);
    dg::geo::SafetyFactor     qprof( c);
    dg::geo::FluxSurfaceAverage<dg::DVec>  fsa( grid2d, c, psipog2d, xpoint_weights);
    dg::Grid1d grid1d(psipmin, psipmax, npsi ,Npsi,dg::NEU);
    map1d["psi_fsa"]    = dg::evaluate( fsa,        grid1d);
    map1d["q-profile"]  = dg::evaluate( qprof,      grid1d);
    map1d["psip1d"]     = dg::evaluate( dg::cooX1d, grid1d);
    map1d["rho"] = map1d.at("psip1d");
    dg::blas1::axpby( -1./psipmin, map1d.at("rho"), +1., 1.,
        map1d.at("rho")); //transform psi to rho

    //other flux labels
    dg::geo::FluxSurfaceIntegral<dg::HVec> fsi( grid2d, c);
    fsi.set_right( xpoint_weights);

    dg::HVec areaT_psip = dg::evaluate( fsi, grid1d);
    dg::HVec w1d = dg::create::weights( grid1d);
    double volumeCoarea = 2.*M_PI*dg::blas1::dot( areaT_psip, w1d);

    //area
    fsi.set_left( dg::evaluate( dg::geo::GradPsip(c), grid2d));
    map1d["psi_area"] = dg::evaluate( fsi, grid1d);
    dg::blas1::scal( map1d.at("psi_area"), 2.*M_PI);

    dg::geo::FluxVolumeIntegral<dg::HVec> fvi( (dg::CartesianGrid2d)grid2d, c);
    std::cout << "Delta Rho for Flux surface integrals = "<<-fsi.get_deltapsi()/psipmin<<"\n";

    fvi.set_right( xpoint_weights);
    map1d["psi_vol"] = dg::evaluate( fvi, grid1d);
    dg::blas1::scal(map1d["psi_vol"], 2.*M_PI);
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
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    err = nc_put_att_text( ncid, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim1d_ids[1], dim2d_ids[2], dim3d_ids[3] ;
    err = file::define_dimension( ncid,"psi", &dim1d_ids[0], grid1d);
    dg::CylindricalGrid3d grid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, n,Nx,Ny,Nz);
    err = file::define_dimensions( ncid, &dim3d_ids[0], grid3d);
    dim2d_ids[0] = dim3d_ids[1], dim2d_ids[1] = dim3d_ids[2];

    //write 1d vectors
    for( auto pair : map1d)
    {
        int vid;
        err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 1,
            &dim1d_ids[0], &vid);
        err = nc_enddef( ncid);
        err = nc_put_var_double( ncid, vid, pair.second.data());
        err = nc_redef(ncid);
    }
    //write 2d vectors
    for(auto pair : map)
    {
        int vectorID;
        err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 2,
            &dim2d_ids[0], &vectorID);
        err = nc_enddef( ncid);
        hvisual = dg::evaluate( pair.second, grid2d);
        err = nc_put_var_double( ncid, vectorID, hvisual.data());
        err = nc_redef(ncid);

    }
    //compute & write 3d vectors
    dg::HVec vecR = dg::evaluate( dg::geo::BFieldR(c), grid3d);
    dg::HVec vecZ = dg::evaluate( dg::geo::BFieldZ(c), grid3d);
    dg::HVec vecP = dg::evaluate( dg::geo::BFieldP(c), grid3d);
    int vecID[3];
    err = nc_def_var( ncid, "B_R", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[0]);
    err = nc_def_var( ncid, "B_Z", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[1]);
    err = nc_def_var( ncid, "B_P", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[2]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, vecID[0], vecR.data());
    err = nc_put_var_double( ncid, vecID[1], vecZ.data());
    err = nc_put_var_double( ncid, vecID[2], vecP.data());
    err = nc_redef(ncid);
    //////////////////////////////Finalize////////////////////////////////////
    err = nc_close(ncid);
    std::cout << "TEST ACCURACY OF CURVATURES (values must be close to 0!)\n";
    dg::geo::CylindricalVectorLvl0 bhat_ = dg::geo::createBHat( c);
    dg::geo::CylindricalVectorLvl0 curvB_ = dg::geo::createTrueCurvatureNablaB( c);
    dg::geo::CylindricalVectorLvl0 curvK_ = dg::geo::createTrueCurvatureKappa( c);
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
    dg::HVec Bmodule = dg::pullback( dg::geo::Bmodule(c), grid3d);
    dg::blas1::pointwiseDot( Bmodule, Bmodule, Bmodule);
    for( int i=0; i<3; i++)
        dg::blas1::pointwiseDot( Bmodule, curvB[i], curvB[i]);
    dg::HVec R = dg::pullback( dg::cooX3d, grid3d);
    dg::HVec IR =  dg::pullback( c.ipolR(), grid3d);
    dg::blas1::pointwiseDivide( gp.R_0, IR, R, 0., IR);
    dg::HVec IZ =  dg::pullback( c.ipolZ(), grid3d);
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
