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
#include "average.h"
#include "testfunctors.h"

struct Parameters
{
    unsigned n, Nx, Ny, Nz;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;
    double amp, k_psi, bgprofamp, nprofileamp;
    double sigma, posX, posY;
    Parameters( const Json::Value& js){
        n = js.get("n",3).asUInt();
        Nx = js.get("Nx",100).asUInt();
        Ny = js.get("Ny",100).asUInt();
        Nz = js.get("Nz", 1).asUInt();
        boxscaleRm = js.get("boxscaleRm", 1.).asDouble();
        boxscaleRp = js.get("boxscaleRp", 1.).asDouble();
        boxscaleZm = js.get("boxscaleZm", 1.3).asDouble();
        boxscaleZp = js.get("boxscaleZp", 1.).asDouble();
        amp = js.get("amplitude", 1.).asDouble();
        k_psi = js.get("k_psi", 1.).asDouble();
        bgprofamp = js.get("bgprofamp", 1.).asDouble();
        nprofileamp = js.get("nprofileamp", 1.).asDouble();
        sigma = js.get("sigma", 10).asDouble();
        posX = js.get("posX", 0.5).asDouble();
        posY = js.get("posY", 0.5).asDouble();
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
    const double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    const double Z_X = -1.1*gp.elongation*gp.a;
    const double R_H = gp.R_0-gp.triangularity*gp.a;
    const double Z_H = gp.elongation*gp.a;
    const double alpha_ = asin(gp.triangularity);
    const double N1 = -(1.+alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.+alpha_);
    const double N2 =  (1.-alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.-alpha_);
    const double N3 = -gp.elongation/(gp.a*cos(alpha_)*cos(alpha_));
    bool Xpoint = false;
    for( int i=7; i<12; i++)
        if( gp.c[i] != 0)
            Xpoint = true;
    std::cout << "TEST ACCURACY OF PSI (values must be close to 0!)\n";
    if( Xpoint)
        std::cout << "    Equilibrium with X-point!\n";
    else
        std::cout << "    NO X-point in flux function!\n";
    std::cout << "psip( 1+e,0)           "<<c.psip()(gp.R_0 + gp.a, 0.)<<"\n";
    std::cout << "psip( 1-e,0)           "<<c.psip()(gp.R_0 - gp.a, 0.)<<"\n";
    std::cout << "psip( 1-de,ke)         "<<c.psip()(R_H, Z_H)<<"\n";
    if( !Xpoint)
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

    dg::Grid2d grid2d(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny);

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
        {"TanhDamping", dg::geo::TanhDamping(c.psip(), gp.psipmin, gp.alpha)},
        ////
        {"BathRZ", dg::BathRZ( 16, 16, Rmin,Zmin, 30.,5., p.amp)},
        {"Gaussian3d", dg::Gaussian3d(gp.R_0+p.posX*gp.a, p.posY*gp.a,
            M_PI, p.sigma, p.sigma, p.sigma, p.amp)}
    };
    //Compute flux average
    dg::geo::Alpha alpha(c); // = B^phi / |nabla psip |
    dg::DVec psipog2d   = dg::evaluate( c.psip(), grid2d);
    dg::DVec alphaog2d  = dg::evaluate( alpha, grid2d);
    double psipmin = (float)thrust::reduce( psipog2d .begin(), psipog2d .end(), 0.0,thrust::minimum<double>()  );
    unsigned npsi = 3, Npsi = 150;//set number of psivalues
    psipmin += (gp.psipmax - psipmin)/(double)Npsi; //the inner value is not good
    dg::Grid1d grid1d(psipmin , gp.psipmax, npsi ,Npsi,dg::DIR);
    dg::geo::SafetyFactor< dg::DVec>     qprof(grid2d, c, alphaog2d );
    dg::HVec sf         = dg::evaluate( qprof,    grid1d);
    dg::HVec abs        = dg::evaluate( dg::cooX1d, grid1d);



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
    int avgID[2];
    err = nc_def_var( ncid, "q-profile", NC_DOUBLE, 1, &dim1d_ids[0], &avgID[0]);
    err = nc_def_var( ncid, "psip1d", NC_DOUBLE, 1, &dim1d_ids[0], &avgID[1]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, avgID[0], sf.data());
    err = nc_put_var_double( ncid, avgID[1], abs.data());
    err = nc_redef(ncid);

    //write 2d vectors
    for(auto pair : map)
    {
        int vectorID;
        err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 2, &dim2d_ids[0], &vectorID);
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
