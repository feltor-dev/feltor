#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "json/json.h"

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "file/nc_utilities.h"

#include "solovev.h"
#include "taylor.h"
#include "init.h"
#include "magnetic_field.h"
#include "average.h"

struct Parameters
{
    unsigned n, Nx, Ny, Nz;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;
    double amp, k_psi, bgprofamp, nprofileamp;
    double sigma, posX, posY;
    Parameters( const Json::Value& js){
        n = js["n"].asUInt();
        Nx = js["Nx"].asUInt();
        Ny = js["Ny"].asUInt();
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
    Json::Reader reader;
    Json::Value input_js, geom_js;
    if( argc == 4) 
    {
        newfilename = argv[3];
        std::cout << argv[0]<< " "<<argv[1]<<" & "<<argv[2]<<" -> " <<argv[3]<<std::endl;
        std::ifstream isI( argv[1]);
        std::ifstream isG( argv[2]);
        reader.parse( isI, input_js, false);
        reader.parse( isG, geom_js, false);
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
        reader.parse( input, input_js, false);
        reader.parse( geom, geom_js, false);
    }
    const Parameters p(input_js);
    const dg::geo::solovev::GeomParameters gp(geom_js);
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
 
    //construct all geometry quantities
    dg::geo::TokamakMagneticField c = dg::geo::createTaylorField(gp);
    const double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    const double Z_X = -1.1*gp.elongation*gp.a;
    const double R_H = gp.R_0-gp.triangularity*gp.a;
    const double Z_H = gp.elongation*gp.a;
    const double alpha_ = asin(gp.triangularity);
    const double N1 = -(1.+alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.+alpha_);
    const double N2 =  (1.-alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.-alpha_);
    const double N3 = -gp.elongation/(gp.a*cos(alpha_)*cos(alpha_));
    std::cout << "TEST ACCURACY OF PSI\n";
    std::cout << "psip( 1+e,0)           "<<c.psip()(gp.R_0 + gp.a, 0.)<<"\n";
    std::cout << "psip( 1-e,0)           "<<c.psip()(gp.R_0 - gp.a, 0.)<<"\n";
    std::cout << "psip( 1-de,ke)         "<<c.psip()(R_H, Z_H)<<"\n";
    std::cout << "psip( 1-1.1de,-1.1ke)  "<<c.psip()(R_X, Z_X)<<"\n";
    std::cout << "psipZ( 1+e,0)          "<<c.psipZ()(gp.R_0 + gp.a, 0.)<<"\n";
    std::cout << "psipZ( 1-e,0)          "<<c.psipZ()(gp.R_0 - gp.a, 0.)<<"\n";
    std::cout << "psipR( 1-de,ke)        "<<c.psipR()(R_H,Z_H)<<"\n";
    std::cout << "psipR( 1-1.1de,-1.1ke) "<<c.psipR()(R_X,Z_X)<<"\n";
    std::cout << "psipZ( 1-1.1de,-1.1ke) "<<c.psipZ()(R_X,Z_X)<<"\n";
    std::cout << "psipZZ( 1+e,0)         "<<c.psipZZ()(gp.R_0+gp.a,0.)+N1*c.psipR()(gp.R_0+gp.a,0)<<"\n";
    std::cout << "psipZZ( 1-e,0)         "<<c.psipZZ()(gp.R_0-gp.a,0.)+N2*c.psipR()(gp.R_0-gp.a,0)<<"\n";
    std::cout << "psipRR( 1-de,ke)       "<<c.psipRR()(R_H,Z_H)+N3*c.psipZ()(R_H,Z_H)<<"\n";

    //Feltor quantities
    dg::geo::InvB invB(c);
    dg::geo::BR bR(c);
    dg::geo::BZ bZ(c);
    dg::geo::CurvatureNablaBR curvatureR(c);
    dg::geo::CurvatureNablaBZ curvatureZ(c);
    dg::geo::GradLnB gradLnB(c);
    dg::geo::FieldR  field(c);
    dg::geo::FieldR fieldR(c);
    dg::geo::FieldZ fieldZ(c);
    dg::geo::FieldP fieldP(c);
    dg::geo::Iris iris( c.psip(), gp.psipmin, gp.psipmax );
    dg::geo::Pupil pupil(c.psip(), gp.psipmaxcut);
    dg::geo::GaussianDamping dampgauss(c.psip(), gp.psipmaxcut, gp.alpha);
    dg::geo::GaussianProfDamping dampprof(c.psip(),gp.psipmax, gp.alpha);
    dg::geo::ZonalFlow zonalflow(p.amp, p.k_psi, gp, c.psip());
    dg::geo::PsiLimiter psilimiter(c.psip(), gp.psipmaxlim);
    dg::geo::Nprofile prof(p.bgprofamp, p.nprofileamp, gp, c.psip());

    dg::BathRZ bath(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
//     dg::Gaussian3d bath(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
    dg::Gaussian3d blob(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
    dg::Grid2d grid2d(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny);

    std::vector<dg::HVec> hvisual(21);
        //allocate mem for visual
    std::vector<dg::HVec> visual(21);

    //B field functions
    hvisual[1] = dg::evaluate( c.psip(), grid2d);
    hvisual[2] = dg::evaluate( c.ipol(), grid2d);
    hvisual[3] = dg::evaluate( invB, grid2d);
    hvisual[4] = dg::evaluate( field, grid2d);
    hvisual[5] = dg::evaluate( curvatureR, grid2d);
    hvisual[6] = dg::evaluate( curvatureZ, grid2d);
    hvisual[7] = dg::evaluate( gradLnB, grid2d);
    //cut functions
    hvisual[8] = dg::evaluate( iris, grid2d);
    hvisual[9] = dg::evaluate( pupil, grid2d);
    hvisual[10] = dg::evaluate( dampprof, grid2d);
    hvisual[11] = dg::evaluate( dampgauss, grid2d);
    hvisual[12] = dg::evaluate( psilimiter, grid2d);
    //initial functions
    hvisual[13] = dg::evaluate( zonalflow, grid2d);
    hvisual[14] = dg::evaluate( prof, grid2d);
    hvisual[15] = dg::evaluate( blob, grid2d);
    hvisual[16] = dg::evaluate( bath, grid2d);
    //initial functions damped and with profile
    hvisual[17] = dg::evaluate( dg::one, grid2d);
    hvisual[18] = dg::evaluate( dg::one, grid2d);
    hvisual[19] = dg::evaluate( dg::one, grid2d);
    hvisual[20] = dg::evaluate( dg::one, grid2d);            
    dg::blas1::axpby( 1.,hvisual[16] , 1.,hvisual[14],hvisual[17]); //prof + bath
    dg::blas1::axpby( 1.,hvisual[13] , 1.,hvisual[14],hvisual[18]); //prof + zonal
    dg::blas1::axpby( 1.,hvisual[15] , 1.,hvisual[14],hvisual[19]); //prof + blob
    dg::blas1::plus(hvisual[17], -1); //to n -1
    dg::blas1::plus(hvisual[18], -1); //to n -1
    dg::blas1::plus(hvisual[19], -1); //to n -1
    dg::blas1::pointwiseDot(hvisual[10], hvisual[17], hvisual[17]); //damped 
    dg::blas1::pointwiseDot(hvisual[10], hvisual[18], hvisual[18]); //damped 
    dg::blas1::pointwiseDot(hvisual[10], hvisual[19], hvisual[19]); //damped 

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

    
    std::string names[] = { "", "psip", "ipol", "invB","invbf", "KR", 
                            "KZ", "gradLnB", "iris", "pupil", "dampprof", 
                            "damp", "lim",  "zonal", "prof", "blob", 
                            "bath", "ini1","ini2","ini3","ini4"};

    /////////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_create( newfilename.data(), NC_NETCDF4|NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    err = nc_put_att_text( ncid, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim1d_ids[1], dim2d_ids[2], dim3d_ids[3] ;
    err = file::define_dimension( ncid,"psi", &dim1d_ids[0], grid1d);
    dg::Grid3d grid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, n,Nx,Ny,Nz);
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
    for(unsigned i=1; i<hvisual.size(); i++)
    {
        int vectorID[1];
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 2, &dim2d_ids[0], &vectorID[0]);
        err = nc_enddef( ncid);
        err = nc_put_var_double( ncid, vectorID[0], hvisual[i].data());
        err = nc_redef(ncid);

    }
    //compute & write 3d vectors
    dg::HVec vecR = dg::evaluate( fieldR, grid3d);
    dg::HVec vecZ = dg::evaluate( fieldZ, grid3d);
    dg::HVec vecP = dg::evaluate( fieldP, grid3d);
    int vecID[3];
    err = nc_def_var( ncid, "BR", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[0]);
    err = nc_def_var( ncid, "BZ", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[1]);
    err = nc_def_var( ncid, "BP", NC_DOUBLE, 3, &dim3d_ids[0], &vecID[2]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, vecID[0], vecR.data());
    err = nc_put_var_double( ncid, vecID[1], vecZ.data());
    err = nc_put_var_double( ncid, vecID[2], vecP.data());
    err = nc_redef(ncid);
    //////////////////////////////Finalize////////////////////////////////////
    err = nc_close(ncid);


    return 0;
}
