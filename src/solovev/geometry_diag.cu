#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "file/read_input.h"
#include "file/nc_utilities.h"

#include "geometry.h"
#include "init.h"

int main( int argc, char* argv[])
{
    std::vector<double> v, v2;
    std::string input, geom, newfilename;
    if( !(argc == 4 || argc == 3))
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] [outputfile]\n";
        std::cerr << "Or \n Usage: "<< argv[0]<<" [file.nc] [outputfile.nc]\n";
        return -1;
    }
    try{ 
        if( argc == 4) 
        {
            newfilename = argv[3];
            std::cout << argv[0]<< " "<<argv[1]<<" & "<<argv[2]<<" -> " <<argv[3]<<std::endl;
            input = file::read_file( argv[1]);
            geom = file::read_file( argv[2]);
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
            size_t length;
            err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
            input.resize( length, 'x');
            err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
            err = nc_inq_attlen( ncid, NC_GLOBAL, "geomfile", &length);
            geom.resize( length, 'x');
            err = nc_get_att_text( ncid, NC_GLOBAL, "geomfile", &geom[0]);
            nc_close( ncid);
        }

    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;
    }
    //write parameters from file into variables
    std::cout << input << std::endl;
    std::cout << geom << std::endl;
    const eule::Parameters p(file::read_input( input));
    const solovev::GeomParameters gp(file::read_input( geom));
    p.display( std::cout);
    gp.display( std::cout);
    unsigned n, Nx, Ny, Nz;
    n = p.n, Nx = p.Nx, Ny = p.Ny, Nz = p.Nz;
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //old boxscale
//     double Rmin=gp.R_0-p.boxscaleRp*gp.a;
//     double Zmin=-p.boxscaleRp*gp.a*gp.elongation;
//     double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
//     double Zmax=p.boxscaleRp*gp.a*gp.elongation;
 
    //construct all geometry quantities
    solovev::Psip psip(gp);
    solovev::PsipR psipR(gp);
    solovev::PsipRR psipRR(gp);  
    solovev::PsipZ psipZ(gp);  
    solovev::PsipZZ psipZZ(gp);   
    solovev::PsipRZ psipRZ(gp);  
    solovev::Ipol ipol(gp);
    solovev::InvB invB(gp);
    solovev::BR bR(gp);
    solovev::BZ bZ(gp);

    //Feltor quantities
    solovev::CurvatureR curvatureR(gp);
    solovev::CurvatureZ curvatureZ(gp);
    solovev::GradLnB gradLnB(gp);
    solovev::Field field(gp);
    solovev::FieldR fieldR(gp);
    solovev::FieldZ fieldZ(gp);
    solovev::FieldP fieldP(gp);
    solovev::Iris iris(gp);
    solovev::Pupil pupil(gp);
    solovev::GaussianDamping dampgauss(gp);
    solovev::GaussianProfDamping dampprof(gp);
    solovev::ZonalFlow zonalflow(p, gp);
    solovev::PsiLimiter psilimiter(gp);
    solovev::Nprofile prof(p, gp);

    dg::BathRZ bath(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
//     dg::Gaussian3d bath(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
    dg::Gaussian3d blob(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
    dg::Grid2d<double> grid2d(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny);

    std::vector<dg::HVec> hvisual(21);
        //allocate mem for visual
    std::vector<dg::HVec> visual(21);

    //B field functions
    hvisual[1] = dg::evaluate( psip, grid2d);
    hvisual[2] = dg::evaluate( ipol, grid2d);
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
    dg::blas1::transform(hvisual[17], hvisual[17], dg::PLUS<>(-1)); //to n -1
    dg::blas1::transform(hvisual[18], hvisual[18], dg::PLUS<>(-1)); //to n -1
    dg::blas1::transform(hvisual[19], hvisual[19], dg::PLUS<>(-1)); //to n -1
    dg::blas1::pointwiseDot(hvisual[10], hvisual[17], hvisual[17]); //damped 
    dg::blas1::pointwiseDot(hvisual[10], hvisual[18], hvisual[18]); //damped 
    dg::blas1::pointwiseDot(hvisual[10], hvisual[19], hvisual[19]); //damped 

    //Compute flux average
    solovev::Alpha alpha(gp); // = B^phi / |nabla psip |
    dg::DVec psipog2d   = dg::evaluate( psip, grid2d);
    dg::DVec alphaog2d  = dg::evaluate( alpha, grid2d); 
    double psipmin = (float)thrust::reduce( psipog2d .begin(), psipog2d .end(), 0.0,thrust::minimum<double>()  );
    unsigned npsi = 3, Npsi = 50;//set number of psivalues
    psipmin += (gp.psipmax - psipmin)/(double)Npsi; //the inner value is not good
    dg::Grid1d<double> grid1d(psipmin , gp.psipmax, npsi ,Npsi,dg::DIR);
    solovev::SafetyFactor<dg::HVec>     qprof(grid2d, gp, alphaog2d );
    dg::HVec sf         = dg::evaluate( qprof,    grid1d);
    dg::HVec abs        = dg::evaluate( dg::coo1, grid1d);

    
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
    dg::Grid3d<double> grid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, n,Nx,Ny,Nz);
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
