
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>

#include <cusp/print.h>

// #define DG_DEBUG


#include "file/read_input.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/xspacelib.cuh"
#include "geometry.h"
#include "init.h"
#include "dg/backend/dz.cuh"
#include "dg/algorithm.h"
#include "dg/poisson.h"
#include "dg/backend/functions.h"
#include "dg/backend/interpolation.cuh"
#include "draw/host_window.h"
#include "geom_parameters.h"
#include "file/nc_utilities.h"
struct InvNormR
{
    InvNormR( solovev::GeomParameters gp): R_0(gp.R_0){}
    double operator()( double R, double Z, double phi)
    {
        return R_0/R;
    }
    
    private:
    double R_0;
}; 

int main( int argc, char* argv[])
{
    double Rmin,Zmin,Rmax,Zmax;
//     double A,R_0,a, elongation;
//     double psipmin,psipmax;
    unsigned n, Nx, Ny, Nz;

    std::vector<double> c(13);
    //read and store geom data
       std::vector<double> v, v2;
    try{ 
        if( argc==1)
        {
            v = file::read_input( "geometry_params.txt"); 
            v2 = file::read_input( "../feltor/input.txt");

        }
        else
        {
            v = file::read_input( argv[1]); 
            v2 = file::read_input( argv[2]);
        }
    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;}

    //write parameters from file into variables
    const solovev::GeomParameters gp(v);
    const eule::Parameters p(v2);
    p.display( std::cout);
    gp.display( std::cout);

    Rmin=gp.R_0-(p.boxscale)*gp.a;
    Zmin=-(p.boxscale)*gp.a*gp.elongation;
    Rmax=gp.R_0+(p.boxscale)*gp.a; 
    Zmax=(p.boxscale)*gp.a*gp.elongation;
    std::cout << "The grid parameters" <<"\n";
    std::cout  << Rmin<<" rho_s " << Rmax <<" rho_s " << Zmin <<" rho_s " <<Zmax <<" rho_s " <<"\n";
    std::cout << "Type n, Nx, Ny, Nz\n";
    std::cin >> n>> Nx>>Ny>>Nz;
        
    solovev::Field field(gp);
    solovev::Psip psip(gp);
    solovev::PsipR psipR(gp);
    solovev::PsipRR psipRR(gp);  
    solovev::PsipZ psipZ(gp);  
    solovev::PsipZZ psipZZ(gp);   
    solovev::PsipRZ psipRZ(gp);  
    solovev::Ipol ipol(gp);
    solovev::InvB invB(gp);
    solovev::LnB lnB(gp);
    solovev::BR bR(gp);
    solovev::BZ bZ(gp);
    solovev::CurvatureR curvatureR(gp);
    solovev::CurvatureZ curvatureZ(gp);
    solovev::GradLnB gradLnB(gp);
    solovev::Pupil pupil(gp);
    InvNormR invnormr(gp);
    solovev::FieldR fieldR(gp);
    solovev::FieldZ fieldZ(gp);
    solovev::FieldP fieldP(gp);
    solovev::BHatR bhatR(gp);
    solovev::BHatZ bhatZ(gp);
    solovev::BHatP bhatP(gp);
    dg::Grid3d<double> grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,n, Nx, Ny,Nz);
    dg::HVec vecR = dg::evaluate( fieldR, grid);
    dg::HVec vecZ = dg::evaluate( fieldZ, grid);
    dg::HVec vecP = dg::evaluate( fieldP, grid);

    file::NC_Error_Handle err;
    int ncid, dim_ids[3];
//     err = nc_create( "geometry.nc", NC_CLOBBER, &ncid);
    err = nc_create( "geometry.nc", NC_NETCDF4|NC_CLOBBER, &ncid);

    err = file::define_dimensions( ncid, dim_ids, grid);
    int vecID[3];
    err = nc_def_var( ncid, "BR", NC_DOUBLE, 3, dim_ids, &vecID[0]);
    err = nc_def_var( ncid, "BZ", NC_DOUBLE, 3, dim_ids, &vecID[1]);
    err = nc_def_var( ncid, "BP", NC_DOUBLE, 3, dim_ids, &vecID[2]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, vecID[0], vecR.data());
    err = nc_put_var_double( ncid, vecID[1], vecZ.data());
    err = nc_put_var_double( ncid, vecID[2], vecP.data());
    nc_close(ncid);
    std::cout << "-----(0) Check single field by integrating from 0 to 2pi (psi=0 surface)" << "\n";
    thrust::host_vector<double>  in(3);
    thrust::host_vector<double>  out(3);
    in[0]=gp.R_0+gp.a; 
    in[1]=0.0;
    in[2]=0.;
    dg::integrateRK4( field, in, out,  2*M_PI, gp.rk4eps);
    
    std::cout <<"Rin =  "<< in[0] <<" Zin =  "<<in[1] <<" sin  = "<<in[2]<<"\n";
    std::cout <<"Rout = "<< out[0]<<" Zout = "<<out[1]<<" sout = "<<out[2]<<"\n";


    
    for (unsigned k=1;k<2;k++) //n iterator
    {
        for (unsigned i=0;i<1;i++) //Nxy iterator
        {
            std::stringstream ss1,ss2;
            ss1 << "dzerr1n" <<k*n<<"Nxy"<<pow(2,i)* Nx<<".txt";
            ss2 << "dzerr2n" <<k*n<<"Nxy"<<pow(2,i)* Nx<<".txt";
            std::string dzerr1fn = ss1.str();
            std::string dzerr2fn = ss2.str();
    //         std::cout << dzerr1fn;
            std::ofstream dzerrfile1((char *) dzerr1fn.c_str());
            std::ofstream dzerrfile2((char *) dzerr2fn.c_str());
            for (unsigned zz=0;zz<1;zz++) //Nz iterator
            {
                std::cout << "n = " << k*n << " Nx = " <<pow(2,i)* Nx << " Ny = " <<pow(2,i)* Ny << " Nz = "<<pow(2,zz)* Nz <<"\n";
                //Similar to feltor grid
                dg::Grid3d<double> g3d( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,k*n,pow(2,i)* Nx,pow(2,i)* Ny, pow(2,zz)*Nz,dg::NEU, dg::NEU, dg::PER, dg::cylindrical);
                const dg::DVec w3d = dg::create::weights( g3d);
                dg::DVec pupilongrid = dg::evaluate( pupil, g3d);

                std::cout <<"---------------------------------------------------------------------------------------------" << "\n";


                std::cout <<"-----(1a) test with testfunction" << "\n";
                solovev::TestFunction func(gp);
                solovev::DeriTestFunction derifunc(gp);
                std::cout << "Construct parallel  derivative\n";
                dg::Timer t;
                t.tic();
                dg::DZ<dg::DMatrix, dg::DVec> dz( field, g3d,gp.rk4eps,solovev::PsiLimiter(gp), g3d.bcx()); //choose bc of grid
                t.toc();
                std::cout << "-----> Creation of parallel Derivative took "<<t.diff()<<"s\n";

                dg::DVec function = dg::evaluate( func, g3d),dzfunc(function);
                dg::DVec diff(g3d.size());

                dg::DVec solution = dg::evaluate( derifunc, g3d);
                dz( function, dzfunc);
                //cut boundaries
//                 dg::blas1::pointwiseDot( pupilongrid, dzfunc, dzfunc);  //damped dzfunc
//                 dg::blas1::pointwiseDot( pupilongrid, solution, solution); //damped dzsol

                dg::blas1::axpby( 1., solution, -1., dzfunc,diff);
                double normdz = dg::blas2::dot( w3d, dzfunc);
                std::cout << "Norm dz  = "<<sqrt( normdz)<<"\n";
                double normsol = dg::blas2::dot( w3d,solution);
                std::cout << "Norm sol = "<<sqrt( normsol)<<"\n";
                double normdiff = dg::blas2::dot( w3d, diff);
                double reldiff=sqrt( normdiff/normsol );
                std::cout << "Rel Diff = "<< reldiff<<"\n";
                  std::cout <<"-----(1b) test parallel derivative created brackets with testfunction" << "\n";
//                 solovev::TestFunction func(psip);
//                 solovev::DeriTestFunction derifunc(gp,psip,psipR,psipZ,ipol,invB);
                std::cout << "-----> Construct parallel  derivative\n";
                t.tic();
                dg::DVec dzRZPhifunction(g3d.size());
                dg::DVec dzR(g3d.size());
                dg::DVec dzZ(g3d.size());
                dg::DVec dzPHI(g3d.size());
                dg::DVec BvecR = dg::evaluate( bhatR, grid);
                dg::DVec BvecZ = dg::evaluate( bhatZ, grid);
                dg::DVec BvecPHI = dg::evaluate( bhatP, grid);
//                 dg::DVec invBfordz = dg::evaluate( invB, grid);

                dg::DMatrix dR   =dg::create::dx( g3d, g3d.bcx(),dg::normed,dg::forward);
                dg::DMatrix dZ   =dg::create::dy( g3d, g3d.bcy(),dg::normed,dg::forward);
                dg::DMatrix dPHI =dg::create::dz( g3d, g3d.bcz(),dg::centered);
                
                dg::blas2::symv( dR, function, dzR);  
                dg::blas2::symv( dZ, function,   dzZ); 
                dg::blas2::symv( dPHI,function, dzPHI); 
                dg::blas1::pointwiseDot(BvecR ,dzR,dzR); //BR*dR f
                dg::blas1::pointwiseDot(BvecZ ,dzZ,dzZ); //BZ*dZ f
                dg::blas1::pointwiseDot(BvecPHI ,dzPHI,dzPHI);//Bphi*dphi f
                
                dg::blas1::axpby(1.,dzR,1.,dzZ,dzRZPhifunction); //BR*dR f + BZ*dZ f
                dg::blas1::axpby(1.,dzPHI,1.,dzRZPhifunction,dzRZPhifunction); //BR*dR f + BZ*dZ f+Bphi*dphi f
//                 dg::blas1::pointwiseDot(invBfordz,dzRZPhifunction,dzRZPhifunction);//1/B (BR*dR f + BZ*dZ f+Bphi*dphi f)
                t.toc();

                std::cout << "-----> Creation of parallel Derivative took "<<t.diff()<<"s\n";

                dg::DVec diffRZPhi(g3d.size());

                //cut boundaries
//                 dg::blas1::pointwiseDot( pupilongrid,dzRZPhifunction, dzRZPhifunction);  //damped dzfunc
//                 dg::blas1::pointwiseDot( pupilongrid, solution, solution); //damped dzsol

                dg::blas1::axpby( 1., solution, -1., dzRZPhifunction, diffRZPhi);
                double normdzRZPhi = dg::blas2::dot( w3d,  dzRZPhifunction);
                std::cout << "Norm dzRZPhi  = "<<sqrt( normdzRZPhi)<<"\n";
                std::cout << "Norm sol = "<<sqrt( normsol)<<"\n";
                double normdiffRZPhi = dg::blas2::dot( w3d, diffRZPhi);
                double reldiffRZPhi=sqrt( normdiffRZPhi/normsol );
                std::cout << "Rel Diff = "<< reldiffRZPhi<<"\n";
                
                std::cout <<"---------------------------------------------------------------------------------------------" << "\n";
                std::cout <<"-----(2a) test with gradlnb" << "\n";    
                dg::DVec gradLnBsolution = dg::evaluate( gradLnB, g3d);
                dg::DVec lnBongrid = dg::evaluate( lnB, g3d);
                dg::DVec dzlnBongrid(g3d.size());
                dg::DVec diff2(g3d.size());
                dg::DVec pupilongradLnBsolution(gradLnBsolution);

                dz(lnBongrid,dzlnBongrid);
                
                //cut boundaries
//                 dg::blas1::pointwiseDot( pupilongrid, dzlnBongrid, dzlnBongrid); 
//                 dg::blas1::pointwiseDot( pupilongrid, gradLnBsolution, pupilongradLnBsolution); 

                dg::blas1::axpby( 1., gradLnBsolution , -1., dzlnBongrid,diff2); //diff = gradlnB - dz(ln(B))
                //cut boundaries

                double normdz2 = dg::blas2::dot( w3d, dzlnBongrid); //=  Integral (gdz(ln(B))^2 )
                std::cout << "Norm dz  = "<<sqrt( normdz2)<<"\n";
                double normsol2 = dg::blas2::dot( w3d,gradLnBsolution);//=  Integral (gradlnB^2 )
                std::cout << "Norm sol = "<<sqrt( normsol2)<<"\n";
                double normdiff2=dg::blas2::dot( w3d, diff2); //=  Integral ((gradlnB - dz(ln(B)))^2)
                double reldiff2 =sqrt( normdiff2/normsol2 ); ;//=  sqrt(Integral ((gradlnB - dz(ln(B)))^2)/Integral (gradlnB^2 ))
                std::cout << "Rel Diff = "<<reldiff2 <<"\n";
                       std::cout <<"-----(2b) test with gradlnb" << "\n";    
                dg::DVec dzRlnB(g3d.size());
                dg::DVec dzZlnB(g3d.size());
                dg::DVec dzPHIlnB(g3d.size());
                dg::DVec dzRZPHIlnB(g3d.size());

                dg::DVec diff2b(g3d.size());
                dg::blas2::symv( dR, lnBongrid, dzRlnB);  
                dg::blas2::symv( dZ, lnBongrid,   dzZlnB); 
                dg::blas2::symv( dPHI,lnBongrid, dzPHIlnB); 
                dg::blas1::pointwiseDot(BvecR ,dzRlnB,dzRlnB); //BR*dR f
                dg::blas1::pointwiseDot(BvecZ ,dzZlnB,dzZlnB); //BZ*dZ f
                dg::blas1::pointwiseDot(BvecPHI ,dzPHIlnB,dzPHIlnB);//Bphi*dphi f
                
                dg::blas1::axpby(1.,dzRlnB,1.,dzZlnB,dzRZPHIlnB); //BR*dR f + BZ*dZ f
                dg::blas1::axpby(1.,dzPHIlnB,1.,dzRZPHIlnB,dzRZPHIlnB); //BR*dR f + BZ*dZ f+Bphi*dphi 
                
                //cut boundaries
//                 dg::blas1::pointwiseDot( pupilongrid, dzlnBongrid, dzlnBongrid); 
//                 dg::blas1::pointwiseDot( pupilongrid, gradLnBsolution, pupilongradLnBsolution); 

                dg::blas1::axpby( 1., gradLnBsolution , -1.,dzRZPHIlnB,diff2b); //diff = gradlnB - dz(ln(B))
                //cut boundaries

                double normdz2b = dg::blas2::dot( w3d,dzRZPHIlnB); //=  Integral (gdz(ln(B))^2 )
                std::cout << "Norm dz  = "<<sqrt( normdz2b)<<"\n";
//                 double normsol2b = dg::blas2::dot( w3d,pupilongradLnBsolution);//=  Integral (gradlnB^2 )
                std::cout << "Norm sol = "<<sqrt( normsol2)<<"\n";
                double normdiff2b=dg::blas2::dot( w3d, diff2b); //=  Integral ((gradlnB - dz(ln(B)))^2)
                double reldiff2b =sqrt( normdiff2b/normsol2 ); ;//=  sqrt(Integral ((gradlnB - dz(ln(B)))^2)/Integral (gradlnB^2 ))
                std::cout << "Rel Diff = "<<reldiff2b <<"\n";
                std::cout <<"---------------------------------------------------------------------------------------------" << "\n";
                std::cout <<"-----(3) test with gradlnb and with (a) Arakawa and (b) Poisson discretization" << "\n";    
                dg::ArakawaX< dg::DMatrix, dg::DVec>    arakawa(g3d); 
                dg::Poisson< dg::DMatrix, dg::DVec>     poiss(g3d);
                dg::DVec invBongrid = dg::evaluate( invB, g3d);
                dg::DVec psipongrid = dg::evaluate( psip, g3d);
                dg::DVec invnormrongrid = dg::evaluate( invnormr, g3d);
                dg::DVec arakawasolution(g3d.size());
                dg::DVec poisssolution(g3d.size());
                dg::DVec diff3(g3d.size());
                dg::DVec diff4(g3d.size());
//                 dg::blas1::pointwiseDot( pupilongrid, invnormrongrid, invnormrongrid); 
//                 dg::blas1::pointwiseDot( pupilongrid, invBongrid, invBongrid); 

                arakawa( lnBongrid, psipongrid, arakawasolution); //1/B [B,psip]
                poiss(   lnBongrid, psipongrid, poisssolution); //1/B [B,psip]
                dg::blas1::pointwiseDot( invBongrid, arakawasolution, arakawasolution); //1/B^2 [B,psip]
                dg::blas1::pointwiseDot( invnormrongrid, arakawasolution, arakawasolution); //1/(R B^2) [B,psip]
//                 dg::blas1::pointwiseDot( pupilongrid, arakawasolution, arakawasolution); 

                dg::blas1::pointwiseDot( invBongrid, poisssolution, poisssolution); //    1/B^2 [B,psip]
                dg::blas1::pointwiseDot( invnormrongrid, poisssolution, poisssolution); //1/(R B^2) [B,psip]
//                 dg::blas1::pointwiseDot( pupilongrid, poisssolution, poisssolution); 

                
                dg::blas1::axpby( 1., pupilongradLnBsolution , -1., arakawasolution,diff3);

                double normarak= dg::blas2::dot( w3d, arakawasolution); //=  Integral (gdz(ln(B))^2 )
                std::cout << "Norm normarak  = "<<sqrt( normarak)<<"\n";
                double normdiff3=dg::blas2::dot( w3d, diff3); //=  Integral ((gradlnB - dz(ln(B)))^2)
                double reldiff3 =sqrt( normdiff3/normsol2 ); ;//=  sqrt(Integral ((gradlnB - dz(ln(B)))^2)/Integral (gradlnB^2 ))
                std::cout << "Rel Diff = "<<reldiff3 <<"\n";
                
                dg::blas1::axpby( 1., pupilongradLnBsolution , -1., poisssolution,diff4);

                double normpoiss= dg::blas2::dot( w3d, poisssolution); //=  Integral (gdz(ln(B))^2 )
                std::cout << "Norm normpoiss  = "<<sqrt( normpoiss)<<"\n";
                double normdiff4=dg::blas2::dot( w3d, diff4); //=  Integral ((gradlnB - dz(ln(B)))^2)
                double reldiff4 =sqrt( normdiff4/normsol2 ); ;//=  sqrt(Integral ((gradlnB - dz(ln(B)))^2)/Integral (gradlnB^2 ))
                std::cout << "Rel Diff = "<<reldiff4 <<"\n";
                
                dzerrfile1 << pow(2,zz)*Nz <<" " << reldiff << std::endl;
                dzerrfile2 << pow(2,zz)*Nz <<" " << reldiff2 << std::endl;
                std::cout <<"---------------------------------------------------------------------------------------------" << "\n";
                std::cout <<"----(4) test div(B) != 0 "<<"\n";
                dg::DVec bRongrid = dg::evaluate( fieldR, grid);
                dg::DVec bZongrid = dg::evaluate( fieldZ, grid);
                dg::DVec dRbR(g3d.size());
                dg::DVec dZbZ(g3d.size());                
                dg::DVec invRbR(g3d.size());                
//              //cut boundaries
//                 dg::blas1::pointwiseDot( pupilongrid, bRongrid,bRongrid); 
//                 dg::blas1::pointwiseDot( pupilongrid, bZongrid, bZongrid); 
                invnormrongrid = dg::evaluate( invnormr, g3d);

                dg::DVec divB(g3d.size());                
//                 dg::blas2::gemv( arakawa.dx(), bRongrid, dRbR);
//                 dg::blas2::gemv( arakawa.dy(), bZongrid, dZbZ);
                dg::blas2::gemv( poiss.dxlhs(), bRongrid, dRbR); //d_R B^R
                dg::blas2::gemv( poiss.dylhs(), bZongrid, dZbZ); //d_Z B^Z
                dg::blas1::pointwiseDot( invnormrongrid , bRongrid, invRbR); // R_0/R B^R
                dg::blas1::axpby( 1., dRbR   , 1., dZbZ, divB); //d_R B^R + d_Z B^Z
                dg::blas1::axpby( 1./gp.R_0, invRbR , 1., divB); //( B^R/R/R_0 + d_R B^R + d_Z B^Z)
                dg::blas1::pointwiseDot( pupilongrid, divB, divB);  //cut 

                double normdivB2= dg::blas2::dot( w3d, divB); 
                std::cout << "divB = "<<sqrt( normdivB2)<<"\n";
                
             }
            dzerrfile1.close();
            dzerrfile2.close();
   
        }

    }
    

    return 0;
}
