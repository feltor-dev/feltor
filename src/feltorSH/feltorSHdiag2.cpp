#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <thrust/extrema.h>

#include "dg/algorithm.h"

#include "dg/file/file.h"
#include "parameters.h"

double X( double x, double) {return x;}
double Y( double, double y) {return y;}

struct Heaviside2d
{
    Heaviside2d( double sigma):sigma2_(sigma*sigma/4.), x_(0), y_(0){}
//     Heaviside2d( double sigma):sigma2_(sigma*sigma), x_(0), y_(0){}
    void set_origin( double x0, double y0){ x_=x0, y_=y0;}
    double operator()(double x, double y)const
    {
        double r2 = (x-x_)*(x-x_)+(y-y_)*(y-y_);
        if( r2 >= sigma2_)
            return 0.;
        return 1.;
    }
  private:
    const double sigma2_;
    double x_,y_;
};


int main( int argc, char* argv[])
{
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [output.nc]\n";
        return -1;
    }
//     std::ofstream os( argv[2]);
    std::cout << argv[1]<< " -> "<<argv[2]<<std::endl;

    ///////////////////read in and show inputfile//////////////////
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input(length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
    std::cout << "input "<<input<<std::endl;
    auto js = dg::file::string2Json( input, dg::file::comments::are_forbidden);
    const eule::Parameters p(js);
    p.display(std::cout);
    
    ///////////////////////////////////////////////////////////////////////////
    //Grids
    dg::Grid2d g2d( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
    dg::Grid1d g1d( 0., p.lx, p.n_out, p.Nx_out, p.bc_x);
    double time = 0.;
    //2d field
    size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[6] = {"electrons", "ions", "Telectrons","Tions", "potential","vor"}; 
    int dataIDs[6];
  
    std::vector<dg::DVec> npe(2,dg::evaluate(dg::zero,g2d));
    std::vector<dg::DVec> tpe(2,dg::evaluate(dg::zero,g2d));
    dg::DVec phi(dg::evaluate(dg::zero,g2d));
    dg::DVec vor(dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> npe_h(2,dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> tpe_h(2,dg::evaluate(dg::zero,g2d));
    dg::HVec phi_h(dg::evaluate(dg::zero,g2d));
    dg::HVec vor_h(dg::evaluate(dg::zero,g2d));
    dg::DVec xvec = dg::evaluate( dg::cooX2d, g2d);
    dg::DVec yvec = dg::evaluate( dg::cooY2d, g2d);
    dg::HVec xcoo(dg::evaluate(dg::cooX1d,g1d));
    dg::DVec one = dg::evaluate( dg::one, g2d);
    dg::DVec w2d = dg::create::weights( g2d);
    dg::DVec nemamp(dg::evaluate(dg::zero,g2d));
    dg::DVec helper(dg::evaluate(dg::zero,g2d));
    dg::DVec helper2(dg::evaluate(dg::zero,g2d));
    dg::DVec helper3(dg::evaluate(dg::zero,g2d));
    dg::DVec helper4(dg::evaluate(dg::zero,g2d));
    dg::DVec target(dg::evaluate(dg::zero,g2d));
    dg::DVec helper1d(dg::evaluate(dg::zero,g1d));
    
    dg::HVec transfer2d(dg::evaluate(dg::zero,g2d));
    dg::HVec transfer1d(dg::evaluate(dg::zero,g1d));
    dg::IDMatrix equi = dg::create::backscatter( g2d);


    
    double mass_=0.;
    double posX=0.,posY=0.,posX_init=0.,posY_init=0.,posX_old=0.,posY_old=0.;
    double velX,velY,velX_old=0. , velY_old=0.;    
    double accX,accY=0.;
    double deltaT = p.dt*p.itstp;
    size_t count1d[2]  = {1, g2d.n()*g2d.Nx()};
    size_t start1d[2]  = {0, 0};    
    //1d netcdf output file    

    dg::file::NC_Error_Handle err_out;
    int ncid_out;
    int namescomID[12],names1dID[4],names2dID[4],tvarID1d,timeID,timevarID;
    int dim_ids2d[3];
    std::string namescom[12] = {"posX" , "posY" , "velX" , "velY" , "accX" , "accY","posXmax","posYmax","velXmax" , "velYmax","compactness_ne","velCOM" };
    std::string names1d[4] = {"ne_max", "te_max", "ti_max","x_"};
    std::string names2d[4] = {"ti","Omega_d","weiss","Omega_E"}; 
    
    err_out = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid_out);
    err_out = nc_put_att_text( ncid_out, NC_GLOBAL, "inputfile", input.size(), input.data());
    err_out = dg::file::define_dimensions( ncid_out, dim_ids2d, &tvarID1d, g2d);
    err_out = nc_close(ncid_out); 
    
    
    err_out = nc_open( argv[2], NC_WRITE, &ncid_out);
    err_out = nc_redef(ncid_out);
    err_out = dg::file::define_time( ncid_out, "ptime", &timeID, &timevarID);
    for( unsigned i=0; i<12; i++){
        err_out = nc_def_var( ncid_out, namescom[i].data(),  NC_DOUBLE, 1, dim_ids2d, &namescomID[i]);
    }   
    for( unsigned i=0; i<4; i++){
        err_out = nc_def_var( ncid_out, names1d[i].data(),  NC_DOUBLE, 2, dim_ids2d, &names1dID[i]);
    }   
    for( unsigned i=0; i<4; i++){
        err_out = nc_def_var( ncid_out, names2d[i].data(),  NC_DOUBLE, 3, dim_ids2d, &names2dID[i]);
    }  
   
    err_out = nc_enddef(ncid_out);   
    
    const double hx = g2d.hx()/(double)g2d.n();
    const double hy = g2d.hy()/(double)g2d.n();
    unsigned Nx = p.Nx*p.n; 
    //routiens to compute ti
    //dg::PCG<dg::DVec> invert_invgamma2( helper, helper.size(), 1e-3);
    dg::Helmholtz2< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > invgamma2( g2d,g2d.bcx(), g2d.bcy(), -0.5*p.tau[1]*p.mu[1],dg::centered);
    dg::DVec binv(dg::evaluate( dg::LinearX( p.mcv, 1.), g2d) );
    dg::DVec B2(dg::evaluate( dg::one, g2d));
    dg::blas1::pointwiseDivide(B2,binv,B2);
    dg::blas1::pointwiseDivide(B2,binv,B2);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> polti(g2d, g2d.bcx(), g2d.bcy(), dg::centered);
    
    //weiss field
    //dg::Weiss<dg::DMatrix,dg::DVec,dg::DVec> weissfield(g2d, dg::centered);  
    //BLOB COMPACTNESS Heaviside function
    Heaviside2d heavi(2.0* p.sigma);
    double normalize = 1.;
    dg::DVec heavy;
    //////////////////////////////open nc file//////////////////////////////////
    err_out = nc_open( argv[2], NC_WRITE, &ncid_out);
    unsigned position = 0;
    double posX_max = 0.0,posY_max = 0.0,posX_max_old = 0.0,posY_max_old = 0.0,velX_max=0.0, velY_max=0.0,posX_max_hs=0.0,posY_max_hs=0.0,velCOM=0.0;
    double compactness_ne=0.0;
    //-----------------Start timestepping
    for( unsigned i=0; i<p.maxout; i++)
    {
        start2d[0] = i;
        start1d[0] = i;
        //Get fields
        for (unsigned j=0;j<2;j++)
        {
            err = nc_inq_varid(ncid, names[j].data(), &dataIDs[j]);
            err = nc_get_vara_double( ncid, dataIDs[j], start2d, count2d, npe_h[j].data());
            npe[j] = npe_h[j];
            dg::blas1::transform(npe[j], npe[j], dg::PLUS<double>(p.bgprofamp + p.nprofileamp));
        }
        for (unsigned j=0;j<2;j++)
        {
            err = nc_inq_varid(ncid, names[j+2].data(), &dataIDs[j+2]);
            err = nc_get_vara_double( ncid, dataIDs[j+2], start2d, count2d, tpe_h[j].data());
            tpe[j] = tpe_h[j];
            dg::blas1::transform(tpe[j], tpe[j], dg::PLUS<double>(p.bgprofamp + p.nprofileamp));
        }
        err = nc_inq_varid(ncid, names[4].data(), &dataIDs[4]);
        err = nc_get_vara_double( ncid, dataIDs[4], start2d, count2d, phi_h.data());
        phi = phi_h;
        err = nc_inq_varid(ncid, names[5].data(), &dataIDs[5]);
        err = nc_get_vara_double( ncid, dataIDs[5], start2d, count2d, vor_h.data());
        vor = vor_h;
        dg::blas1::transform(npe[0], nemamp, dg::PLUS<double>(-p.bgprofamp - p.nprofileamp));
        //Compute mass
        mass_ = dg::blas2::dot( one, w2d, nemamp ); 
        if (i==0){
            posX_init = dg::blas2::dot( xvec, w2d, nemamp)/mass_;
            posY_init = dg::blas2::dot( yvec, w2d, nemamp)/mass_;    
        }
        if (i>0){
            time += p.itstp*p.dt;

            posX = dg::blas2::dot( xvec, w2d,nemamp)/mass_-posX_init ;
            posY = dg::blas2::dot( yvec, w2d, nemamp)/mass_-posY_init;
        }
        if (i==0){
            velX_old = -posX/deltaT;
            velY_old = -posY/deltaT; 
            posX_old = posX;
            posY_old = posY;
        }
        
        velX = (posX - posX_old)/deltaT;
        velY = (posY - posY_old)/deltaT;
        velCOM=sqrt(velX*velX+velY*velY);
        accX = (velX - velX_old)/deltaT;
        accY = (velY - velY_old)/deltaT;
        if (i>0){
        posX_old = posX; posY_old = posY;
        velX_old = velX; velY_old = velY;   
        }

        
       err_out = nc_put_vara_double( ncid_out, timevarID, start1d, count1d, &time);
       err_out = nc_put_vara_double( ncid_out, namescomID[0], start1d, count1d, &posX);
       err_out = nc_put_vara_double( ncid_out, namescomID[1], start1d, count1d, &posY);
       err_out = nc_put_vara_double( ncid_out, namescomID[2], start1d, count1d, &velX);
       err_out = nc_put_vara_double( ncid_out, namescomID[3], start1d, count1d, &velY);
       err_out = nc_put_vara_double( ncid_out, namescomID[4], start1d, count1d, &accX);
       err_out = nc_put_vara_double( ncid_out, namescomID[5], start1d, count1d, &accY);
       err_out = nc_put_vara_double( ncid_out, namescomID[11], start1d, count1d, &velCOM);
       err_out = nc_put_vara_double( ncid_out, tvarID1d, start1d, count1d, &time);         
       
      
       //transform back
//         dg::blas1::transform(npe[0], npe[0], dg::PLUS<double>(p.bgprofamp + p.nprofileamp));

       //get max position and value(x,y_max) of electron density
        dg::blas2::gemv( equi, npe[0], helper);
        position = thrust::distance( helper.begin(), thrust::max_element( helper.begin(), helper.end()) );
        posX_max = hx*(1./2. + (double)(position%Nx))-posX_init;
        posY_max = hy*(1./2. + (double)(position/Nx))-posY_init;
        posX_max_hs = hx*(1./2. + (double)(position%Nx));
        posY_max_hs = hy*(1./2. + (double)(position/Nx));
//         std::cout << "posXmax "<<posX_max<<" posYmax "<<posY_max << std::endl;
        err_out = nc_put_vara_double( ncid_out, namescomID[6], start1d, count1d, &posX_max);
        err_out = nc_put_vara_double( ncid_out, namescomID[7], start1d, count1d, &posY_max);      
        velX_max = (posX_max - posX_max_old)/deltaT;
        velY_max = (posY_max - posY_max_old)/deltaT;
        std::cout << "COM: time = "<< time << " mass :" << mass_ << " velX :" << velX << " velY :" << velY << " posX :" << posX << " posY :" << posY << std::endl;
        std::cout << "MAX: time = "<< time << " mass :" << mass_ << " velX :" << velX_max << " velY :" << velY_max << " posX :" << posX_max << " posY :" << posY_max << std::endl;
        if (i>0){
            posX_max_old = posX_max; posY_max_old = posY_max;
        }  
        err_out = nc_put_vara_double( ncid_out, namescomID[8], start1d, count1d, &velX_max);
        err_out = nc_put_vara_double( ncid_out, namescomID[9], start1d, count1d, &velY_max);      
//         std::cout << "maxval "<<*thrust::max_element( helper.begin(), helper.end())<< std::endl;        
        //Compute interpolation matrix for 1d field    
        dg::HVec y0coone(dg::evaluate(dg::CONSTANT(posY_max_hs ),g1d));
        dg::IDMatrix interpne(dg::create::interpolation(xcoo,y0coone, g2d)) ;
        
        dg::blas2::gemv(interpne,npe[0],helper1d); 
        dg::assign( helper1d, transfer1d);
        err_out = nc_put_vara_double( ncid_out, names1dID[0], start1d, count1d, transfer1d.data());    
        
       //get max position and value(x,y_max) of electron temperature
//         dg::blas2::gemv( equi, tpe[0], helper);
//         position = thrust::distance( helper.begin(), thrust::max_element( helper.begin(), helper.end()) );
//         posX_max = hx*(1./2. + (double)(position%Nx));
//         posY_max = hy*(1./2. + (double)(position/Nx));
//         std::cout << "posXmax "<<posX_max<<"posYmax "<<posY_max << std::endl;
//         std::cout << "maxval "<<*thrust::max_element( helper.begin(), helper.end())<< std::endl;        
        //Compute interpolation matrix for 1d field    
        dg::HVec y0coote(dg::evaluate(dg::CONSTANT(posY_max_hs ),g1d));
        dg::IDMatrix interpte(dg::create::interpolation(xcoo,y0coote, g2d)) ;
        
        dg::blas2::gemv(interpte,tpe[0],helper1d); 
        dg::assign( helper1d, transfer1d);
        err_out = nc_put_vara_double( ncid_out, names1dID[1], start1d, count1d, transfer1d.data());    
        
        
        
        //
        //compute weiss field
//         weissfield.symv(phi,helper2);
//         transfer2d = helper2;
// 
//         err_out = nc_put_vara_double( ncid_out, names2dID[2], start2d, count2d, transfer2d.data());
// 
//      
// //         //Compute ion temperature with phi=0
//         dg::blas1::pointwiseDivide(B2,tpe[1],helper);        //helper  = B^2/Ti
//         invgamma2.set_chi(helper);                           //helmholtz2 chi = B^2/Ti
// 
//         //Solve (pi = gamma1^dagger + gamma2^dagger Pi)
// //         invert_invgamma2(invgamma2,helper2,helper3); //solve for p_i_tilde -amp^2        
// //         dg::blas1::pointwiseDot(helper2,helper,helper2); //target = B^2/Ti target        
// //         dg::blas1::transform(helper2,helper2, dg::PLUS<>(+(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp))); 
//         dg::blas1::transform( tpe[1], helper2, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //t_i_tilde
//         dg::blas1::pointwiseDivide(helper2,B2,helper2); //chi=t_i_tilde/b^2 
//         polti.set_chi(one);
//         dg::blas2::gemv(polti,helper2,helper3);
//         dg::blas1::pointwiseDivide(helper3,helper,helper2); //-Ti/B^2 lap T_i_tilde/B^2
//         dg::blas2::gemv(polti,helper2,target);//target = + lap (Ti/B^2 lap T_i_tilde/B)
//         dg::blas1::pointwiseDot( npe[1],tpe[1],helper4);     //helper3 = P_i
//         dg::blas1::transform(helper4,helper4, dg::PLUS<>(-(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)));//Pi_tilde = P_i -amp^2
//         dg::blas1::axpby(1.0,helper4 ,-(1.-p.tau[1]*0.5*p.mcv*p.mcv*(p.bgprofamp + p.nprofileamp))*(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1],helper3,helper3);  
//         dg::blas1::axpby(1.0, helper3, -(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*p.tau[1]*0.25,target, helper3); 
//         dg::blas1::axpby(1.0, helper3, (p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*2.*p.mcv*p.mcv*(1.-0.5*p.tau[1]*(p.bgprofamp + p.nprofileamp)*p.mcv*p.mcv),one, helper3); 
//         invert_invgamma2(invgamma2,target,helper3); //=Ti/B^2(p_i_tilde_bar) = bar(Gamma)_dagger { P_i_tilde + a tau_i lap T_i_tilde/B^2  - a tau^2 /4 lap (Ti/B^2 lap T_i_tilde/B^2)   }
//         dg::blas1::pointwiseDot(target,helper,target); //target = B^2/Ti target = (p_i_tilde) 
//         dg::blas1::transform(target,target, dg::PLUS<>(+(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp))); //pi_bar
// 
//         //set up polarisation term
//         dg::blas1::pointwiseDot( npe[1],tpe[1],helper4);     //helper3 = P_i
//         dg::blas1::pointwiseDot( helper4, binv, helper2);
//         dg::blas1::pointwiseDot( helper2, binv, helper2);    //helper2 = P_i/B^2   
//         polti.set_chi(helper2); //elliptic chi =  P_i/B^2    for polarisation
// 
//         //helper2=pi_tilde
//         dg::blas2::symv(polti,phi,helper3); //-nabla(P_i/B^2 (nabla_perp phi));
//         dg::blas1::axpby(1.0, target, -2.0*p.mu[1], helper3,helper2);//pi  =  2 nabla(P_i/B^2 (nabla_perp phi)) + p_i_bar
//         dg::blas1::pointwiseDivide(helper2,npe[0],helper2); //ti=(pi-amp^2)/ne
// //         
// //          //compute Omega_d
//                 dg::blas1::transform(helper2,helper, dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); 
// //                 
//         polti.set_chi(one);
//         dg::blas2::symv(polti,helper,helper3); //nabla(nabla_perp p_i/B^2 );
//         dg::blas1::scal(helper3,-1.0*p.tau[1]);
// //         
// //         //write t_i into 2dnetcdf
//         transfer2d = helper2;
//         err_out = nc_put_vara_double( ncid_out, names2dID[0], start2d, count2d, transfer2d.data());
//         //write Omega_d into 2dnetcdf
//         transfer2d = helper3;
//         err_out = nc_put_vara_double( ncid_out, names2dID[1], start2d, count2d, transfer2d.data());
//               
//         
        
        //compute Omega_E and write Omega_E into 2dnetcdf
        dg::blas1::pointwiseDot( npe[1],one,helper3); 
        dg::blas1::pointwiseDot( helper3, binv, helper2);
        dg::blas1::pointwiseDot( helper2, binv, helper2);   
        polti.set_chi(helper2);
        dg::blas2::symv(polti,phi,helper3); 
        dg::blas1::scal(helper3,-1.0*p.mu[1]);
        polti.set_chi(one);
        dg::blas2::symv(polti,phi,helper); 
        dg::blas1::pointwiseDot(helper,helper2,helper);
        dg::blas1::axpby(1.0,helper3,p.mu[1],helper,helper3);
        dg::assign( helper3, transfer2d);
        err_out = nc_put_vara_double( ncid_out, names2dID[3], start2d, count2d, transfer2d.data());

  /*
        
        //get max position and value(x,y_max) of electron temperature
//         dg::blas2::gemv( equi, helper2, helper);
//         position = thrust::distance( helper.begin(), thrust::max_element( helper.begin(), helper.end()) );
//         posX_max = hx*(1./2. + (double)(position%Nx));
//         posY_max = hy*(1./2. + (double)(position/Nx));
//         std::cout << "posXmax "<<posX_max<<"posYmax "<<posY_max << std::endl;
//         std::cout << "maxval "<<*thrust::max_element( helper.begin(), helper.end())<< std::endl;        
        //Compute interpolation matrix for 1d field    
        dg::DVec y0cooti(dg::evaluate(dg::CONSTANT(posY_max),g1d));
        dg::IDMatrix interpti(dg::create::interpolation(xcoo,y0cooti, g2d)) ;
        
        dg::blas2::gemv(interpti, helper2,helper1d); 
        transfer1d=helper1d;
        err_out = nc_put_vara_double( ncid_out, names1dID[2], start1d, count1d, transfer1d.data());      */
        dg::assign( xcoo, transfer1d);
        err_out = nc_put_vara_double( ncid_out, names1dID[3],   start1d, count1d,transfer1d.data());       
        
        dg::blas1::transform(npe[0], npe[0], dg::PLUS<double>(-p.bgprofamp - p.nprofileamp));
        //BLOB COMPACTNESS
        if (i==0) {
            heavi.set_origin( posX_max_hs, posY_max_hs );
            heavy = dg::evaluate( heavi, g2d);
            normalize = dg::blas2::dot( heavy, w2d, npe[0]);
        }
        heavi.set_origin( posX_max_hs, posY_max_hs);
        heavy = dg::evaluate( heavi, g2d);
//         std::cout <<std::scientific<< dg::blas2::dot( heavy, w2d, npe[0])/normalize << std::endl;
        compactness_ne =  dg::blas2::dot( heavy, w2d, npe[0])/normalize ;
        err_out = nc_put_vara_double( ncid_out, namescomID[10], start1d, count1d, &compactness_ne);            
    }
    err_out = nc_close(ncid_out);
    err = nc_close(ncid);
    
    return 0;
}

