#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

#include "dg/algorithm.h"

#include "dg/file/file.h"
#include "feltorShw/parameters.h"
// #include "probes.h"

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
    dg::Grid2d g2d( 0., p.lx, 0.,p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::Grid2d g2d_in( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
    dg::Grid1d g1d( 0., p.lx,p.n, p.Nx, p.bc_x);
    
    double time = 0.;
    
    dg::HVec w2d = dg::create::weights( g2d);
    dg::HVec w1d = dg::create::weights( g1d);

    
    dg::HVec phi(dg::evaluate(dg::zero,g2d));
    dg::HVec vor(dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> ntilde(2,dg::evaluate(dg::zero,g2d)); // = N/n_G -1
    std::vector<dg::HVec> navgtilde(2,dg::evaluate(dg::zero,g2d)); // = N/<N> -1
    std::vector<dg::HVec> npe(2,dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> logn(2,dg::evaluate(dg::zero,g2d));
    dg::HVec temp(dg::evaluate(dg::zero,g2d));
    dg::HVec temp1(dg::evaluate(dg::zero,g2d));
    dg::HVec temp2(dg::evaluate(dg::zero,g2d));
    dg::HVec temp3(dg::evaluate(dg::zero,g2d));
    dg::HVec temp_in(dg::evaluate(dg::zero,g2d_in));
    dg::HVec one(dg::evaluate(dg::one,g2d));
    dg::HVec one1d(dg::evaluate(dg::one,g1d));
    dg::HVec temp1d(dg::evaluate(dg::zero,g1d));
    dg::HVec xcoo(dg::evaluate(dg::cooX1d,g1d));
    dg::ExpProfX prof(p.nprofileamp, p.bgprofamp,p.invkappa); 
    dg::HVec nprof(dg::evaluate(prof,g2d));
    dg::HVec y0coo(dg::evaluate(dg::CONSTANT(0.0),g1d));
    dg::Average<dg::IHMatrix,dg::HVec> polavg(g2d, dg::coo2d::y);
    dg::IHMatrix interp(dg::create::interpolation(xcoo,y0coo,g2d));
    dg::IHMatrix interp_in = dg::create::interpolation(g2d,g2d_in);
    dg::Poisson<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> poisson(g2d,  p.bc_x, p.bc_y,  p.bc_x_phi, p.bc_y);
    dg::Elliptic<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> pol(g2d,   p.bc_x_phi, p.bc_y, dg::centered);
    dg::Elliptic<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> lap(g2d,   p.bc_x, p.bc_y, dg::centered);
    
    dg::HVec lhs(dg::evaluate(dg::TanhProfX(p.lx*p.sourceb,p.sourcew,-1.0,0.0,1.0),g2d));

    //2d field
    size_t count2d[3]  = {1, g2d_in.n()*g2d_in.Ny(), g2d_in.n()*g2d_in.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[4] = {"electrons", "ions",  "potential","vor"}; 
    int dataIDs[4]; 
    //1d profiles
    dg::file::NC_Error_Handle err_out;
    int ncid_out,dataIDs1d[33], tvarIDout;
    //Rfx = -\partial_x\overbar{\overbar{\delta} u_x \overbar{\delta} u_y } = -\partial_x R_favre
    //A   = -\overbar{u_x} \partial_x \overbar{u_y} 
    //Rfn = \overbar{\overbar{\delta} u_x \overbar{\delta} u_y } \partial_x ln(<n_e>)
    //An  = 2 \overbar{u_x}   \overbar{u_y} \partial_x ln(<n_e>)
    //Rx  = -\partial_x<\delta u_x \delta u_y > = -\partial_x R
    //Rnx = -\partial_x (<n_e> <\delta u_x \delta u_y >)
    //Guyx= -\partial_x (<\delta n_e \delta u_x> < u_y >)
    //Tx  = -\partial_x <\delta n_e \delta u_x \delta u_y>
    //Guynx= -\partial_x (<\delta n_e \delta u_x> < u_y >/<n_e>)
    //Tx  = -\partial_x (<\delta n_e \delta u_x \delta u_y>/<n_e>)
    std::string names1d[33] =  {"neavg", "Niavg",  "ln(ne)avg","ln(Ni)avg","potentialavg","voravg","x_", "vyavg","Rfx","A","Rfn","An","dtfauy","Rx","invkappa","vyfavg","nvyavg","Rnx","Guyx","Tx","Guynx","Tnx","net","neat","nuturb","dxfaux","fauxik","difflnn","difffauy1","difffauy2","Slnn","Sfauy","Rn"};  
    int dim_ids2d[3];
    size_t count1d[2]  = {1, g2d.n()*g2d.Nx()};
    size_t start1d[2]  = {0, 0};    
    err_out = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid_out);
    err_out= nc_put_att_text( ncid_out, NC_GLOBAL, "inputfile", input.size(), input.data());
    err_out= dg::file::define_dimensions( ncid_out, dim_ids2d, &tvarIDout, g2d);
     int dim_ids1d[2] = {dim_ids2d[0],dim_ids2d[2]};
    for( unsigned i=0; i<33; i++){
        err_out = nc_def_var( ncid_out, names1d[i].data(), NC_DOUBLE, 2, dim_ids1d, &dataIDs1d[i]);
    }   

//     }   
    err_out = nc_close(ncid_out); 
    //2d field netcdf vars read
    
    unsigned imin,imax;
    imin= 0;
    time = imin*p.itstp;
    //////////////////////////////open nc file//////////////////////////////////
    err_out = nc_open( argv[2], NC_WRITE, &ncid_out);

    unsigned num_probes = 5;
    dg::HVec xprobecoords(num_probes,1.);
    for (unsigned i=0;i<num_probes; i++) {
        xprobecoords[i] = (1+i)*p.lx/((double)(num_probes+1));
    }
    const dg::HVec yprobecoords(num_probes,p.ly/2.);
    dg::HVec gamma(phi);
    dg::HVec uy(phi);
    dg::HVec ux(phi);
    dg::HVec aux(phi);
    dg::HVec auy(phi);
    dg::HVec tux(phi);
    dg::HVec tuy(phi);
    dg::HVec faux(phi);
    dg::HVec fauy(phi);
    dg::HVec ftux(phi);
    dg::HVec ftuy(phi);
    dg::HVec anpe(phi);
    dg::HVec R(phi);
    dg::HVec Rf(phi);
    dg::HVec dne(phi); //ne - <N_G>
    dg::HVec npe_probes(num_probes);
    dg::HVec phi_probes(num_probes);
    dg::HVec gamma_probes(num_probes);
    dg::IHMatrix probe_interp(dg::create::interpolation(xprobecoords, yprobecoords, g2d)) ;
    dg::HMatrix dy(dg::create::dy(g2d));
    //probe netcdf file
    err_out = nc_redef(ncid_out);
    int npe_probesID[num_probes],phi_probesID[num_probes],gamma_probesID[num_probes];
    int OmegaID, OmegazID, OmegaratioID, TperpzID, TperpID, TperpratioID, Gamma_neID,invkappaavgID,RfxnormID,AnormID,RfnnormID,AnnormID,RxnormID,RnxnormID,GuyxnormID,TxnormID,GuynxnormID,TnxnormID,netnormID,neatnormID,dtfauynormID,RxnormscalID,GuynxnormscalID,TnxnormscalID,AnormscalID,AnnormscalID,RfnnormscalID,neatsupnormID,nuturbnormID,dxfauxnormID,fauxiknormID,difflnnnormID,difffauy1normID,difffauy2normID,SlnnnormID,SfauynormID,RnnormscalID,dfnormscalID,RnffnormscalID,vyfavgnormID;
    std::string npe_probes_names[num_probes] ;
    std::string phi_probes_names[num_probes] ;
    std::string gamma_probes_names[num_probes];
    int timeID, timevarID;
    err_out = dg::file::define_time( ncid_out, "ptime", &timeID, &timevarID);
    for( unsigned i=0; i<num_probes; i++){
        std::stringstream ss1,ss2,ss3;
        ss1<<"Ne_p"<<i;
        npe_probes_names[i] =ss1.str();
        err_out = nc_def_var( ncid_out, npe_probes_names[i].data(),     NC_DOUBLE, 1, &timeID, &npe_probesID[i]);
        ss2<<"phi_p"<<i;
        phi_probes_names[i] =ss2.str();
        err_out = nc_def_var( ncid_out, phi_probes_names[i].data(),    NC_DOUBLE, 1, &timeID, &phi_probesID[i]);  
        ss3<<"G_x"<<i;
        gamma_probes_names[i] =ss3.str();
        err_out = nc_def_var( ncid_out, gamma_probes_names[i].data(),    NC_DOUBLE, 1, &timeID, &gamma_probesID[i]);
    }
    err_out = nc_def_var( ncid_out, "Uperpz",    NC_DOUBLE, 1, &timeID, &TperpzID);
    err_out = nc_def_var( ncid_out, "Uperp",     NC_DOUBLE, 1, &timeID, &TperpID);
    err_out = nc_def_var( ncid_out, "Uperpratio",     NC_DOUBLE, 1, &timeID, &TperpratioID);
    err_out = nc_def_var( ncid_out, "Omega",    NC_DOUBLE, 1, &timeID, &OmegaID);
    err_out = nc_def_var( ncid_out, "Omegaz",     NC_DOUBLE, 1, &timeID, &OmegazID);
    err_out = nc_def_var( ncid_out, "Omegaratio",     NC_DOUBLE, 1, &timeID, &OmegaratioID);
    err_out = nc_def_var( ncid_out, "Gamma",    NC_DOUBLE, 1, &timeID, &Gamma_neID);
    err_out = nc_def_var( ncid_out, "invkappaavg",    NC_DOUBLE, 1, &timeID, &invkappaavgID);
    err_out = nc_def_var( ncid_out, "Rfxnorm",    NC_DOUBLE, 1, &timeID, &RfxnormID);
    err_out = nc_def_var( ncid_out, "Anorm",    NC_DOUBLE, 1, &timeID, &AnormID);
    err_out = nc_def_var( ncid_out, "Rfnnorm",    NC_DOUBLE, 1, &timeID, &RfnnormID);
    err_out = nc_def_var( ncid_out, "Annorm",    NC_DOUBLE, 1, &timeID, &AnnormID);
    err_out = nc_def_var( ncid_out, "Rxnorm",    NC_DOUBLE, 1, &timeID, &RxnormID);
    err_out = nc_def_var( ncid_out, "Rnxnorm",    NC_DOUBLE, 1, &timeID, &RnxnormID);
    err_out = nc_def_var( ncid_out, "Guyxnorm",    NC_DOUBLE, 1, &timeID, &GuyxnormID);
    err_out = nc_def_var( ncid_out, "Txnorm",    NC_DOUBLE, 1, &timeID, &TxnormID);
    err_out = nc_def_var( ncid_out, "Guynxnorm",    NC_DOUBLE, 1, &timeID, &GuynxnormID);
    err_out = nc_def_var( ncid_out, "Tnxnorm",    NC_DOUBLE, 1, &timeID, &TnxnormID);
    err_out = nc_def_var( ncid_out, "dtfauynorm",    NC_DOUBLE, 1, &timeID, &dtfauynormID);
    err_out = nc_def_var( ncid_out, "netnorm",    NC_DOUBLE, 1, &timeID, &netnormID);
    err_out = nc_def_var( ncid_out, "neatnorm",    NC_DOUBLE, 1, &timeID, &neatnormID);
    err_out = nc_def_var( ncid_out, "neatsupnorm",    NC_DOUBLE, 1, &timeID, &neatsupnormID);
    
    err_out = nc_def_var( ncid_out, "Rxnormscal",    NC_DOUBLE, 1, &timeID, &RxnormscalID);
    err_out = nc_def_var( ncid_out, "Guynxnormscal",    NC_DOUBLE, 1, &timeID, &GuynxnormscalID);
    err_out = nc_def_var( ncid_out, "Tnxnormscal",    NC_DOUBLE, 1, &timeID, &TnxnormscalID);
    err_out = nc_def_var( ncid_out, "Anormscal",    NC_DOUBLE, 1, &timeID, &AnormscalID);
    err_out = nc_def_var( ncid_out, "Annormscal",    NC_DOUBLE, 1, &timeID, &AnnormscalID);
    err_out = nc_def_var( ncid_out, "Rfnnormscal",    NC_DOUBLE, 1, &timeID, &RfnnormscalID);
    
    err_out = nc_def_var( ncid_out, "nuturbnorm",    NC_DOUBLE, 1, &timeID, &nuturbnormID);
    
    err_out = nc_def_var( ncid_out, "dxfauxnorm",    NC_DOUBLE, 1, &timeID, &dxfauxnormID);
    err_out = nc_def_var( ncid_out, "fauxiknorm",    NC_DOUBLE, 1, &timeID, &fauxiknormID);
    err_out = nc_def_var( ncid_out, "difflnnnorm",    NC_DOUBLE, 1, &timeID, &difflnnnormID);
    err_out = nc_def_var( ncid_out, "difffauy1norm",    NC_DOUBLE, 1, &timeID, &difffauy1normID);
    err_out = nc_def_var( ncid_out, "difffauy2norm",    NC_DOUBLE, 1, &timeID, &difffauy2normID);
    err_out = nc_def_var( ncid_out, "Slnnnorm",    NC_DOUBLE, 1, &timeID, &SlnnnormID);
    err_out = nc_def_var( ncid_out, "Sfauynorm",    NC_DOUBLE, 1, &timeID, &SfauynormID);
    err_out = nc_def_var( ncid_out, "Rnnormscal",    NC_DOUBLE, 1, &timeID, &RnnormscalID);
    err_out = nc_def_var( ncid_out, "dfnormscal",    NC_DOUBLE, 1, &timeID, &dfnormscalID);
    err_out = nc_def_var( ncid_out, "Rnffnormscal",    NC_DOUBLE, 1, &timeID, &RnffnormscalID);
    err_out = nc_def_var( ncid_out, "vyfavgnorm",    NC_DOUBLE, 1, &timeID, &vyfavgnormID);
    
    err_out = nc_enddef(ncid_out);   
    err_out = nc_open( argv[2], NC_WRITE, &ncid_out);
    err = nc_inq_dimid(ncid, "time", &timeID);
    size_t steps;
    err = nc_inq_dimlen(ncid, timeID, &steps);
    steps-=1;
    imax = steps;
    for( unsigned i=imin; i<imax; i++)//timestepping
    {
            start2d[0] = i;
//             start2d_out[0] = i;
            start1d[0] = i;
            time += p.itstp*p.dt;

            std::cout << "time = "<< time <<  std::endl;

            double Tperp,Tperpz,Tperpratio,Gamma_ne,Omega,Omegaz,Omegaratio;
            
            err = nc_inq_varid(ncid, names[2].data(), &dataIDs[2]);
            err = nc_get_vara_double( ncid, dataIDs[2], start2d, count2d,  temp_in.data());
            dg::blas2::gemv( interp_in, temp_in,phi);
            err = nc_inq_varid(ncid, names[3].data(), &dataIDs[3]);
            err = nc_get_vara_double( ncid, dataIDs[3], start2d, count2d,  temp_in.data());
            dg::blas2::gemv( interp_in, temp_in,vor);
            //Full-F
            if (p.modelmode==0 || p.modelmode==1)
            {
                err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]);
                err = nc_get_vara_double( ncid, dataIDs[0], start2d, count2d, temp_in.data());
                dg::blas2::gemv( interp_in, temp_in,npe[0]);
                err = nc_inq_varid(ncid, names[1].data(), &dataIDs[1]);
                err = nc_get_vara_double( ncid, dataIDs[1], start2d, count2d,  temp_in.data());
                dg::blas2::gemv( interp_in, temp_in,npe[1]);
                 
                for (unsigned i=0;i<2;i++) {
                    dg::blas1::transform(npe[i], npe[i], dg::PLUS<>(p.bgprofamp + p.nprofileamp));
                    dg::blas1::transform( npe[i], logn[i], dg::LN<double>());
                    dg::blas1::pointwiseDivide(npe[i],nprof,ntilde[i]);
                    dg::blas1::transform(ntilde[i], ntilde[i], dg::PLUS<>(-1.0));
		    
		    polavg(npe[i],temp);
		    dg::blas1::pointwiseDivide(npe[i],temp,navgtilde[i]);
		    dg::blas1::transform(navgtilde[i], navgtilde[i], dg::PLUS<>(-1.0));
		    
                }

                pol.variation(phi,temp2);
                Tperp = 0.5*dg::blas2::dot( one, w2d, temp2);   // 0.5   u_E^2            
                polavg(phi,temp);      // <N u_E^2 > 
                pol.variation(temp,temp2);
                Tperpz = 0.5*dg::blas2::dot( one, w2d, temp2);   //0.5 ( D_x <phi> )^2 
                Tperpratio = Tperpz/Tperp;
                dg::blas2::gemv( poisson.dyrhs(), phi, temp2); 
                Gamma_ne = -1.* dg::blas2::dot(npe[0],w2d,temp2);

               
                pol.set_chi(one);
                dg::blas2::symv(pol,phi,temp); //- nabla ( nabla phi)
                dg::blas1::scal(temp,-1.);     // nabla ( nabla phi)
                Omega = 0.5* dg::blas2::dot( temp, w2d, temp);  // (nabla (N nabla phi) )^2              
                polavg(phi,temp2);      // <N u_E^2 > 
                dg::blas2::symv(pol,temp2,temp); //- nabla ( nabla <phi>)
                dg::blas1::scal(temp,-1.);     // nabla ( nabla <phi>)
                Omegaz =0.5* dg::blas2::dot( temp, w2d, temp);   //< nabla (N nabla phi) >^2 or better < nabla (N nabla phi)^2 >?
                Omegaratio = Omegaz/Omega;
                
                //Favre and conventional Reynolds stress
                polavg(npe[0],anpe);
                dg::blas1::axpby(1.0,npe[0],-1.0,anpe,dne);
                dg::blas2::gemv(poisson.dxrhs(),phi,uy);
                dg::blas2::gemv(poisson.dyrhs(),phi,ux);
                dg::blas1::scal(ux,-1.0);

                //conventional Reynolds stress
                polavg(ux,aux);
                polavg(uy,auy);
                dg::blas1::axpby(1.0,ux,-1.0,aux,tux);
                dg::blas1::axpby(1.0,uy,-1.0,auy,tuy);
                dg::blas1::pointwiseDot(tuy,tux,temp);
                polavg(temp,R);
                
                //Favre Reynolds stress
                dg::blas1::pointwiseDot(ux,npe[0],temp1);
                polavg(temp1,faux);
                dg::blas1::pointwiseDivide(faux,anpe,faux);
                dg::blas1::axpby(1.0,ux,-1.0,faux,ftux);
                dg::blas1::pointwiseDot(uy,npe[0],temp1);
                polavg(temp1,fauy);
                dg::blas1::pointwiseDivide(fauy,anpe,fauy);
                dg::blas1::axpby(1.0,uy,-1.0,fauy,ftuy);
                dg::blas1::pointwiseDot(ftuy,ftux,temp);
                dg::blas1::pointwiseDot(temp,npe[0],temp);
                polavg(temp,temp1);
                dg::blas1::pointwiseDivide(temp1,anpe,Rf);
            }
            //delta-f
            if (p.modelmode==2)
            { 
                err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]);
                err = nc_get_vara_double( ncid, dataIDs[0], start2d, count2d, temp_in.data());
                dg::blas2::gemv( interp_in, temp_in,ntilde[0]);
                err = nc_inq_varid(ncid, names[1].data(), &dataIDs[1]);
                err = nc_get_vara_double( ncid, dataIDs[1], start2d, count2d,  temp_in.data());
                dg::blas2::gemv( interp_in, temp_in,ntilde[1]);

                for (unsigned i=0;i<2;i++) {
                    dg::blas1::pointwiseDot(nprof,ntilde[i],temp);
                    dg::blas1::axpby(1.0,nprof,1.0,temp,npe[i]);
                    dg::blas1::transform( npe[i], logn[i], dg::LN<double>());
		    
		    polavg(npe[i],temp);
		    dg::blas1::pointwiseDivide(npe[i],temp,navgtilde[i]);
		    dg::blas1::transform(navgtilde[i], navgtilde[i], dg::PLUS<>(-1.0));
                }
                                    
                pol.variation(phi,temp2);
                Tperp = 0.5*dg::blas2::dot( one, w2d, temp2);   // 0.5   u_E^2            
                polavg(phi,temp);      // <N u_E^2 > 
                pol.variation(temp,temp2);
                Tperpz = 0.5*dg::blas2::dot( one, w2d, temp2);   //0.5 ( D_x <phi> )^2 
                Tperpratio = Tperpz/Tperp;
                dg::blas2::gemv( poisson.dyrhs(), phi, temp2); 
                Gamma_ne = -1.* dg::blas2::dot(npe[0],w2d,temp2);

               
                pol.set_chi(one);
                dg::blas2::symv(pol,phi,temp); //- nabla ( nabla phi)
                dg::blas1::scal(temp,-1.);     // nabla ( nabla phi)
                Omega = 0.5* dg::blas2::dot( temp, w2d, temp);  // (nabla (N nabla phi) )^2              
                polavg(phi,temp2);      // <N u_E^2 > 
                dg::blas2::symv(pol,temp2,temp); //- nabla ( nabla <phi>)
                dg::blas1::scal(temp,-1.);     // nabla ( nabla <phi>)
                Omegaz =0.5* dg::blas2::dot( temp, w2d, temp);   //< nabla (N nabla phi) >^2 or better < nabla (N nabla phi)^2 >?
                Omegaratio = Omegaz/Omega;
                
                //Favre Reynolds stress
                polavg(npe[0],anpe);
                dg::blas1::axpby(1.0,npe[0],-1.0,anpe,dne);
                dg::blas2::gemv(poisson.dxrhs(),phi,uy);
                dg::blas2::gemv(poisson.dyrhs(),phi,ux);
                dg::blas1::scal(ux,-1.0);

                dg::blas1::pointwiseDot(ux,npe[0],temp1);
                polavg(temp1,faux);
                dg::blas1::pointwiseDivide(faux,anpe,faux);
                dg::blas1::axpby(1.0,ux,-1.0,faux,ftux);
                dg::blas1::pointwiseDot(uy,npe[0],temp1);
                polavg(temp1,fauy);
                dg::blas1::pointwiseDivide(fauy,anpe,fauy);
                dg::blas1::axpby(1.0,uy,-1.0,fauy,ftuy);
                dg::blas1::pointwiseDot(ftuy,ftux,temp);
                dg::blas1::pointwiseDot(temp,npe[0],temp);
                polavg(temp,temp1);
                dg::blas1::pointwiseDivide(temp1,anpe,Rf);
            }
            
        polavg(phi,temp);      //<phi>      
        dg::blas2::gemv(interp,temp,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[4],   start1d, count1d, temp1d.data()); //<phi>      
        polavg(vor,temp);     //<nabla_perp^2 phi>      
        dg::blas2::gemv(interp,temp,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[5],   start1d, count1d, temp1d.data()); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[6],   start1d, count1d, xcoo.data());             
        dg::blas2::gemv(poisson.dxrhs(),phi,temp);
        polavg(temp,temp2);             //<u_Ey>    
        dg::blas2::gemv(interp,temp2,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[7],   start1d, count1d, temp1d.data());           
        
        //Compute the 4 terms on the RHS
        dg::blas2::gemv(poisson.dxrhs(),Rf,temp1);
        dg::blas1::scal(temp1,-1.0); //-dx Rf
        dg::blas1::axpby(1.0,temp1,0.0,temp); 
	    double Rfxnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[8],   start1d, count1d, temp1d.data()); //Rfx =- dx Rf
        dg::blas2::gemv(poisson.dxrhs(),fauy,temp1); // dx fauy
        dg::blas1::pointwiseDot(faux,temp1,temp1); // faux dx fauy
        dg::blas1::scal(temp1,-1.0);  // -  faux dx fauy
        dg::blas1::axpby(1.0,temp1,1.0,temp);  
	    double Anorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[9],   start1d, count1d, temp1d.data()); //A =  -  faux dx fauy
        dg::blas1::transform(anpe, temp2, dg::LN<double>()); //ln<n_e>
        dg::blas2::gemv(poisson.dxlhs(),temp2,temp1); // dx ln<n_e>
        dg::blas2::gemv(interp,temp1,temp1d); // dx ln<n_e> =- kappa(x)
        double invkappaavg = -p.lx/dg::blas2::dot(one1d,w1d,temp1d); //-1/(dx ln<n_e>)
        dg::blas1::scal(temp1d,-1.0);   // -dx ln<n_e>   
        dg::blas1::transform(temp1d,temp1d, dg::INVERT<double>()); // -1/dx ln<n_e>   
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[14],   start1d, count1d, temp1d.data());  //invkappa(x) = -dxln<n_e>  
        dg::blas1::pointwiseDot(Rf,temp1,temp2); // Rf dx ln<n_e>
        dg::blas1::pointwiseDot(R ,temp1,temp3); // R dx ln<n_e>
        dg::blas1::scal(temp2,-1.0); //-Rf dx ln<n_e>
        dg::blas1::axpby(1.0,temp2,1.0,temp);  
	    double Rfnnorm = sqrt(dg::blas2::dot(temp2,w2d,temp2))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp2,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[10],   start1d, count1d, temp1d.data()); //Rfn = -Rf dx ln<n_e>
        dg::blas1::scal(temp3,-1.0); //-R dx ln<n_e>
        dg::blas1::axpby(1.0,temp2,-1.0,temp3,temp2);
        double Rnnorm = sqrt(dg::blas2::dot(temp3,w2d,temp3))/p.lx/p.ly;
        double Rnffnorm = sqrt(dg::blas2::dot(temp2,w2d,temp2))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp3,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[32],   start1d, count1d, temp1d.data()); //Rn = -R dx ln<n_e>
        dg::blas1::pointwiseDot(faux,temp1,temp2); // faux dx ln<n_e>
        dg::blas1::pointwiseDot(fauy,temp2,temp1); // faux fauy dx ln<n_e>
        dg::blas1::scal(temp1,2.0);  // 2 faux fauy dx ln<n_e>
//         dg::blas1::axpby(1.0,temp1,1.0,temp);  
	    double Annorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[11],   start1d, count1d, temp1d.data()); //An = 2 faux fauy dx ln<n_e>
        double dtfauynorm = sqrt(dg::blas2::dot(temp,w2d,temp))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[12],   start1d, count1d, temp1d.data()); //Rfx+A+Rfn+An
        dg::blas2::gemv(poisson.dxrhs(),R,temp1);
        dg::blas1::scal(temp1,-1.0); //-dx R
	    double Rxnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp1,temp1d); 
        dg::blas1::axpby(1.0,temp1,1.0,temp3,temp1); //-dx R - R dx ln<n_e>
        double dfnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly;
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[13],   start1d, count1d, temp1d.data()); // Rx = - dx R
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[15],   start1d, count1d, fauy.data()); 
        dg::blas1::pointwiseDot(npe[0],uy,temp2); //n u_y
        polavg(temp2,temp1);   //< n u_y >
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[16],   start1d, count1d, temp1d.data()); //< n u_y >
        dg::blas1::pointwiseDot(R,anpe,temp2);
        dg::blas2::gemv(poisson.dxrhs(),temp2,temp1);
        dg::blas1::scal(temp1,-1.0); //temp1 = -dx ( <n_e>  R)
	    double Rnxnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly;
	    dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[17],   start1d, count1d, temp1d.data()); //Rnx = -dx ( <n_e>  R)
        dg::blas1::pointwiseDot(dne,tux,temp1); //= \delta ne \ðelta u_x 
        polavg(temp1,temp2);   //temp2 = <\delta ne \ðelta u_x >
        dg::blas1::pointwiseDot(temp2,auy,temp2); //temp2 =<u_y> <\delta ne \ðelta u_x >
        dg::blas2::gemv(poisson.dxrhs(),temp2,temp1); //temp1 = dx (<u_y> <\delta ne \ðelta u_y >)
        dg::blas1::scal(temp1,-1.0); //temp1 = -dx (<u_y> <\delta ne \ðelta u_y >)
	    double Guyxnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly; //norm of  -dx (<u_y> <\delta ne \ðelta u_y >)
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[18],   start1d, count1d, temp1d.data()); //Guyx =  -dx (<u_y> <\delta ne \ðelta u_x >)
        dg::blas1::pointwiseDot(fauy,faux,temp2); //= ||u_y|| ||u_x||
        dg::blas1::pointwiseDot(auy,faux,temp); //= <u_y> ||u_x||
        dg::blas1::axpby(-1.0,temp2,1.0,temp,temp); //||u_x|| (<u_y>- ||u_y||) = -||u_x|| ||du_y||
        dg::blas2::gemv(poisson.dxrhs(),temp,temp1); //temp1 =- dx (||u_x|| ||d u_y|| )
        dg::blas1::scal(temp1,-1.0); //temp1 = dx (||u_x|| ||du_y|| )
	    double Guynxnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly; 
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[20],   start1d, count1d, temp1d.data()); //Guynx =  dx (||u_x|| ||du_y||            
        dg::blas1::pointwiseDot(dne,tux,temp1);       //temp1 = \delta n_e \delta u_x
        dg::blas1::pointwiseDot(tuy,temp1,temp1);     //temp1 = \delta n_e \delta u_x \delta u_y
        polavg(temp1,temp2);                          //temp2 = <\delta n_e \delta u_x \delta u_y >
        dg::blas1::pointwiseDivide(temp2,anpe,temp);
        dg::blas2::gemv(poisson.dxrhs(),temp2,temp1); //temp1 = dx <\delta n_e \delta u_x \delta u_y >
        dg::blas1::scal(temp1,-1.0);                  //temp1 =-dx <\delta n_e \delta u_x \delta u_y >
	    double Txnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp1,temp1d);             
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[19], start1d, count1d, temp1d.data()); //Tx =  - dx <\delta n_e \delta u_x \delta u_y >
        dg::blas2::gemv(poisson.dxrhs(),temp,temp1); //temp1 = dx (<\delta n_e \delta u_x \delta u_y >/<n_e>)
        dg::blas1::scal(temp1,-1.0);                  //temp1 =-dx (<\delta n_e \delta u_x \delta u_y >/<n_e>)
	    double Tnxnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[21],   start1d, count1d, temp1d.data()); //Tnx =  - dx ( <\delta n_e \delta u_x \delta u_y >/<n_e> )
        
	    double netnorm = sqrt(dg::blas2::dot(ntilde[0],w2d,ntilde[0]))/p.lx/p.ly;
	    dg::blas2::gemv(interp,ntilde[0],temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[22],   start1d, count1d, temp1d.data()); //netilde
	    double neatnorm = sqrt(dg::blas2::dot(navgtilde[0],w2d,navgtilde[0]))/p.lx/p.ly;
	    dg::blas2::gemv(interp,navgtilde[0],temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[23],   start1d, count1d, temp1d.data()); //neavgtilde

        dg::blas1::transform(navgtilde[0],temp, dg::ABS<double>());
        double neatsupnorm =*thrust::max_element(temp.begin(),temp.end()); 
        
        dg::blas2::gemv(poisson.dxrhs(),fauy,temp1); 
        dg::blas1::pointwiseDivide(Rf,temp1,temp1); 
        dg::blas1::scal(temp1,-1.0);
        double nuturbnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[24],   start1d, count1d, temp1d.data()); //nuturb = -R/dx[[u_y]]
        
        double sumnorm = Rxnorm + Guynxnorm+ Tnxnorm + Anorm + Rfnnorm;
        double Rxnormscal = Rxnorm/sumnorm;
        double Guynxnormscal = Guynxnorm/sumnorm;
        double Tnxnormscal = Tnxnorm/sumnorm;
        double Anormscal = Anorm/sumnorm;
        double Annormscal = Annorm/sumnorm;
        double Rfnnormscal = Rfnnorm/sumnorm;
        double Rnnormscal = Rnnorm/sumnorm;
        double dfnormscal = dfnorm/sumnorm;
        double Rnffnormscal = Rnffnorm/sumnorm;
        
        
        
        dg::blas2::gemv(poisson.dxrhs(),faux,temp1); //dx ||u_x||
        dg::blas1::scal(temp1,-1.0); //temp1 = -dx ||u_x||
        double dxfauxnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly; 
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[25],   start1d, count1d, temp1d.data()); //dxfaux =  -dx ||u_x||
        
        dg::blas1::transform(anpe, temp2, dg::LN<double>()); //ln<n_e>
        dg::blas2::gemv(poisson.dxlhs(),temp2,temp1); // dx ln<n_e>
        dg::blas1::pointwiseDot(temp1,faux,temp1);
        dg::blas1::scal(temp1,-1.0); //temp1 = -||u_x||dx ln<n_e> 
        double fauxiknorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly; 
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[26],   start1d, count1d, temp1d.data()); //fauxik = -||u_x||dx ln<n_e>
        
        
        double vyfavgnorm = sqrt(dg::blas2::dot(fauy,w2d,fauy))/p.lx/p.ly; 
        
        //DIFFUSIVE TERMS
        //dt ln(n)
        lap.set_chi(one);
        dg::blas1::transform(npe[0], temp, dg::PLUS<>(-p.bgprofamp - p.nprofileamp));
        dg::blas2::symv(lap,temp,temp2);
        dg::blas2::symv(lap,temp2,temp);
        polavg(temp,temp1);
        dg::blas1::scal( temp1, -p.nu_perp); 
        dg::blas1::pointwiseDivide(temp1,anpe,temp1);
        double difflnnnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly; 
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[27],   start1d, count1d, temp1d.data()); //difflnn = -\nu/<n> dx^4 <n-1>         
        //dt [[u_y]]
        dg::blas1::pointwiseDot(temp1,fauy,temp1);
        dg::blas1::scal(temp1,-1.0);                 //[[u_y]] \nu/<n> dx^4(<n_e>)
        double difffauy1norm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly;
        dg::blas2::gemv(interp,temp1,temp1d);
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[28],   start1d, count1d, temp1d.data()); //difffauy1 = [[u_y]] \nu/<n> dx^4(<n_e>-<N_i>)
        dg::blas1::axpby(1.0,npe[0],-1.0,npe[1],temp); //n_e-N_i
        dg::blas2::symv(lap,temp,temp2); //-dx^2(n_e - N_i)
        dg::blas2::gemv(poisson.dxrhs(),temp2,temp); // -dx^3 (n_e - N_i) 
        polavg(temp,temp1); //- dx^3 (<n_e> - <N_i>) 
        dg::blas1::scal( temp1,p.nu_perp);  // -\nu dx^3(<n_e> - <N_i>)
        dg::blas1::pointwiseDivide(temp1,anpe,temp1);
        double difffauy2norm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly; 
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[29],   start1d, count1d, temp1d.data()); //difffauy2 =  - \nu/<n>  dx^3(<n_e> - <N_i>)
        
//         dg::blas2::symv(lap,temp2,temp);
//         polavg(temp,temp1);
//         dg::blas1::scal( temp1, -p.nu_perp);  // -\nu dx^4(<n_e> - <N_i>)
//         dg::blas2::gemv(interp,temp1,temp1d); 
//         double difffauy2temp = dg::blas2::dot(temp1d,w1d,one1d); //int_x0^x1 (-\nu dx^4(<n_e>-<N_i>)) dx
//         dg::blas1::pointwiseDivide(one,anpe,temp1);
//         dg::blas1::scal(temp1,difffauy2temp);
//         double difffauy2norm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly; 
//         dg::blas2::gemv(interp,temp1,temp1d); 
//         err_out = nc_put_vara_double( ncid_out, dataIDs1d[29],   start1d, count1d, temp1d.data()); //difffauy2 =  - \nu/<n> int_x0^x1 ( dx^4(<n_e>-<N_i>)) dx
  
        //SOURCE TERMS
        //dt ln(n)
        dg::blas1::axpby(1.0,nprof,-1.0,anpe,temp1); //ne0p - <ne>
        dg::blas1::pointwiseDot(temp1,lhs,temp1); //lhs*(ne0p - <ne>)
        dg::blas1::transform(temp1,temp1, dg::POSVALUE<double>()); //= P [lhs*(n0ep - <ne>) ]
        dg::blas1::scal(temp1,p.omega_source); // omega_s P [lhs*(ne0p - <ne>) ]
        dg::blas1::pointwiseDivide(temp1,anpe,temp1);
        double Slnnnorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly; 
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[30],   start1d, count1d, temp1d.data()); //Sn = 1/<n>  omega_s P [lhs*(ne0p - <ne>)]
        //dt [[u_y]]
        dg::blas1::pointwiseDot(temp1,fauy,temp1); 
        dg::blas1::scal(temp1,-1.0);
        double Sfauynorm = sqrt(dg::blas2::dot(temp1,w2d,temp1))/p.lx/p.ly; 
        dg::blas2::gemv(interp,temp1,temp1d); 
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[31],   start1d, count1d, temp1d.data()); //Su = -[[u_y]]/<n>  omega_s P [lhs*(ne0p - <ne>)]
        
        //Compute avg 2d fields and convert them into 1d field
        polavg(npe[0],temp1d,false);
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[0],   start1d, count1d, temp1d.data()); 
        polavg(npe[1],temp1d,false);
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[1],   start1d, count1d, temp1d.data()); 
        polavg(logn[0],temp1d,false);
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[2],   start1d, count1d, temp1d.data()); 
        polavg(logn[1],temp1d,false);
        err_out = nc_put_vara_double( ncid_out, dataIDs1d[3],   start1d, count1d, temp1d.data()); 
        
        //compute probe values by interpolation and write 2d data fields
        dg::blas2::gemv(probe_interp, ntilde[0], npe_probes);
        polavg(phi,temp);
        dg::blas2::gemv(probe_interp, phi, phi_probes);
        dg::blas2::gemv(dy, phi, temp);
        dg::blas2::gemv(probe_interp, temp, gamma_probes);
        
        //write data in netcdf file
        err_out = nc_put_vara_double( ncid_out, timevarID, start1d, count1d, &time);
        for( unsigned i=0; i<num_probes; i++){
            err_out= nc_put_vara_double( ncid_out, npe_probesID[i], start1d, count1d, &npe_probes[i]);
            err_out= nc_put_vara_double( ncid_out, phi_probesID[i], start1d, count1d, &phi_probes[i]);
            err_out= nc_put_vara_double( ncid_out, gamma_probesID[i], start1d, count1d, &gamma_probes[i]);
        }
        err_out = nc_put_vara_double( ncid_out, TperpzID, start1d, count1d, &Tperpz);
        err_out = nc_put_vara_double( ncid_out, TperpID, start1d, count1d, &Tperp);
        err_out = nc_put_vara_double( ncid_out, TperpratioID, start1d, count1d, &Tperpratio);
        err_out = nc_put_vara_double( ncid_out, OmegaID, start1d, count1d, &Omega);
        err_out = nc_put_vara_double( ncid_out, OmegazID, start1d, count1d, &Omegaz);
        err_out = nc_put_vara_double( ncid_out, OmegaratioID, start1d, count1d, &Omegaratio);
        err_out = nc_put_vara_double( ncid_out, Gamma_neID, start1d, count1d, &Gamma_ne);
        err_out = nc_put_vara_double( ncid_out, invkappaavgID, start1d, count1d, &invkappaavg);
        err_out = nc_put_vara_double( ncid_out, RfxnormID, start1d, count1d, &Rfxnorm);
        err_out = nc_put_vara_double( ncid_out, AnormID,   start1d, count1d, &Anorm);            
        err_out = nc_put_vara_double( ncid_out, RfnnormID, start1d, count1d, &Rfnnorm);
        err_out = nc_put_vara_double( ncid_out, AnnormID, start1d, count1d, &Annorm);
        err_out = nc_put_vara_double( ncid_out, RxnormID, start1d, count1d, &Rxnorm);
        err_out = nc_put_vara_double( ncid_out, RnxnormID, start1d, count1d, &Rnxnorm);
        err_out = nc_put_vara_double( ncid_out, GuyxnormID, start1d, count1d, &Guyxnorm);
        err_out = nc_put_vara_double( ncid_out, TxnormID, start1d, count1d, &Txnorm);
        err_out = nc_put_vara_double( ncid_out, GuynxnormID, start1d, count1d, &Guynxnorm);
        err_out = nc_put_vara_double( ncid_out, TnxnormID, start1d, count1d, &Tnxnorm);
        err_out = nc_put_vara_double( ncid_out, netnormID, start1d, count1d,  &netnorm);
        err_out = nc_put_vara_double( ncid_out, neatnormID, start1d, count1d, &neatnorm);
        err_out = nc_put_vara_double( ncid_out, neatsupnormID, start1d, count1d, &neatsupnorm);
        err_out = nc_put_vara_double( ncid_out, dtfauynormID, start1d, count1d, &dtfauynorm);
        err_out = nc_put_vara_double( ncid_out, RxnormscalID, start1d, count1d, &Rxnormscal);
        err_out = nc_put_vara_double( ncid_out, GuynxnormscalID, start1d, count1d, &Guynxnormscal);
        err_out = nc_put_vara_double( ncid_out, TnxnormscalID, start1d, count1d, &Tnxnormscal);
        err_out = nc_put_vara_double( ncid_out, AnormscalID, start1d, count1d, &Anormscal);
        err_out = nc_put_vara_double( ncid_out, AnnormscalID, start1d, count1d, &Annormscal);
        err_out = nc_put_vara_double( ncid_out, RfnnormscalID, start1d, count1d, &Rfnnormscal);
        err_out = nc_put_vara_double( ncid_out, nuturbnormID, start1d, count1d, &nuturbnorm);
        err_out = nc_put_vara_double( ncid_out, dxfauxnormID, start1d, count1d, &dxfauxnorm);
        err_out = nc_put_vara_double( ncid_out, fauxiknormID, start1d, count1d, &fauxiknorm);
        err_out = nc_put_vara_double( ncid_out, difflnnnormID, start1d, count1d, &difflnnnorm);
        err_out = nc_put_vara_double( ncid_out, difffauy1normID, start1d, count1d, &difffauy1norm);
        err_out = nc_put_vara_double( ncid_out, difffauy2normID, start1d, count1d, &difffauy2norm);
        err_out = nc_put_vara_double( ncid_out, SlnnnormID, start1d, count1d, &Slnnnorm);
        err_out = nc_put_vara_double( ncid_out, SfauynormID, start1d, count1d, &Sfauynorm);
        err_out = nc_put_vara_double( ncid_out, tvarIDout, start1d, count1d, &time);        
        err_out = nc_put_vara_double( ncid_out, RnnormscalID, start1d, count1d, &Rnnormscal);
        err_out = nc_put_vara_double( ncid_out, dfnormscalID, start1d, count1d, &dfnormscal);
        err_out = nc_put_vara_double( ncid_out, RnffnormscalID, start1d, count1d, &Rnffnormscal);
        err_out = nc_put_vara_double( ncid_out, vyfavgnormID, start1d, count1d, &vyfavgnorm);
	    
    }
    err = nc_close(ncid);
    err_out = nc_close(ncid_out);
    return 0;
}

