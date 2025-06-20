#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

#include "dg/algorithm.h"

#include "dg/file/file.h"
#include "parameters.h"
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

    std::vector<dg::HVec> npe(2,dg::evaluate(dg::zero,g2d));
    dg::HVec phi(dg::evaluate(dg::zero,g2d));
    dg::HVec vor(dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> logn(2,dg::evaluate(dg::zero,g2d));
    dg::HVec temp(dg::evaluate(dg::zero,g2d));
    dg::HVec temp2(dg::evaluate(dg::zero,g2d));
    dg::HVec temp_in(dg::evaluate(dg::zero,g2d_in));
    dg::HVec one(dg::evaluate(dg::one,g2d));
    dg::HVec temp1d(dg::evaluate(dg::zero,g1d));
    dg::HVec xcoo(dg::evaluate(dg::cooX1d,g1d));
//     dg::HVec y0coo(dg::evaluate(1,0.0));
    dg::HVec y0coo(dg::evaluate(dg::CONSTANT(0.0),g1d));
    dg::Average<dg::IHMatrix, dg::HVec> polavg(g2d, dg::coo2d::y);
    dg::IHMatrix interp(dg::create::interpolation(xcoo,y0coo,g2d));
    dg::IHMatrix interp_in = dg::create::interpolation(g2d,g2d_in);
    dg::Poisson<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> poisson(g2d,  p.bc_x, p.bc_y,  p.bc_x, p.bc_y);
    dg::Elliptic<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> pol(g2d, p.bc_x, p.bc_y);


    //2d field
    size_t count2d[3]  = {1, g2d_in.n()*g2d_in.Ny(), g2d_in.n()*g2d_in.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[4] = {"electrons", "ions",  "potential","vor"}; 
    int dataIDs[4]; 
    //1d profiles
    dg::file::NC_Error_Handle err_out;
    int ncid_out,dataIDs1d[8], tvarIDout;
    std::string names1d[8] =  {"neavg", "Niavg",  "ln(ne)avg","ln(Ni)avg","potentialavg","voravg","x_","vyavg"}; 
    int dim_ids2d[3],dataIDs2d[4];
    std::string names2d[1] = {"deltane"};
    size_t count1d[2]  = {1, g2d.n()*g2d.Nx()};
    size_t start1d[2]  = {0, 0};    
    size_t count2d_out[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d_out[3]  = {0, 0, 0};
    err_out = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid_out);
    err_out= nc_put_att_text( ncid_out, NC_GLOBAL, "inputfile", input.size(), input.data());
    err_out= dg::file::define_dimensions( ncid_out, dim_ids2d, &tvarIDout, g2d);
 int dim_ids1d[2] = {dim_ids2d[0],dim_ids2d[2]};
    for( unsigned i=0; i<8; i++){
        err_out = nc_def_var( ncid_out, names1d[i].data(), NC_DOUBLE, 2, dim_ids1d, &dataIDs1d[i]);
    }   
    for( unsigned i=0; i<1; i++){
        err_out  = nc_def_var(ncid_out, names2d[i].data(),  NC_DOUBLE, 3, dim_ids2d, &dataIDs2d[i]);
    }   
    err_out = nc_close(ncid_out); 
    //2d field netcdf vars read


    unsigned imin,imax;
    imin= 0;
//     std::cout << "tmin = 0 tmax =" << p.maxout*p.itstp << std::endl;
//     std::cout << "enter new imin(>0) and imax(<maxout):" << std::endl;
//     std::cin >> imin >> imax;
    time = imin*p.itstp;
    err_out = nc_open( argv[2], NC_WRITE, &ncid_out);

    constexpr unsigned num_probes = 5;
    dg::HVec xprobecoords(num_probes,1.);
    for (unsigned i=0;i<num_probes; i++) {
        xprobecoords[i] = (1+i)*p.lx/((double)(num_probes+1));
    }
    const dg::HVec yprobecoords(num_probes,p.ly/2.);
    dg::HVec gamma(phi);
    dg::HVec npe_probes(num_probes);
    dg::HVec phi_probes(num_probes);
    dg::HVec gamma_probes(num_probes);
    dg::IHMatrix probe_interp(dg::create::interpolation(xprobecoords, yprobecoords, g2d)) ;
    dg::HMatrix dy(dg::create::dy(g2d));
    //probe netcdf file
    err_out = nc_redef(ncid_out);
    int npe_probesID[num_probes],phi_probesID[num_probes],gamma_probesID[num_probes];
    int T_perp_zonalID, T_perpID, T_perpratioID, Gamma_neID;
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
    err_out = nc_def_var( ncid_out, "Uperpz",    NC_DOUBLE, 1, &timeID, &T_perp_zonalID);
    err_out = nc_def_var( ncid_out, "Uperp",     NC_DOUBLE, 1, &timeID, &T_perpID);
    err_out = nc_def_var( ncid_out, "Uperpratio",     NC_DOUBLE, 1, &timeID, &T_perpratioID);
    err_out = nc_def_var( ncid_out, "Gamma",    NC_DOUBLE, 1, &timeID, &Gamma_neID);

    err_out = nc_enddef(ncid_out);   
    err_out = nc_open( argv[2], NC_WRITE, &ncid_out);
    err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]);
    size_t steps;
    err = nc_inq_dimlen(ncid, dataIDs[0], &steps);
    steps-=1;
    imax = steps/p.itstp;
    //////////////////////////////open nc file//////////////////////////////////
    for( unsigned i=imin; i<imax; i++)//timestepping
    {
            start2d[0] = i;
            start2d_out[0] = i;
            start1d[0] = i;
            time += p.itstp*p.dt;

            std::cout << "time = "<< time <<  std::endl;

            err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]);
            err = nc_get_vara_double( ncid, dataIDs[0], start2d, count2d, temp_in.data());
            dg::blas2::gemv( interp_in, temp_in,npe[0]);
            err = nc_inq_varid(ncid, names[1].data(), &dataIDs[1]);
            err = nc_get_vara_double( ncid, dataIDs[1], start2d, count2d,  temp_in.data());
            dg::blas2::gemv( interp_in, temp_in,npe[1]);
            err = nc_inq_varid(ncid, names[2].data(), &dataIDs[2]);
            err = nc_get_vara_double( ncid, dataIDs[2], start2d, count2d,  temp_in.data());
            dg::blas2::gemv( interp_in, temp_in,phi);
            err = nc_inq_varid(ncid, names[3].data(), &dataIDs[3]);
            err = nc_get_vara_double( ncid, dataIDs[3], start2d, count2d,  temp_in.data());
            dg::blas2::gemv( interp_in, temp_in,vor);

            dg::blas1::transform(npe[0], npe[0], dg::PLUS<>(p.bgprofamp + p.nprofileamp));
            dg::blas1::transform(npe[1], npe[1], dg::PLUS<>(p.bgprofamp + p.nprofileamp));
            dg::blas1::transform( npe[0], logn[0], dg::LN<double>());
            dg::blas1::transform( npe[1], logn[1], dg::LN<double>());

  
            //Compute avg 2d fields and convert them into 1d field
            polavg(npe[0],temp1d,false);
            err_out = nc_put_vara_double( ncid_out, dataIDs1d[0],   start1d, count1d, temp1d.data()); 
            polavg(npe[1],temp1d,false);
            err_out = nc_put_vara_double( ncid_out, dataIDs1d[1],   start1d, count1d, temp1d.data()); 
            polavg(logn[0],temp1d,false);
            err_out = nc_put_vara_double( ncid_out, dataIDs1d[2],   start1d, count1d, temp1d.data()); 
            polavg(logn[1],temp1d,false);
            err_out = nc_put_vara_double( ncid_out, dataIDs1d[3],   start1d, count1d, temp1d.data()); 
            polavg(phi,temp1d,false);
            pol.variation(phi,temp2);
            double T_perp = 0.5*dg::blas2::dot( npe[1], w2d, temp2);
            pol.variation(temp,temp2);
            double T_perp_zonal = 0.5*dg::blas2::dot( npe[1], w2d, temp2);   
            double T_perpratio = T_perp_zonal/T_perp;
            dg::blas2::gemv( poisson.dyrhs(), phi, temp2); 
            double Gamma_ne = -1.* dg::blas2::dot(npe[0],w2d,temp2);
            dg::blas2::gemv(interp,temp,temp1d); 
            err_out = nc_put_vara_double( ncid_out, dataIDs1d[4],   start1d, count1d, temp1d.data()); 
            polavg(vor,temp1d,false);
            err_out = nc_put_vara_double( ncid_out, dataIDs1d[5],   start1d, count1d, temp1d.data()); 

            err_out = nc_put_vara_double( ncid_out, dataIDs1d[6],   start1d, count1d, xcoo.data());
            dg::blas2::gemv(poisson.dxrhs(),phi,temp);
            polavg(temp,temp1d,false);
            err_out = nc_put_vara_double( ncid_out, dataIDs1d[7],   start1d, count1d, temp1d.data()); 

            //compute probe values by interpolation and write 2d data fields
            //normalize
                polavg(npe[0],temp);
                dg::blas1::pointwiseDivide(npe[0],temp,temp);
                dg::blas1::axpby(1.0,temp,-1.0,one,temp);
            err_out = nc_put_vara_double( ncid_out, dataIDs2d[0], start2d_out, count2d_out, temp.data());

            dg::blas2::gemv(probe_interp, temp, npe_probes);

                polavg(phi,temp);
                //do not normalise to fluctuations if <phi>=0
                dg::blas1::pointwiseDivide(phi,temp,temp);
                dg::blas1::axpby(1.0,temp,-1.0,one,temp);
            dg::blas2::gemv(probe_interp, temp, phi_probes);
//                 dg::blas2::gemv(probe_interp, phi, phi_probes);

            dg::blas2::gemv(dy, phi, temp);
            dg::blas2::gemv(probe_interp, temp, gamma_probes);

//             dg::blas2::gemv(probe_interp, npe[0], npe_probes);
//             dg::blas2::gemv(probe_interp, phi, phi_probes);
//             dg::blas2::gemv(dy, phi, temp);
//             dg::blas2::gemv(probe_interp, temp, gamma_probes);

            //write data in netcdf file
            err_out = nc_put_vara_double( ncid_out, timevarID, start1d, count1d, &time);
            for( unsigned i=0; i<num_probes; i++){
                err_out= nc_put_vara_double( ncid_out, npe_probesID[i], start1d, count1d, &npe_probes[i]);
                err_out= nc_put_vara_double( ncid_out, phi_probesID[i], start1d, count1d, &phi_probes[i]);
                err_out= nc_put_vara_double( ncid_out, gamma_probesID[i], start1d, count1d, &gamma_probes[i]);
            }
            err_out = nc_put_vara_double( ncid_out, T_perp_zonalID, start1d, count1d, &T_perp_zonal);
            err_out = nc_put_vara_double( ncid_out, T_perpID, start1d, count1d, &T_perp);
            err_out = nc_put_vara_double( ncid_out, T_perpratioID, start1d, count1d, &T_perpratio);
            err_out = nc_put_vara_double( ncid_out, Gamma_neID, start1d, count1d, &Gamma_ne);
            err_out = nc_put_vara_double( ncid_out, tvarIDout, start1d, count1d, &time);

    }
    err = nc_close(ncid);

    err_out = nc_close(ncid_out);
    return 0;
}

