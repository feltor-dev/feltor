#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

#include "dg/algorithm.h"
#include "dg/backend/xspacelib.cuh"
#include "file/nc_utilities.h"
#include "impurities/parameters.h"

#include <boost/timer.hpp>   //e3r:tbr
// boost::timer t;
// double duration;
//
// t.restart();
// duration = t.elapsed();

struct Heaviside2d
{   Heaviside2d( double sigma):sigma2_(sigma*sigma), x_(0), y_(0) {}
    void set_origin( double x0, double y0)
    {   x_=x0, y_=y0;
    }
    double operator()(double x, double y)
    {   double r2 = (x-x_)*(x-x_)+(y-y_)*(y-y_);
        if( r2 >= sigma2_)
            return 0.;
        return 1.;
    }
private:
    const double sigma2_;
    double x_,y_;
};

int main( int argc, char* argv[])
{   if( argc != 3)   // lazy check: command line parameters
    {   std::cerr << "Usage: "<< argv[0] <<" [input.nc] [output.nc]\n";
        return -1;
    }
    std::cout << argv[1 ]<< " -> " << argv[2]<<std::endl;
    ////////process parameter from .nc datafile////////
    file::NC_Error_Handle err_in;
    int ncid_in;
    err_in = nc_open(argv[1], NC_NOWRITE, &ncid_in);
    //read & print parameter string
    size_t length;
    err_in = nc_inq_attlen( ncid_in, NC_GLOBAL, "inputfile", &length);
    std::string input(length, 'x');
    err_in = nc_get_att_text( ncid_in, NC_GLOBAL, "inputfile", &input[0]);
    std::cout << "input "<<input<<std::endl;
    //parse: parameter string--json-->p.xxx
    Json::Reader reader;
    Json::Value js;
    reader.parse(input, js, false);
    const imp::Parameters p(js);
    p.display(std::cout);
    //.nc parameter
    int dim_et_id;
    size_t num_etime;
    err_in = nc_inq_dimid(ncid_in, "energy_time", &dim_et_id);
    err_in = nc_inq_dimlen(ncid_in, dim_et_id, &num_etime);
//    err_in = nc_close( ncid_in);
    ////////compose data////////
    const size_t num_species = 3, num_fields = 2, num_errsets = 2;
    const size_t num_spatial = 14;
    const size_t num_err_time = 6, num_err_etime = 5;
    int species_id[num_species], field_id[num_fields], errset_id[num_errsets];
    int species_idf[num_species], field_idf[num_fields];
    int species_phys_idf[num_species], fieldphys_idf[num_fields];
    int physfield_id, spatial_id[num_spatial];
    int err_time_id[num_err_time], err_etime_id[num_err_etime];
    //groups
    std::string species_name[num_species] =
    {   "electrons", "ions", "impurities"
    };
    std::string field_name[num_fields] =
    {   "potential", "vorticity"
    };
    std::string errset[num_errsets] =
    {   "err_time", "err_etime"
    };
    //species:
    std::string spatial[num_spatial] =
    {   "posX" , "posY" , "velX" , "velY" ,
        "accX" , "accY", "posXmax", "posYmax", "velXmax", "velYmax",
        "compactness", "velCOM", "maxamp", "mass"
    };
    //err_time
    std::string err_time[num_err_time] =
    {   "Se", "Si", "Sz",
        "Uphii", "Uphiz", "cn"
    };
    //err_etime
    std::string err_etime[num_err_etime] =
    {   "energy", "mass", "dissipation",
        "dEdt", "accuracy"
    };

    ////////grid////////
    dg::Grid2d<double > g2d( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
    const double hx = g2d.hx()/(double)g2d.n();
    const double hy = g2d.hy()/(double)g2d.n();
    unsigned Nx = p.Nx_out*p.n_out;
    unsigned Ny = p.Ny_out*p.n_out;
    dg::DVec xvec = dg::evaluate( dg::coo1, g2d);
    dg::DVec yvec = dg::evaluate( dg::coo2, g2d);
    dg::DVec one = dg::evaluate( dg::one, g2d);
    dg::DVec w2d = dg::create::weights( g2d);
    dg::DVec helper(dg::evaluate( dg::zero, g2d));
    dg::IDMatrix equi = dg::create::backscatter( g2d);
    //.nc structures
    size_t count2d[3] = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3] = {0, 0, 0};
    size_t count0d[1] = {1};
    size_t start0d[1] = {0};

    ////////create .nc structure////////
    int shuffle_flag = 0, compress_flag = 0, compress_level = 5;
    file::NC_Error_Handle err_out;
    int ncid_out, dim_t_x_y_id[3], tvar_id, etvar_id;
    err_out = nc_create( argv[2], NC_NETCDF4|NC_CLOBBER, &ncid_out);
    err_out = nc_put_att_text( ncid_out, NC_GLOBAL, "inputfile", input.size(), input.data());
    err_out = file::define_limtime_xy( ncid_out, dim_t_x_y_id, p.maxout ,&tvar_id, g2d);
    err_out = file::define_limited_time( ncid_out, "etime", num_etime, &dim_et_id, &etvar_id);   //tbc!
    //species
    for (unsigned i = 0; i < num_species; i++)
    {   err_out = nc_def_grp( ncid_out, species_name[i].data(), &species_idf[i]);
        err_out = nc_def_var( species_idf[i], "physfield", NC_DOUBLE, 3, dim_t_x_y_id, &species_phys_idf[i]);
        err_out = nc_def_var_deflate( species_idf[i], species_phys_idf[i], shuffle_flag, compress_flag, compress_level);
        for (unsigned j = 0; j < num_spatial; j++)
        {   err_out = nc_def_var( species_idf[i], spatial[j].data(), NC_DOUBLE, 1, &dim_t_x_y_id[0], &spatial_id[j]);
            err_out = nc_def_var_deflate( species_idf[i], spatial_id[j], shuffle_flag, compress_flag, compress_level);
        }
    }
    //fields
    for (unsigned i = 0; i < num_fields; i++)
    {   err_out = nc_def_grp( ncid_out, field_name[i].data(), &field_id[i]);
        err_out = nc_def_var( field_id[i], "physfield", NC_DOUBLE, 3, dim_t_x_y_id, &physfield_id);
        err_out = nc_def_var_deflate( field_id[i], physfield_id, shuffle_flag, compress_flag, compress_level);
    }
    //error measures time
    err_out = nc_def_grp( ncid_out, "err_time", &errset_id[0]);
    for (unsigned i = 0; i < num_err_time; i++)
    {   err_out = nc_def_var( errset_id[0], err_time[i].data(), NC_DOUBLE, 1, &dim_t_x_y_id[0], &err_time_id[i]);
        err_out = nc_def_var_deflate( errset_id[0], err_time_id[i], shuffle_flag, compress_flag, compress_level);
    }
    //error measures etime
    err_out = nc_def_grp( ncid_out, "err_etime", &errset_id[1]);
    for (unsigned i = 0 ; i < num_err_etime; i++)
    {   err_out = nc_def_var( errset_id[1], err_etime[i].data(), NC_DOUBLE, 1, &dim_et_id, &err_etime_id[i]);
        err_out = nc_def_var_deflate( errset_id[1], err_etime_id[i], shuffle_flag, compress_flag, compress_level);
    }
    err_out = nc_enddef(ncid_out);

    ////////input data: 2d time series////////
    double deltaT = p.dt*p.itstp;
    // std::vector<dg::DVec> npe(num_species, dg::evaluate(dg::zero, g2d));
    // std::vector<dg::DVec> ntilde(num_species, dg::evaluate(dg::zero, g2d));
    // std::vector<dg::DVec> lnn(num_species, dg::evaluate(dg::zero, g2d));
    // std::vector<dg::DVec> field(num_fields, dg::evaluate(dg::zero, g2d));
    // std::vector<dg::HVec> field_host(num_fields, dg::evaluate(dg::zero, g2d));
    // std::vector<dg::HVec> npe_h(3, dg::evaluate(dg::zero, g2d));
    // //eval field
    // dg::ArakawaX< dg::CartesianGrid2d, dg::DMatrix, dg::DVec> arakawa(g2d);
    // //eval particle densities
    // const dg::DVec binv( dg::evaluate(dg::LinearX(p.kappa, 1.), g2d));
    // dg::DVec chi = dg::evaluate(dg::zero, g2d);
    // dg::DVec gamma_n = dg::evaluate(dg::zero, g2d);
    // dg::Helmholtz< dg::CartesianGrid2d, dg::DMatrix, dg::DVec> gamma_s(g2d, 1.0, dg::centered);
    // dg::Elliptic< dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol(g2d, dg::normed, dg::centered);
    // dg::Invert< dg::DVec > invert_invgamma(chi, chi.size(), p.eps_gamma);
    // //calculation variables per species
    // double mass_[num_species] = {}, cn[num_species] = {};
    // double posX[num_species] = {}, posY[num_species] = {};
    // double posX_init[num_species] = {}, posY_init[num_species] = {};
    // double posX_old[num_species] = {} ,posY_old[num_species] = {};
    // double posX_max[num_species] = {}, posY_max[num_species] = {};
    // double posX_max_old[num_species] = {}, posY_max_old[num_species] = {};
    // double posX_max_hs[num_species] = {}, posY_max_hs[num_species] = {};
    // double velX[num_species] = {}, velY[num_species] = {};
    // double velX_old[num_species] = {}, velY_old[num_species] = {};
    // double velX_max[num_species] = {}, velY_max[num_species] = {};
    // double velCOM[num_species] = {};
    // double accX[num_species] = {}, accY[num_species] = {};
    // double compactness[num_species] = {};
    // unsigned position[num_species] = {};
    // double maxamp[num_species];
    // Heaviside2d heavi(2.0* p.sigma);
    // double normalize = 1.;
    // dg::DVec heavy;
    // dg::HVec transfer2d(dg::evaluate(dg::zero,g2d));
    double time = 0.;

//    err_in = nc_open(argv[1], NC_NOWRITE, &ncid_in);
//    err_out = nc_open(argv[2], NC_WRITE, &ncid_out);
    //transfer etime
    double transfer_etime[num_etime];
    static size_t count_etime[] = {num_etime};
    static size_t start_etime[] = {0};
    for (unsigned i = 0; i < num_err_etime; i++)
    {   err_in = nc_inq_varid(ncid_in, err_etime[i].data(), &err_etime_id[i]);
        err_in = nc_get_vara_double(ncid_in, err_etime_id[i], start_etime, count_etime, transfer_etime);
        err_out = nc_inq_grp_ncid(ncid_out, "err_etime", &errset_id[0]);
        err_out = nc_inq_varid(errset_id[0], err_etime[i].data(), &err_etime_id[i]);
        err_out = nc_put_vara_double(errset_id[0], err_etime_id[i], start_etime, count_etime, transfer_etime);
    }

    //--//
    boost::timer t;   //e3r:tbr
    double duration;  //e3r:tbr
    //--//

    int cache_size = Nx*Ny*4*100, cache_nelems = 100;
    std::cout << cache_size << "\n";
    double cache_preemption = 0.8;
    err_out = nc_set_var_chunk_cache(species_idf[0], species_phys_idf[0], cache_size, cache_nelems, cache_preemption);
    err_out = nc_set_var_chunk_cache(species_idf[1], species_phys_idf[1], cache_size, cache_nelems, cache_preemption);
    err_out = nc_set_var_chunk_cache(species_idf[2], species_phys_idf[2], cache_size, cache_nelems, cache_preemption);
    ////////timestepping////////
    for (unsigned i = 0; i < p.maxout; i++)
    {   start2d[0] = i;
        start0d[0] = i;
        if (i>0)
        {   time += p.itstp*p.dt;
        }
        std::cout << i << "\n";   // tbr!
        for (unsigned j = 0; j < num_fields; j++)
        {   err_in = nc_inq_varid(ncid_in, field_name[j].data(), &field_id[j]);
            err_in = nc_get_vara_double(ncid_in, field_id[j], start2d, count2d, field_host[j].data());
            field[j] = field_host[j];
            err_out = nc_inq_grp_ncid(ncid_out, field_name[j].data(), &field_id[j]);
            err_out = nc_inq_varid(field_id[j], "physfield", &physfield_id);
            dg::blas1::transfer(field[j], transfer2d);
            //--//
            t.restart();
            err_out = nc_put_vara_double(field_id[j], physfield_id, start2d, count2d, transfer2d.data());
            duration = t.elapsed();
            std::cout << "written field in: " << duration << "\n";
            //--//
        }
        for (unsigned j = 0; j < num_species; j++)
        {   err_in = nc_inq_varid(ncid_in, species_name[j].data(), &species_rphyid[j]);
            err_in = nc_get_vara_double(ncid_in, species_id[j], start2d, count2d, npe_h[j].data());
            npe[j] = npe_h[j];
            dg::blas1::plus( npe[j], 1);
            dg::blas1::transform( npe[j], ntilde[j], dg::PLUS<double>(-1));
//            err_out = nc_inq_grp_ncid(ncid_out, species_name[j].data(), &species_id[j]);
//            err_out = nc_inq_varid(species_id[j], "physfield", &physfield_id);
            if( j == 0)
            {   //err_out = nc_put_vara_double(species_idf[j], species_phys_idf[j], start2d, count2d, npe_h[j].data());
            }
            else
            {
                gamma_s.alpha() = -0.5*p.tau[j]*p.mu[j];
                invert_invgamma(gamma_s, gamma_n, ntilde[j]);

                dg::blas1::axpby(-p.a[j]*p.mu[j], ntilde[j], 0., chi);
                dg::blas1::pointwiseDot(chi, binv, chi);
                dg::blas1::pointwiseDot(chi, binv, chi);
                pol.set_chi(chi);
                pol.symv(field[0], ntilde[j]);
                dg::blas1::axpby( 1., gamma_n, 1., ntilde[j]);

                dg::blas1::transfer(ntilde[j], transfer2d);
                //--//
                t.restart();
                err_out = nc_put_vara_double(species_idf[j], species_phys_idf[j], start2d, count2d, transfer2d.data());
                duration = t.elapsed();
                std::cout << "written heavy in: " << duration << "\n";
                //--//
            }
            for (unsigned k = 0; k < num_spatial; k++)
            {   err_out = nc_inq_varid(species_idf[j], spatial[k].data(), &spatial_id[k]);
            }

            //mass
            mass_[j] = dg::blas2::dot(one, w2d, ntilde[j]);
            err_out = nc_put_vara_double( species_idf[j], spatial_id[13], start0d, count0d, &mass_[j]);
            //charge number
            dg::blas1::axpby(p.a[j], one, 0., helper);
            cn[j] = dg::blas2::dot(helper, w2d, ntilde[j]);
            //position, velocity, acceleration
            if (i==0)
            {   posX_init[j] = dg::blas2::dot( xvec, w2d, ntilde[j])/mass_[j];
                posY_init[j] = dg::blas2::dot( yvec, w2d, ntilde[j])/mass_[j];
            }
            if (i>0)
            {   posX[j] = dg::blas2::dot( xvec, w2d, ntilde[j])/mass_[j]-posX_init[j];
                posY[j] = dg::blas2::dot( yvec, w2d, ntilde[j])/mass_[j]-posY_init[j];
            }
            if (i==0)
            {   velX_old[j] = -posX[j]/deltaT;
                velY_old[j] = -posY[j]/deltaT;
                posX_old[j] = posX[j];
                posY_old[j] = posY[j];
            }
            velX[j] = (posX[j] - posX_old[j])/deltaT;
            velY[j] = (posY[j] - posY_old[j])/deltaT;
            velCOM[j]=sqrt(velX[j]*velX[j]+velY[j]*velY[j]);
            accX[j] = (velX[j] - velX_old[j])/deltaT;
            accY[j] = (velY[j] - velY_old[j])/deltaT;
            if (i>0)
            {   posX_old[j] = posX[j];
                posY_old[j] = posY[j];
                velX_old[j] = velX[j];
                velY_old[j] = velY[j];
            }
            err_out = nc_put_vara_double( species_idf[j], spatial_id[0],  start0d, count0d, &posX[j]);
            err_out = nc_put_vara_double( species_idf[j], spatial_id[1],  start0d, count0d, &posY[j]);
            err_out = nc_put_vara_double( species_idf[j], spatial_id[2],  start0d, count0d, &velX[j]);
            err_out = nc_put_vara_double( species_idf[j], spatial_id[3],  start0d, count0d, &velY[j]);
            err_out = nc_put_vara_double( species_idf[j], spatial_id[4],  start0d, count0d, &accX[j]);
            err_out = nc_put_vara_double( species_idf[j], spatial_id[5],  start0d, count0d, &accY[j]);
            err_out = nc_put_vara_double( species_idf[j], spatial_id[11], start0d, count0d, &velCOM[j]);
            // maximal amplitude
            if ( p.amp > 0)
                maxamp[j] = *thrust::max_element( npe[j].begin(), npe[j].end());
            else
                maxamp[j] = *thrust::min_element( npe[j].begin(), npe[j].end());
            err_out = nc_put_vara_double( species_idf[j], spatial_id[12], start0d, count0d, &maxamp[j]);

            //get max position and value(x,y_max)
            dg::blas2::gemv( equi, npe[j], helper);
            position[j] = thrust::distance( helper.begin(), thrust::max_element( helper.begin(), helper.end()) );
            posX_max[j] = hx*(1./2. + (double)(position[j]%Nx))-posX_init[j];
            posY_max[j] = hy*(1./2. + (double)(position[j]/Nx))-posY_init[j];   // Nx->Ny?
            posX_max_hs[j] = hx*(1./2. + (double)(position[j]%Nx));
            posY_max_hs[j] = hy*(1./2. + (double)(position[j]/Nx));   // Nx->Ny?
            err_out = nc_put_vara_double( species_idf[j], spatial_id[6], start0d, count0d, &posX_max[j]);
            err_out = nc_put_vara_double( species_idf[j], spatial_id[7], start0d, count0d, &posY_max[j]);
            velX_max[j] = (posX_max[j] - posX_max_old[j])/deltaT;
            velY_max[j] = (posY_max[j] - posY_max_old[j])/deltaT;
            if (i>0)
            {   posX_max_old[j] = posX_max[j];
                posY_max_old[j] = posY_max[j];
            }
            err_out = nc_put_vara_double( species_idf[j], spatial_id[8], start0d, count0d, &velX_max[j]);
            err_out = nc_put_vara_double( species_idf[j], spatial_id[9], start0d, count0d, &velY_max[j]);
            //compactness
            if (i==0)
            {   heavi.set_origin( posX_max_hs[j], posY_max_hs[j] );
                heavy = dg::evaluate( heavi, g2d);
                normalize = dg::blas2::dot( heavy, w2d, ntilde[j]);
            }
            heavi.set_origin( posX_max_hs[j], posY_max_hs[j]);
            heavy = dg::evaluate( heavi, g2d);
            compactness[j] =  dg::blas2::dot( heavy, w2d, ntilde[j])/normalize;
            err_out = nc_put_vara_double(species_idf[j], spatial_id[10], start0d, count0d, &compactness[j]);
            //energy
            dg::blas1::transform( npe[j], lnn[j], dg::LN<double>());
        }
        //field
        arakawa.variation(field[0], helper);
        double energy[5] = {};
        energy[0] = dg::blas2::dot( npe[0], w2d, lnn[0]);
        energy[1] = p.a[1]*p.tau[1]*dg::blas2::dot( npe[1], w2d, lnn[1]);
        energy[2] = p.a[2]*p.tau[2]*dg::blas2::dot( npe[2], w2d, lnn[2]);
        energy[3] = 0.5*p.a[1]*p.mu[1]*dg::blas2::dot( npe[1], w2d, helper);
        energy[4] = 0.5*p.a[2]*p.mu[2]*dg::blas2::dot( npe[2], w2d, helper);

        err_out = nc_inq_grp_ncid(ncid_out, "err_time", &errset_id[1]);
        err_out = nc_inq_varids(errset_id[1], 0, err_time_id);
        for (unsigned j = 0; j < 5; j++)
        { err_out = nc_put_vara_double(errset_id[1], err_time_id[j], start0d, count0d, &energy[j]);
        }
        double dcn = cn[0]+cn[1]+cn[2];
        err_out = nc_put_vara_double(errset_id[1], err_time_id[5], start0d, count0d, &dcn);
    }
    err_out = nc_close(ncid_out);
    err_in = nc_close(ncid_in);
    return 0;
}
