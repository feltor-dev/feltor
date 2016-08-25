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

#include <boost/timer.hpp>
// boost::timer t;
// double duration;
//
// t.restart();
// duration = t.elapsed();

struct Heaviside2d
{ Heaviside2d( double sigma):sigma2_(sigma*sigma), x_(0), y_(0) {}
  void set_origin( double x0, double y0)
  { x_ = x0, y_ = y0;
  }
  double operator()(double x, double y)
  { double r2 = (x-x_)*(x-x_)+(y-y_)*(y-y_);
    if( r2 >= sigma2_)
      return 0.;
    return 1.;
  }
private:
  const double sigma2_;
  double x_,y_;
};

int main( int argc, char* argv[])
{ if( argc != 3)   // lazy check: command line parameters
  { std::cerr << "Usage: "<< argv[0] <<" [input.nc] [output.nc]\n";
    return -1;
  }
  std::cout << argv[1] << " -> " << argv[2]<<std::endl;
  ////////process parameter from .nc datafile////////
  file::NC_Error_Handle err_in;
  int ncid_in;
  err_in = nc_open(argv[1], NC_NOWRITE, &ncid_in);
  //read & print parameter string
  size_t length;
  err_in = nc_inq_attlen(ncid_in, NC_GLOBAL, "inputfile", &length);
  std::string input(length, 'x');
  err_in = nc_get_att_text(ncid_in, NC_GLOBAL, "inputfile", &input[0]);
  std::cout << "input "<< input << std::endl;
  //parse: parameter string--json-->p.xxx
  Json::Reader reader;
  Json::Value js;
  reader.parse(input, js, false);
  const imp::Parameters p(js);
  p.display(std::cout);
  //.nc parameter: etime not found in .json file
  int dim_et_id;
  size_t num_etime;
  err_in = nc_inq_dimid(ncid_in, "energy_time", &dim_et_id);
  err_in = nc_inq_dimlen(ncid_in, dim_et_id, &num_etime);
  ////////compose data////////
  const size_t num_species = 3, num_fields = 2;
  const size_t num_spatial = 14;
  const size_t num_err_time = 6, num_err_etime = 5;
  int species_rgyro_id[num_species], field_rphys_id[num_species];
  int species_wgrp_id[num_species];
  int species_wphys_id[num_species], field_wphys_id[num_fields];
  int species_wspatial_id[num_species*num_spatial];
  int err_time_wgrp_id, err_etime_wgrp_id;
  int err_time_wval_id[num_err_time];
  int err_etime_rval_id[num_err_etime], err_etime_wval_id[num_err_etime];
  //groups
 std::string species_grp_name[num_species] =
  { "electrons_analysed", "ions_analysed", "impurities_analysed"
  };
  std::string species_name[num_species] =
  { "electrons", "ions", "impurities"
  };
  std::string field_name[num_fields] =
  { "potential", "vorticity"
  };
  std::string spatial[num_spatial] =
  { "posX" , "posY" , "velX" , "velY" ,
    "accX" , "accY", "posXmax", "posYmax", "velXmax", "velYmax",
    "compactness", "velCOM", "maxamp", "mass"
  };
  std::string err_time[num_err_time] =
  { "Se", "Si", "Sz",
    "Uphii", "Uphiz", "dcn"
  };
  std::string err_etime[num_err_etime] =
  { "energy", "mass", "dissipation",
    "dEdt", "accuracy"
  };
  ////////grid////////
  dg::Grid2d<double > g2d(0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
  const double hx = g2d.hx()/(double)g2d.n();
  const double hy = g2d.hy()/(double)g2d.n();
  unsigned Nx = p.Nx_out*p.n_out;
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
  ////////input data: 2d time series////////
  double deltaT = p.dt*p.itstp;
  std::vector<dg::DVec> npe(num_species, dg::evaluate(dg::zero, g2d));
  std::vector<dg::DVec> ntilde(num_species, dg::evaluate(dg::zero, g2d));
  std::vector<dg::DVec> nphys(num_species, dg::evaluate(dg::zero, g2d));
  std::vector<dg::DVec> lnn(num_species, dg::evaluate(dg::zero, g2d));
  std::vector<dg::DVec> field(num_fields, dg::evaluate(dg::zero, g2d));
  std::vector<dg::HVec> field_host(num_fields, dg::evaluate(dg::zero, g2d));
  std::vector<dg::HVec> npe_h(3, dg::evaluate(dg::zero, g2d));
  //eval field
  dg::ArakawaX< dg::CartesianGrid2d, dg::DMatrix, dg::DVec> arakawa(g2d);
  //eval particle densities
  const dg::DVec binv( dg::evaluate(dg::LinearX(p.kappa, 1.), g2d));
  dg::DVec chi = dg::evaluate(dg::zero, g2d);
  dg::DVec gamma_n = dg::evaluate(dg::zero, g2d);
  dg::Helmholtz< dg::CartesianGrid2d, dg::DMatrix, dg::DVec> gamma_s(g2d, 1.0, dg::centered);
  dg::Elliptic< dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol(g2d, dg::normed, dg::centered);
  dg::Invert< dg::DVec > invert_invgamma(chi, chi.size(), p.eps_gamma);
  //calculation variables per species
  double mass_[num_species] = {}, cn[num_species] = {};
  double posX = 0, posY = 0;
  double posX_init = 0, posY_init = 0;
  double posX_old = 0 ,posY_old = 0;
  double posX_max = 0, posY_max = 0;
  double posX_max_old = 0, posY_max_old = 0;
  double posX_max_hs = 0, posY_max_hs = 0;
  double velX = 0, velY = 0;
  double velX_old = 0, velY_old = 0;
  double velX_max = 0, velY_max = 0;
  double velCOM = 0;
  double accX = 0, accY = 0;
  double compactness = 0;
  unsigned position = 0;
  double maxamp = 0;
  Heaviside2d heavi(2.0* p.sigma);
  double normalize = 1.;
  dg::DVec heavy;
  dg::HVec transfer2d(dg::evaluate(dg::zero,g2d));
  double time = 0.;
  ////////compose .nc structures////////
  // --- cache and compress switches --- //
  int shuffle_flag = 0, compress_flag = 0, compress_level = 5;
  int nelems = 100;
  // --- --- //
  int cache_size = (transfer2d.capacity()*sizeof(transfer2d[0])+sizeof(transfer2d))*nelems;
  int cache_nelems = nelems;
  double cache_preemption = 0.9;
  file::NC_Error_Handle err_out;
  int ncid_out, dim_t_x_y_id[3], tvar_id, etvar_id;
  err_out = nc_create(argv[2], NC_NETCDF4|NC_CLOBBER, &ncid_out);
  err_out = nc_put_att_text(ncid_out, NC_GLOBAL, "inputfile",
                            input.size(), input.data());
  err_out = file::define_limtime_xy(ncid_out, dim_t_x_y_id, p.maxout+1,
                                    &tvar_id, g2d);
  err_out = file::define_limited_time(ncid_out, "etime", num_etime,
                                      &dim_et_id, &etvar_id);
  int k = 0;
  for (unsigned i = 0; i < num_species; i++)
  { err_in = nc_inq_varid(ncid_in, species_name[i].data(), &species_rgyro_id[i]);;
    err_out = nc_def_var(ncid_out, species_name[i].data(), NC_DOUBLE, 3, dim_t_x_y_id, &species_wphys_id[i]);
    err_out = nc_def_var_deflate(ncid_out, species_wphys_id[i], shuffle_flag, compress_flag, compress_level);
    err_out = nc_set_var_chunk_cache(ncid_out, species_wphys_id[i], cache_size, cache_nelems, cache_preemption);
    err_out = nc_def_grp(ncid_out, species_grp_name[i].data(), &species_wgrp_id[i]);
    for (unsigned j = 0; j < num_spatial; j++)
    { err_out = nc_def_var(species_wgrp_id[i], spatial[j].data(), NC_DOUBLE, 1, &dim_t_x_y_id[0], &species_wspatial_id[k]);
      err_out = nc_def_var_deflate(species_wgrp_id[i], species_wspatial_id[k], shuffle_flag, compress_flag, compress_level);
      k++;
    }
  }
  for (unsigned i = 0; i < num_fields; i++)
  { err_in = nc_inq_varid(ncid_in, field_name[i].data(), &field_rphys_id[i]);
    err_out = nc_def_var(ncid_out, field_name[i].data(), NC_DOUBLE, 3, dim_t_x_y_id, &field_wphys_id[i]);
    err_out = nc_def_var_deflate(ncid_out, field_wphys_id[i], shuffle_flag, compress_flag, compress_level);
    err_out = nc_set_var_chunk_cache(ncid_out, field_wphys_id[i], cache_size, cache_nelems, cache_preemption);
  }
  err_out = nc_def_grp( ncid_out, "err_time", &err_time_wgrp_id);
  for (unsigned i = 0; i < num_err_time; i++)
  { err_out = nc_def_var(err_time_wgrp_id, err_time[i].data(), NC_DOUBLE, 1, &dim_t_x_y_id[0], &err_time_wval_id[i]);
    err_out = nc_def_var_deflate( err_time_wgrp_id, err_time_wval_id[i], shuffle_flag, compress_flag, compress_level);
  }
  err_out = nc_def_grp( ncid_out, "err_etime", &err_etime_wgrp_id);
  for (unsigned i = 0 ; i < num_err_etime; i++)
  { err_in = nc_inq_varid(ncid_in, err_etime[i].data(), &err_etime_rval_id[i]);
    err_out = nc_def_var( err_etime_wgrp_id, err_etime[i].data(), NC_DOUBLE, 1, &dim_et_id, &err_etime_wval_id[i]);
    err_out = nc_def_var_deflate( err_etime_wgrp_id, err_etime_wval_id[i], shuffle_flag, compress_flag, compress_level);
  }
  err_out = nc_enddef(ncid_out);
  ////////remap data////////
  //err_etime
  double transfer_etime[num_etime];
  static size_t count_etime[] = {num_etime};
  static size_t start_etime[] = {0};
  for (unsigned i = 0; i < num_err_etime; i++)
  { err_in = nc_get_vara_double(ncid_in, err_etime_rval_id[i], start_etime, count_etime, transfer_etime);
    err_out = nc_put_vara_double(err_etime_wgrp_id, err_etime_wval_id[i], start_etime, count_etime, transfer_etime);
  }
  //timestepping
  int add_to_spatial_idx = 0;
  for (unsigned i = 0; i <= p.maxout; i++)
  { std::cout << i << "\n";   // tbr!
    start2d[0] = i;
    start0d[0] = i;
    if (i>0)
    { time += p.itstp*p.dt;
    }
    for (unsigned j = 0; j < num_fields; j++)
    { err_in = nc_get_vara_double(ncid_in, field_rphys_id[j], start2d, count2d, field_host[j].data());
      err_out = nc_put_vara_double(ncid_out, field_wphys_id[j], start2d, count2d, field_host[j].data());
      field[j] = field_host[j];
    }
    for (unsigned j = 0; j < num_species; j++)
    { err_in = nc_get_vara_double(ncid_in, species_rgyro_id[j], start2d, count2d, npe_h[j].data());
      npe[j] = npe_h[j];
      dg::blas1::plus(npe[j], 1);
      dg::blas1::transform(npe[j], ntilde[j], dg::PLUS<double>(-1));
      if( j == 0)
      { err_out = nc_put_vara_double(ncid_out, species_wphys_id[j], start2d, count2d, npe_h[j].data());
        nphys[j] = ntilde[j];
      }
      else
      { gamma_s.alpha() = -0.5*p.tau[j]*p.mu[j];
        invert_invgamma(gamma_s, gamma_n, ntilde[j]);
        dg::blas1::axpby(-p.a[j]*p.mu[j], ntilde[j], 0., chi);
        dg::blas1::pointwiseDot(chi, binv, chi);
        dg::blas1::pointwiseDot(chi, binv, chi);
        pol.set_chi(chi);
        pol.symv(field[0], nphys[j]);
        dg::blas1::axpby( 1., gamma_n, 1., nphys[j]);
        dg::blas1::transfer(nphys[j], transfer2d);
        err_out = nc_put_vara_double(ncid_out, species_wphys_id[j], start2d, count2d, transfer2d.data());
      }
      //charge number
      dg::blas1::axpby(p.a[j], one, 0., helper);
      cn[j] = dg::blas2::dot(helper, w2d, nphys[j]);

      add_to_spatial_idx = j*num_spatial;
      //mass
      mass_[j] = dg::blas2::dot(one, w2d, ntilde[j]);
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[13+add_to_spatial_idx], start0d, count0d, &mass_[j]);
      //position, velocity, acceleration
      if (i==0)
      { posX_init = dg::blas2::dot( xvec, w2d, ntilde[j])/mass_[j];
        posY_init = dg::blas2::dot( yvec, w2d, ntilde[j])/mass_[j];
      }
      if (i>0)
      { posX = dg::blas2::dot( xvec, w2d, ntilde[j])/mass_[j]-posX_init;
        posY = dg::blas2::dot( yvec, w2d, ntilde[j])/mass_[j]-posY_init;
      }
      if (i==0)
      { velX_old = -posX/deltaT;
        velY_old = -posY/deltaT;
        posX_old = posX;
        posY_old = posY;
      }
      velX = (posX - posX_old)/deltaT;
      velY = (posY - posY_old)/deltaT;
      velCOM=sqrt(velX*velX+velY*velY);
      accX = (velX - velX_old)/deltaT;
      accY = (velY - velY_old)/deltaT;
      if (i>0)
      { posX_old = posX;
        posY_old = posY;
        velX_old = velX;
        velY_old = velY;
      }
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[0  + add_to_spatial_idx], start0d, count0d, &posX);
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[1  + add_to_spatial_idx], start0d, count0d, &posY);
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[2  + add_to_spatial_idx], start0d, count0d, &velX);
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[3  + add_to_spatial_idx], start0d, count0d, &velY);
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[4  + add_to_spatial_idx], start0d, count0d, &accX);
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[5  + add_to_spatial_idx], start0d, count0d, &accY);
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[11 + add_to_spatial_idx], start0d, count0d, &velCOM);
      // maximal amplitude
      if ( p.amp > 0)
        maxamp = *thrust::max_element( npe[j].begin(), npe[j].end());
      else
        maxamp = *thrust::min_element( npe[j].begin(), npe[j].end());
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[12 + add_to_spatial_idx], start0d, count0d, &maxamp);
      //get max position and value(x,y_max)
      dg::blas2::gemv(equi, npe[j], helper);
      position = thrust::distance( helper.begin(), thrust::max_element( helper.begin(), helper.end()) );
      posX_max = hx*(1./2. + (double)(position%Nx))-posX_init;
      posY_max = hy*(1./2. + (double)(position/Nx))-posY_init;   // Nx->Ny?
      posX_max_hs = hx*(1./2. + (double)(position%Nx));
      posY_max_hs = hy*(1./2. + (double)(position/Nx));   // Nx->Ny?
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[6 + add_to_spatial_idx], start0d, count0d, &posX_max);
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[7 + add_to_spatial_idx], start0d, count0d, &posY_max);
      velX_max = (posX_max - posX_max_old) /deltaT;
      velY_max = (posY_max - posY_max_old) /deltaT;
      if (i>0)
      { posX_max_old = posX_max;
        posY_max_old = posY_max;
      }
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[8 + add_to_spatial_idx], start0d, count0d, &velX_max);
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[9 + add_to_spatial_idx], start0d, count0d, &velY_max);
      //compactness
      if (i==0)
      { heavi.set_origin( posX_max_hs, posY_max_hs );
        heavy = dg::evaluate( heavi, g2d);
        normalize = dg::blas2::dot( heavy, w2d, ntilde[j]);
      }
      heavi.set_origin( posX_max_hs, posY_max_hs);
      heavy = dg::evaluate( heavi, g2d);
      compactness =  dg::blas2::dot( heavy, w2d, ntilde[j])/normalize;
      err_out = nc_put_vara_double(species_wgrp_id[j], species_wspatial_id[10 + add_to_spatial_idx], start0d, count0d, &compactness);
      //energy
      dg::blas1::transform(npe[j], lnn[j], dg::LN<double>());
    }
    //field
    arakawa.variation(field[0], helper);
    double energy[5] = {};
    energy[0] = dg::blas2::dot(npe[0], w2d, lnn[0]);
    energy[1] = p.a[1]*p.tau[1]*dg::blas2::dot(npe[1], w2d, lnn[1]);
    energy[2] = p.a[2]*p.tau[2]*dg::blas2::dot(npe[2], w2d, lnn[2]);
    energy[3] = 0.5*p.a[1]*p.mu[1]*dg::blas2::dot(npe[1], w2d, helper);
    energy[4] = 0.5*p.a[2]*p.mu[2]*dg::blas2::dot(npe[2], w2d, helper);

    for (unsigned j = 0; j < 5; j++)
    { err_out = nc_put_vara_double(err_time_wgrp_id, err_time_wval_id[j], start0d, count0d, &energy[j]);
    }
    double dcn = cn[0]+cn[1]+cn[2];
    std::cout << cn[0] << " | " << cn[0] + cn[1] << " | " << cn[2] << " | " << dcn << "\n";

    err_out = nc_put_vara_double(err_time_wgrp_id, err_time_wval_id[5], start0d, count0d, &dcn);
  }
  err_out = nc_close(ncid_out);
  err_in = nc_close(ncid_in);
  return 0;
}
