#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

#include "dg/backend/xspacelib.cuh"
#include "dg/arakawa.h"
#include "dg/functors.h"
#include "dg/elliptic.h"
#include "solovev/geometry.h"

#include "solovev/geom_parameters.h"
#include "feltor/parameters.h"
#include "file/read_input.h"
#include "file/nc_utilities.h"


//read and evaluate FELTOR nc files

double X( double x, double y) {return x;}
double Y( double x, double y) {return y;}


int main( int argc, char* argv[])
{
    if( argc != 4)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [output.dat] [output2d.nc]\n";
        return -1;
    }
    std::ofstream os( argv[2]);
    std::cout << argv[1]<< " -> "<<argv[2]<<" & "<<argv[3]<<std::endl;

    //////////////////////////////open nc file//////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    ///////////////////read in and show inputfile und geomfile//////////////////
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input( length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
    err = nc_inq_attlen( ncid, NC_GLOBAL, "geomfile", &length);
    std::string geom( length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "geomfile", &geom[0]);
    std::cout << "input "<<input<<std::endl;
    std::cout << "geome "<<geom <<std::endl;
    const eule::Parameters p(file::read_input( input));
    const solovev::GeomParameters gp(file::read_input( geom));
    p.display();
    gp.display();
    ///////////////////////////////////////////////////////////////////////////
    double Rmin=gp.R_0-(p.boxscale)*gp.a;
    double Zmin=-(p.boxscale)*gp.a*gp.elongation;
    double Rmax=gp.R_0+(p.boxscale)*gp.a; 
    double Zmax=(p.boxscale)*gp.a*gp.elongation;
    dg::Grid3d<double > grid_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n_out, p.Nx_out, p.Ny_out, p.Nz_out, dg::DIR, dg::DIR, dg::PER, dg::cylindrical);  
    dg::Grid2d<double> grid2d_out( grid_out.x0(), grid_out.x1(), grid_out.y0(), grid_out.y1(), grid_out.n(), grid_out.Nx(), grid_out.Ny());

    //read in midplane of electrons, ions Ue, Ui, and potential, and energy
    std::string names[5] = {"electrons", "ions", "Ue", "Ui", "potential"}; 
    int dataIDs[5];
    std::string names2d[10] = {"electrons_mp", "ions_mp", "Ue_mp", "Ui_mp", "potential_mp","electrons_avg", "ions_avg", "Ue_avg", "Ui_avg", "potential_avg"}; 
    int dataIDs2d[10];
     //generate 2d nc file for one time step
    file::NC_Error_Handle err2d; 
    int ncid2d; 
    err2d = nc_create(argv[3],NC_NETCDF4|NC_CLOBBER, &ncid2d);
    err2d = nc_put_att_text( ncid2d, NC_GLOBAL, "inputfile", input.size(), input.data());
    err2d = nc_put_att_text( ncid2d, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids[3], tvarID;
    err2d = file::define_dimensions( ncid2d, dim_ids, &tvarID, grid2d_out);
    for( unsigned i=0; i<10; i++){
        err2d = nc_def_var( ncid2d, names2d[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs2d[i]);
    }   
    //midplane 2d fields
    size_t count2d[3] = {1., grid_out.n()*grid_out.Ny(), grid_out.n()*grid_out.Nx()};
    size_t start2d[3] = {0, 0, 0};
    size_t count[4] = {1., 1., grid_out.n()*grid_out.Ny(), grid_out.n()*grid_out.Nx()};
    size_t start[4] = {0, 0, 0, 0};
    dg::HVec data2d = dg::evaluate( dg::one, grid2d_out);
    err2d = nc_close(ncid2d);
    err = nc_close(ncid); 
    double time=0.;
    dg::HVec one3d    = dg::evaluate( dg::one,  grid_out);
    dg::HVec omega    = dg::evaluate( dg::one,  grid_out);
    dg::HVec chi    = dg::evaluate( dg::one,  grid_out);
    dg::HVec lambda    = dg::evaluate( dg::one,  grid_out);
    dg::HVec w3d    = dg::create::weights(  grid_out);
    dg::HVec binv    = dg::evaluate(solovev::Field(gp) , grid_out);
    dg::ArakawaX<dg::HMatrix, dg::HVec >           arakawa (grid_out); 
    dg::Elliptic<dg::HMatrix, dg::HVec, dg::HVec> lapperp(grid_out, dg::not_normed, dg::symmetric);

    double energy_0 =0.,U_i_0=0.,U_e_0=0.,U_phi_0=0.,U_pare_0=0.,U_pari_0=0.,mass_0=0.;
    os << "#Time(1) mass(2) Ue(3) Ui(4) Uphi(5) Upare(6) Upari(7) Utot(8) EDiff(9)\n";
    
    for( unsigned i=0; i<p.maxout; i++)//timestepping
    {
        start[0] = i; //set specific time  
        start2d[0] = i;
        std::cout << "Timestep = " << i << "\n";
        time += p.itstp*p.dt;
        err2d = nc_open(argv[3], NC_WRITE, &ncid2d);
        //3d macroscopic fields
        dg::HVec ne    = dg::evaluate( dg::one,  grid_out);
        dg::HVec ni    = dg::evaluate( dg::one,  grid_out);
        dg::HVec logni = dg::evaluate( dg::one,  grid_out);
        dg::HVec logne = dg::evaluate( dg::one,  grid_out);
        dg::HVec ue    = dg::evaluate( dg::one,  grid_out);
        dg::HVec ui    = dg::evaluate( dg::one,  grid_out);
        dg::HVec phi    = dg::evaluate( dg::one,  grid_out);
        
        std::cout << "Extract midplane 2d field"<< "\n";
        for( unsigned i=0; i<5; i++)
        {
            err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
                start[1] = grid_out.Nz()/2; count[1] =1.;//set midplane
                err = nc_inq_varid(ncid, names[i].data(), &dataIDs[i]);
                err = nc_get_vara_double( ncid, dataIDs[i], start, count, data2d.data());
            err = nc_close(ncid);  //close 3d file
            err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
                start[1] = 0; count[1] = grid_out.Nz(); //3d fields
                err = nc_inq_varid(ncid, names[i].data(), &dataIDs[i]);
                if (i==0) {
                    err = nc_get_vara_double( ncid, dataIDs[0], start, count, ne.data()); 
                    dg::blas1::transform( ne, ne, dg::PLUS<>(+1)); 
                    dg::blas1::transform( ne, logne, dg::LN<double>());                
                }
                if (i==1) {
                    err = nc_get_vara_double( ncid, dataIDs[i], start, count, ni.data()); 
                    dg::blas1::transform( ni, ni, dg::PLUS<>(+1)); 
                    dg::blas1::transform( ni, logni, dg::LN<double>());                
                }
                if (i==2) {
                    err = nc_get_vara_double( ncid, dataIDs[i], start, count, ue.data()); 
                }
                if (i==3) {
                    err = nc_get_vara_double( ncid, dataIDs[i], start, count, ui.data()); 
                }
                if (i==4) {
                    err = nc_get_vara_double( ncid, dataIDs[i], start, count, phi.data()); 
                }

            err = nc_close(ncid); //close 3d file
            //write midplane into 2d netcdf file
            err2d = nc_put_vara_double( ncid2d, dataIDs2d[i], start2d, count2d, data2d.data());
        }

        //Compute toroidal average
        dg::HVec data2davg = dg::evaluate( dg::one, grid2d_out);    
        std::cout << "Extract 2d planes for avg 2d field"<< "\n";
        for( unsigned i=0; i<5; i++)
        {
            dg::blas1::axpby(0.0,data2d,   0.0,data2d); //data2d=0;
            dg::blas1::axpby(0.0,data2davg,0.0,data2davg);  //data2davg=0;
            for( unsigned k=0; k<grid_out.Nz(); k++)
            {
                err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
                start[1] = k;  count[1] = 1.;//get specific plane
                err = nc_inq_varid(ncid, names[i].data(), &dataIDs[i]);
                err = nc_get_vara_double( ncid, dataIDs[i], start, count, data2d.data());
                err = nc_close(ncid);  //close 3d file
                //Sum up avg
                dg::blas1::axpby(1.0,data2d,1.0,data2davg); //data2davg+=data2d;;
            }

            //Scale avg
            dg::blas1::scal(data2davg,1./grid_out.Nz());
            //write avg into 2d netcdf file
            err2d = nc_put_vara_double( ncid2d, dataIDs2d[i+5], start2d, count2d, data2davg.data());

        }

        err2d = nc_put_vara_double( ncid2d, tvarID, start2d, count2d, &time);
        err2d = nc_close(ncid2d); //close 2d netcdf files
      
        // ---- Compute energies ----
        std::cout << "Compute macroscopic timedependent quantities"<< "\n";
        
        //write macroscopic timedependent quantities into output.dat file
//         os << time << " " << mass_norm << " " <<  U_e_norm <<" " <<  U_i_norm <<" " << U_phi_norm <<" " << U_pare_norm <<" " << U_pari_norm <<" "  << energy_norm <<" " << energy_diff<<std::endl;
        
        
    } //end timestepping
    //cross correleation between phi and ne
    //relative fluctuation amplitude(psi) = delta n/n0
    //Compute flux average
//     std::cout << "Compute flux average of psi   "<< "\n";
//     dg::DVec psipongrid   = dg::evaluate( psip, grid2d_out);
//     dg::DVec psipRongrid  = dg::evaluate( psipR, grid2d_out);
//     dg::DVec psipZongrid  = dg::evaluate( psipZ, grid2d_out);
//     dg::DVec oneongrid    = dg::evaluate( dg::one, grid2d_out);
//     
//     double psipRmax = (float)thrust::reduce( psipRongrid .begin(), psipRongrid .end(),  0.,     thrust::maximum<double>()  );    
//     double psipRmin = (float)thrust::reduce( psipRongrid .begin(), psipRongrid .end(),  psipRmax,thrust::minimum<double>()  );
//     double psipZmax = (float)thrust::reduce( psipZongrid .begin(), psipZongrid  .end(), 0.,     thrust::maximum<double>()  );    
//     double psipZmin = (float)thrust::reduce( psipZongrid .begin(), psipZongrid  .end(), psipZmax,thrust::minimum<double>()  );
//     double deltapsi = abs(psipZmin/Nx/n +psipRmin/Ny/n);
//     std::cout << "deltapsi = " << deltapsi << "\n";
//     deltaf.setepsilon(deltapsi/4. );
// 
//     for (unsigned i=0;i<10;i++)
//     {
//         deltaf.setpsi( (double)i/(-10.));
//         dg::DVec deltafongrid = dg::evaluate( deltaf, grid);
//         const dg::DVec w2d = dg::create::weights( grid);
//         double psipcut = dg::blas2::dot( psipongrid,w2d,deltafongrid); //int deltaf psip
//         double vol     = dg::blas2::dot( oneongrid , w2d,deltafongrid); //int deltaf
//         double psipflavg = psipcut/vol;
//         std::cout << "psi = " << (double)i/(-10.)<< " psipflavg  = "<< psipflavg << " diff = "<< psipflavg-(double)i/(-10.)<<"\n";
//     }
    
    //Compute energys
    
    
    //const unsigned num_out = t5file.get_size();

    //dg::Grid2d<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);

    //dg::HVec input_h( grid.size());
    //dg::HVec input0( input_h), input1(input0), ln0( input0), ln1(input0);
    //dg::HVec visual( input0);
    //std::vector<double> mass(p.maxout*p.itstp+1,0), energy(mass), diffusion(mass), dissipation(mass);
    //if( p.global)
    //{
    //    t5file.get_xfile( mass, "mass");
    //    t5file.get_xfile( energy, "energy");
    //    t5file.get_xfile( diffusion, "diffusion");
    //    t5file.get_xfile( dissipation, "dissipation");
    //}

    //dg::HVec xvec = dg::evaluate( X, grid);
    //dg::HVec yvec = dg::evaluate( Y, grid);
    //dg::HVec one = dg::evaluate( dg::one, grid);
    //dg::HVec w2d = dg::create::weights( grid);
    //dg::HMatrix equi = dg::create::backscatter( grid);

    //t5file.get_field( input0, "electrons", 1);
    //if( p.global)
    //    thrust::transform( input0.begin(), input0.end(), input0.begin(), dg::PLUS<double>(-1));
    //double mass_ = dg::blas2::dot( one, w2d, input0 ); 
    //const double posX_init = dg::blas2::dot( xvec, w2d, input0)/mass_;
    //const double posY_init = dg::blas2::dot( yvec, w2d, input0)/mass_;
    //double posX, posY, velX, velY, accX, accY;
    //double posX_max, posY_max;
    //double posX_old = 0, posY_old = 0;
    //double deltaT = p.dt*p.itstp;
    //t5file.get_field( input0, "electrons", 2);
    //    if( p.global)
    //        thrust::transform( input0.begin(), input0.end(), input0.begin(), dg::PLUS<double>(-1));
    //    mass_ = dg::blas2::dot( one, w2d, input0 ); 
    //    posX = dg::blas2::dot( xvec, w2d, input0)/mass_ - posX_init;
    //    posY = dg::blas2::dot( yvec, w2d, input0)/mass_ - posY_init;
    //    double velX_old = -posX/deltaT, velY_old = -posY/deltaT; 
    //    //velX_old = NAN, velY_old = NAN;

    //Vesqr<dg::HMatrix, dg::HVec> vesqr( grid, p.kappa);
    //os << "#Time(1) posX(2) posY(3) velX(4) velY(5) mass(6) diff(7) (m_tot-m_0)/m_0(8) "
    //   << "Ue(9) Ui(10) Uphi(11) Utot(12) (U_tot-U_0)/U_0(13) diss(14) posX_max(15) posY_max(16) max_amp(19)\n";
    ////dg::Timer t;
    //for( unsigned idx=1; idx<=num_out; idx++)
    //{
    //    //t.tic();
    //    //std::cout << idx<<std::endl;
    //    t5file.get_field( input0, "electrons", idx);
    //    //input0 = input_h;
    //    t5file.get_field( input1, "ions", idx);
    //    //input1 = input_h;
    //    //t.toc();
    //    //std::cout << "Reading took "<<t.diff()<<"s\n";
    //    //t.tic();
    //    double Ue = 0, Ui = 0, Uphi = 0;
    //    if( p.global)
    //    {
    //        //std::cout << "in global branch!\n";
    //        log( input0, ln0), log( input1, ln1);
    //        Ue = dg::blas2::dot( input0, w2d, ln0);
    //        Ui = p.tau*dg::blas2::dot( input1, w2d, ln1);
    //        Uphi = energy[(idx-1)*p.itstp] - Ue - Ui;
    //        thrust::transform( input0.begin(), input0.end(), input0.begin(), dg::PLUS<double>(-1));
    //    }
    //    mass_ = dg::blas2::dot( one, w2d, input0 ); 
    //    posX = dg::blas2::dot( xvec, w2d, input0)/mass_ - posX_init;
    //    posY = dg::blas2::dot( yvec, w2d, input0)/mass_ - posY_init;

    //    velX = (posX - posX_old)/deltaT;
    //    velY = (posY - posY_old)/deltaT;
    //    accX = (velX - velX_old)/deltaT;
    //    accY = (velY - velY_old)/deltaT;
    //    posX_old = posX; posY_old = posY;
    //    velX_old = velX; velY_old = velY;
    //    //output
    //    os << t5file.get_time( idx);//(1)
    //    os << " "<<posX << " " << posY << " "<<velX<<" "<<velY;//(2-5)
    //    os << " "<<mass[(idx-1)*p.itstp] << " "<<diffusion[(idx-1)*p.itstp];//(6,7)
    //    os << " "<< (mass[(idx-1)*p.itstp]-mass[0])/(mass[0]-grid.lx()*grid.ly());//blob mass is mass[] - Area (8)
    //    os << " "<<Ue<<" "<<Ui<<" "<<Uphi<<" "<<energy[(idx-1)*p.itstp]; //(9-12)
    //    os << " "<<(energy[(idx-1)*p.itstp]-energy[0])/energy[0];//(13)
    //    os << " "<<dissipation[(idx-1)*p.itstp]; //(14)
    //    //get the maximum amplitude position
    //    dg::blas2::gemv( equi, input0, visual);
    //    unsigned position = thrust::distance( visual.begin(), thrust::max_element( visual.begin(), visual.end()) );
    //    unsigned Nx = p.Nx*p.n; 
    //    const double hx = grid.hx()/(double)grid.n();
    //    const double hy = grid.hy()/(double)grid.n();
    //    posX_max = hx*(1./2. + (double)(position%Nx))-posX_init;
    //    posY_max = hy*(1./2. + (double)(position/Nx))-posY_init;
    //    os << " "<<posX_max<<" "<<posY_max;
    //    os << " "<<accX<<" "<<accY;
    //    os << " "<<*thrust::max_element( visual.begin(), visual.end());
    //    os <<"\n";
    //    //t.toc();
    //    //std::cout << "The rest took "<<t.diff()<<"s\n";
    //}
    //os.close();
    return 0;
}

