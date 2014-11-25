#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

#include "dg/algorithm.h"

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "file/read_input.h"
#include "file/nc_utilities.h"

#include "solovev/geometry.h"
#include "solovev/init.h"



int main( int argc, char* argv[])
{
    if( argc != 4)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [outputfsa.nc] [output2d.nc]\n";
        return -1;
    }
//     std::ofstream os( argv[2]);
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

    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //old boxscale
//     double Rmin=gp.R_0-p.boxscaleRp*gp.a;
//     double Zmin=-p.boxscaleRp*gp.a*gp.elongation;
//     double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
//     double Zmax=p.boxscaleRp*gp.a*gp.elongation;

    //3d grid
    dg::Grid3d<double > grid_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n_out, p.Nx_out, p.Ny_out, 1., dg::DIR, dg::DIR, dg::PER, dg::cylindrical);  
    //2d grid
    dg::Grid2d<double>  grid2d_out( Rmin,Rmax, Zmin,Zmax,p.n_out, p.Nx_out, p.Ny_out,dg::PER,dg::PER);
    //1d grid
    solovev::Psip psip(gp);
    dg::HVec psipog2d   = dg::evaluate( psip, grid2d_out);    
    double psipmin = (float)thrust::reduce( psipog2d.begin(), psipog2d.end(), 0.0,thrust::minimum<double>()  );
    
    unsigned Npsi = 50;//set number of psivalues
    std::cout << "psipmin =" << psipmin << " Npsi =" << Npsi  <<std::endl;
    dg::Grid1d<double>  grid1d_out(psipmin ,0.0, 1, Npsi,dg::DIR); //one dimensional sipgrid
//     
    //read in midplane of electrons, ions Ue, Ui, and potential, and energy
    std::string names[5] = {"electrons", "ions", "Ue", "Ui", "potential"}; 
    int dataIDs[5];
    std::string names2d[11] = {"Ne_avg", "Ni_avg", "Ue_avg", "Ui_avg", "phi_avg","dNe_mp", "dNi_mp", "dUe_mp", "dUi_mp", "dphi_mp","vor_avg"}; 
    int dataIDs2d[11];
     //generate 2d nc file for one time step
    file::NC_Error_Handle err2d; 
    int ncid2d; 
    err2d = nc_create(argv[3],NC_NETCDF4|NC_CLOBBER, &ncid2d);
    err2d = nc_put_att_text( ncid2d, NC_GLOBAL, "inputfile", input.size(), input.data());
    err2d = nc_put_att_text( ncid2d, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids[3], tvarID;
    err2d = file::define_dimensions( ncid2d, dim_ids, &tvarID, grid2d_out);
    for( unsigned i=0; i<11; i++){
        err2d = nc_def_var( ncid2d, names2d[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs2d[i]);
    }   
    //midplane 2d fields
    size_t count2d[3] = {1, grid_out.n()*grid_out.Ny(), grid_out.n()*grid_out.Nx()};
    size_t start2d[3] = {0, 0, 0};
    size_t count[4] = {1, 1, grid_out.n()*grid_out.Ny(), grid_out.n()*grid_out.Nx()};
    size_t start[4] = {0, 0, 0, 0};
    dg::HVec data2d = dg::evaluate( dg::one, grid2d_out);
    dg::HVec vorticity = dg::evaluate( dg::one, grid2d_out);
    err2d = nc_close(ncid2d);
    err = nc_close(ncid); 
    
     //generate 1d nc file for one time step for the f(psi) quantities
    std::string names1d[7] = {"Ne_fsa", "Ni_fsa", "Ue_Fsa", "Ui_fsa", "phi_fsa","q","vor_fsa"}; 
    int dataIDs1d[7];
    file::NC_Error_Handle err1d; 
    int ncid1d; 
    err1d = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid1d);
    err1d = nc_put_att_text( ncid1d, NC_GLOBAL, "inputfile", input.size(), input.data());
    err1d = nc_put_att_text( ncid1d, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids1d[2], tvarID1d;
    err1d = file::define_dimensions( ncid1d, dim_ids1d, &tvarID1d, grid1d_out);
    for( unsigned i=0; i<7; i++){
        err1d = nc_def_var( ncid1d, names1d[i].data(), NC_DOUBLE, 2, dim_ids1d, &dataIDs1d[i]);
    }   
//     midplane 2d fields
    size_t count1d[2] = {1, grid1d_out.n()*grid1d_out.N()};
    size_t start1d[2] = {0, 0};

    err1d = nc_close(ncid1d);
    double time=0.;

//     double energy_0 =0.,U_i_0=0.,U_e_0=0.,U_phi_0=0.,U_pare_0=0.,U_pari_0=0.,mass_0=0.;
//     os << "#Time(1) mass(2) Ue(3) Ui(4) Uphi(5) Upare(6) Upari(7) Utot(8) EDiff(9)\n";
        std::cout << "Compute safety factor   "<< "\n";
        solovev::Alpha alpha(gp); 
        dg::HVec alphaog2d   = dg::evaluate( alpha, grid2d_out);      
        dg::HVec abs = dg::evaluate( dg::coo1, grid1d_out);
        solovev::SafetyFactor<dg::HVec> qprofile(grid2d_out, gp, alphaog2d );
        dg::HVec sf = dg::evaluate(qprofile, grid1d_out);
    //perp laplacian
    dg::Elliptic<dg::HMatrix, dg::HVec, dg::HVec> laplacian(grid2d_out,dg::DIR, dg::DIR, dg::normed, dg::centered);     
//     dg::Elliptic<dg::HMatrix, dg::HVec, dg::HVec> laplacian(grid2d_out,dg::DIR, dg::DIR, dg::normed, dg::forward);
        
    for( unsigned i=0; i<p.maxout; i++)//timestepping
    {
        start[0] = i; //set specific time  
        start2d[0] = i;
        start1d[0] = i;
        time += p.itstp*p.dt;
        err2d = nc_open(argv[3], NC_WRITE, &ncid2d);
        err1d = nc_open(argv[2], NC_WRITE, &ncid1d);

        std::cout << "Timestep = " << i << "at time = " << time << "\n";
        std::cout << "Extract 2d planes for avg 2d field and phi_fluc at midplane and computing fsa of Phi_Avg quantities"<< "\n";
     

        //Compute toroidal average and fluctuation at midplane for every timestep
        dg::HVec data2davg = dg::evaluate( dg::one, grid2d_out);   
        dg::HVec data2dflucmid = dg::evaluate( dg::one, grid2d_out);    
        for( unsigned i=0; i<5; i++)
        {
            dg::blas1::axpby(0.0,data2d,   0.0,data2d); //data2d=0;
            dg::blas1::axpby(0.0,data2davg,0.0,data2davg);  //data2davg=0;
            dg::blas1::axpby(0.0,data2dflucmid ,0.0,data2dflucmid );  //data2davg=0;
            //sum up planes
            for( unsigned k=0; k<grid_out.Nz(); k++)
            {
                err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
                start[1] = k;  count[1] = 1.;//get specific plane
                err = nc_inq_varid(ncid, names[i].data(), &dataIDs[i]);
                err = nc_get_vara_double( ncid, dataIDs[i], start, count, data2d.data());
                err = nc_close(ncid);  //close 3d file
                //Sum up avg
                if (k==grid_out.Nz()/2) dg::blas1::axpby(1.0,data2d ,1.0,data2dflucmid );
                dg::blas1::axpby(1.0,data2d,1.0,data2davg); 
            }

            //Scale avg
            dg::blas1::scal(data2davg,1./grid_out.Nz());
            dg::blas1::axpby(1.0,data2dflucmid,-1.0,data2davg,data2dflucmid); 
  
            //write avg  and fluc of midplane into 2d netcdf file
            err2d = nc_put_vara_double( ncid2d, dataIDs2d[i],   start2d, count2d, data2davg.data());
            err2d = nc_put_vara_double( ncid2d, dataIDs2d[i+5], start2d, count2d, data2dflucmid.data());
            if (i==4) {
                dg::blas2::gemv( laplacian,data2davg,vorticity);
                err2d = nc_put_vara_double( ncid2d, dataIDs2d[10],   start2d, count2d, vorticity.data());
                solovev::FluxSurfaceAverage<dg::HVec> fsavor(grid2d_out,gp, vorticity );
                dg::HVec fsaofvoravg = dg::evaluate(fsavor,grid1d_out);
                err1d = nc_put_vara_double( ncid1d, dataIDs1d[6], start1d, count1d,  fsaofvoravg.data());
            }
            solovev::FluxSurfaceAverage<dg::HVec> fsa(grid2d_out,gp, data2davg );
            dg::HVec fsaofdata2davg = dg::evaluate(fsa,grid1d_out);
            err1d = nc_put_vara_double( ncid1d, dataIDs1d[i], start1d, count1d,  fsaofdata2davg.data());

        }
        err1d = nc_put_vara_double( ncid1d, dataIDs1d[5], start1d, count1d,  sf.data());
        err1d = nc_put_vara_double( ncid1d, tvarID1d, start1d, count1d, &time);
        err1d = nc_close(ncid1d);
        err2d = nc_put_vara_double( ncid2d, tvarID, start2d, count2d, &time);
        err2d = nc_close(ncid2d); //close 2d netcdf files
      
        // ---- Compute energies ----
//         std::cout << "Compute macroscopic timedependent quantities"<< "\n";

    

        //write macroscopic timedependent quantities into output.dat file
//         os << time << " " << mass_norm << " " <<  U_e_norm <<" " <<  U_i_norm <<" " << U_phi_norm <<" " << U_pare_norm <<" " << U_pari_norm <<" "  << energy_norm <<" " << energy_diff<<std::endl;
        
        
    } //end timestepping
    //cross correleation between phi and ne
    
    //relative fluctuation amplitude(R,Z,phi) = delta n(R,Z,phi)/n0(psi)
   
    
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

