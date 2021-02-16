#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>


#include "dg/file/file.h"

#include "reconnection.cuh"

int main( int argc, char* argv[])
{
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Value js;
    if( argc != 3)
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] [outputfile]\n";
        return -1;
    }
    else 
        dg::file::file2Json( argv[1], js, dg::file::comments::are_forbidden);
    const asela::Parameters p( js);
    p.display( std::cout);
    std::string input = js.toStyledString();
    ////////////////////////////////set up computations///////////////////////////
    //Make grids
    dg::Grid2d grid( -p.lxhalf, p.lxhalf, -p.lyhalf, p.lyhalf , p.n, p.Nx, p.Ny, dg::DIR, dg::PER);
    dg::Grid2d grid_out( -p.lxhalf, p.lxhalf, -p.lyhalf, p.lyhalf , p.n_out, p.Nx_out, p.Ny_out, dg::DIR, dg::PER);

     
    //create RHS 
    std::cout << "Constructing Asela...\n";
    asela::Asela<dg::CartesianGrid2d, dg::IDMatrix, dg::DMatrix, dg::DVec> asela( grid, p); //initialize before rolkar!
    std::cout << "Constructing Implicit...\n";
    asela::Implicit< dg::CartesianGrid2d, dg::IDMatrix, dg::DMatrix, dg::DVec > rolkar( grid, p);
    std::cout << "Done!\n";

 /////////////////////The initial field///////////////////////////////////////////
    std::cout << "intiialize fields" << std::endl;
    std::vector<dg::DVec> y0(4, dg::evaluate( dg::one, grid)), y1(y0); 
    
    //Harris sheet problem
    if( p.init == 0) { 
        //true harris is -lambda ln (cosh(x/lambda))
        dg::InvCoshXsq init0( 1., 2.*M_PI/p.lxhalf);
        dg::CosY perty(   1., 0., p.mY*M_PI/p.lyhalf);
        dg::CosXCosY damp(1., 0., M_PI/p.lxhalf/2.,0.);    
        y0[3] = dg::evaluate( init0, grid);
        y1[2] = dg::evaluate( perty, grid);
        y1[3] = dg::evaluate( damp, grid);
    }
    //Island coalescence problem
    if( p.init == 1) { 
        dg::IslandXY init0( p.lxhalf/(2.*M_PI), 0.2);
        dg::CosY perty(   1., 0., p.mY*M_PI/p.lyhalf);
        dg::CosXCosY damp(1., 0., M_PI/p.lxhalf/2.,0.);    
        y0[3] = dg::evaluate( init0, grid);
        y1[2] = dg::evaluate( perty, grid);
        y1[3] = dg::evaluate( damp, grid);
    }
    
    //Compute initial A_par
    dg::blas1::axpby(-p.amp1,y1[2],p.amp0,y0[3],y0[3]); // = [ A*Cos(y*ky) + 1/Cosh2(x*kx) ] (harris)
    dg::blas1::pointwiseDot(y1[3],y0[3],y0[3]);     // A_par = cos(x *kx') * [ A*Cos(y*ky) + 1/Cosh2(x*kx) ] (harris)

    //Compute u_e, U_i, w_e, W_i
    dg::blas2::gemv( rolkar.laplacianM(),y0[3], y0[2]);        //u_e = -nabla_perp^2 A_par
    dg::blas1::scal(y0[2],-1.0);                               //u_e =  nabla_perp^2  A_par
    dg::blas1::axpby(1., y0[2], p.beta/p.mu[0], y0[3], y0[2]); //w_e =  u_e + beta/mue A_par
    asela.initializene( y0[3], y1[3]);                         //A_G = Gamma_1 A_par
    //with mass correction (only consistent for cold ions)
    if (p.tau[1] == 0.) {
        dg::blas1::axpby(p.beta/p.mu[1], y1[3], 0.0, y0[3]);       //w_i =  beta/mui A_G
    }
    else {
          dg::blas1::scal(y0[3],0.0);                               //w_i = 0
    }
    
 
    
    //Compute n_e
    dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-1.)); // =Ni - bg 
    asela.initializene( y0[1], y0[0]);                         //n_e = Gamma_1 N_i
    std::cout << "Done!\n";
    
    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    karniadakis.init( asela, rolkar, 0., y0, p.dt);
    /////////////////////////////set up netcdf/////////////////////////////////////
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_create( argv[2],NC_NETCDF4|NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    int dim_ids[3], tvarID;
    err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid_out);
    
    //field IDs
    std::string names[8] = {"electrons", "ions", "Ue", "Ui", "potential","Aparallel","Vor","Jparallel"}; 
    int dataIDs[8]; 
    for( unsigned i=0; i<8; i++){
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}
    //energy IDs
    int EtimeID, EtimevarID;
    err = dg::file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, energyIDs[6], dissID, alignedID, dEdtID, accuracyID;
    err = nc_def_var( ncid, "energy",   NC_DOUBLE, 1, &EtimeID, &energyID);
    err = nc_def_var( ncid, "mass",   NC_DOUBLE, 1, &EtimeID, &massID);
    std::string energies[6] = {"Se", "Si", "Uperp", "Upare", "Upari","Uapar"}; 
    for( unsigned i=0; i<6; i++){
        err = nc_def_var( ncid, energies[i].data(), NC_DOUBLE, 1, &EtimeID, &energyIDs[i]);}
    err = nc_def_var( ncid, "dissipation",   NC_DOUBLE, 1, &EtimeID, &dissID);
    err = nc_def_var( ncid, "alignment",   NC_DOUBLE, 1, &EtimeID, &alignedID);
    err = nc_def_var( ncid, "dEdt",     NC_DOUBLE, 1, &EtimeID, &dEdtID);
    err = nc_def_var( ncid, "accuracy", NC_DOUBLE, 1, &EtimeID, &accuracyID);
    err = nc_enddef(ncid);

    ///////////////////////////////////first output/////////////////////////
    std::cout << "First output ... \n";
    size_t start[3] = {0, 0, 0};
    size_t count[3] = {1,  grid_out.n()*grid_out.Ny(), grid_out.n()*grid_out.Nx()};
    dg::DVec transfer(  dg::evaluate(dg::zero, grid));
    dg::DVec transferD( dg::evaluate(dg::zero, grid_out));
    dg::DVec helperD( dg::evaluate(dg::zero, grid_out));
    dg::HVec transferH( dg::evaluate(dg::zero, grid_out));
    dg::IDMatrix interpolate = dg::create::interpolation( grid_out, grid); 
    for( unsigned i=0; i<2; i++)
    {
        dg::blas2::symv( interpolate, y0[i], transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[i], start, count, transferH.data() );
    }
    transfer = asela.uparallel()[0];
    dg::blas2::symv( interpolate, transfer, transferD);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[2], start, count, transferH.data() );
    transfer = asela.uparallel()[1];
    dg::blas2::symv( interpolate, transfer, transferD);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[3], start, count, transferH.data() );
    transfer = asela.potential()[0];
    dg::blas2::symv( interpolate, transfer, transferD);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[4], start, count, transferH.data() );
    transfer = asela.aparallel();
    dg::blas2::symv( interpolate, transfer, transferD);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[5], start, count, transferH.data() );
    transfer = asela.potential()[0];
    dg::blas2::symv( interpolate, transfer, helperD);
    dg::blas2::symv( rolkar.laplacianM(), helperD, transferD);
    dg::blas1::scal(transferD,-1.0);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[6], start, count, transferH.data() );
    transfer = asela.aparallel();
    dg::blas2::symv( interpolate, transfer, helperD);
    dg::blas2::symv( rolkar.laplacianM(), helperD, transferD);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[7], start, count, transferH.data() );
    double time = 0;
    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    err = nc_put_vara_double( ncid, EtimevarID, start, count, &time);

    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    double energy0 = asela.energy(), mass0 = asela.mass(), E0 = energy0, mass = mass0, E1 = 0.0, dEdt = 0., diss = 0., aligned=0, accuracy=0.;
    std::vector<double> evec = asela.energy_vector();
    err = nc_put_vara_double( ncid, energyID, Estart, Ecount, &energy0);
    err = nc_put_vara_double( ncid, massID,   Estart, Ecount, &mass0);
    for( unsigned i=0; i<6; i++)
        err = nc_put_vara_double( ncid, energyIDs[i], Estart, Ecount, &evec[i]);

    err = nc_put_vara_double( ncid, dissID,     Estart, Ecount,&diss);
    err = nc_put_vara_double( ncid, alignedID,  Estart, Ecount,&aligned);
    err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount,&dEdt);
    err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount,&accuracy);
    err = nc_close(ncid);
    std::cout << "First write successful!\n";
    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Timer t;
    t.tic();
#ifdef DG_BENCHMARK
    unsigned step = 0;
#endif //DG_BENCHMARK
    for( unsigned i=1; i<=p.maxout; i++)
    {

#ifdef DG_BENCHMARK
        dg::Timer ti;
        ti.tic();
#endif//DG_BENCHMARK
        for( unsigned j=0; j<p.itstp; j++)
        {
            try{ karniadakis.step( asela, rolkar, time, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                err = nc_close(ncid);
                return -1;
            }
            step++;
            Estart[0] = step;
            E1 = asela.energy(), mass = asela.mass(), diss = asela.energy_diffusion();
            dEdt = (E1 - E0)/p.dt; 
            E0 = E1;
            accuracy = 2.*fabs( (dEdt-diss)/(dEdt + diss));
            evec = asela.energy_vector();
            err = nc_open(argv[2], NC_WRITE, &ncid);
            err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
            err = nc_put_vara_double( ncid, energyID, Estart, Ecount, &E1);
            err = nc_put_vara_double( ncid, massID,   Estart, Ecount, &mass);
            for( unsigned i=0; i<6; i++)
                err = nc_put_vara_double( ncid, energyIDs[i], Estart, Ecount, &evec[i]);
            err = nc_put_vara_double( ncid, dissID,     Estart, Ecount,&diss);
            err = nc_put_vara_double( ncid, alignedID,  Estart, Ecount,&aligned);
            err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount,&dEdt);
            err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount,&accuracy);

            std::cout << "(m_tot-m_0)/m_0: "<< (asela.mass()-mass0)/mass0<<"\t";
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout <<" d E/dt = " << dEdt <<" Lambda = " << diss << " -> Accuracy: "<< accuracy << "\n";
            err = nc_close(ncid);

        }
#ifdef DG_BENCHMARK
        ti.toc();
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s";
        ti.tic();
#endif//DG_BENCHMARK
        //////////////////////////write fields////////////////////////
        start[0] = i;
        err = nc_open(argv[2], NC_WRITE, &ncid);
        for( unsigned j=0; j<2; j++)
        {
            dg::blas2::symv( interpolate, y0[j], transferD);
            dg::blas1::transfer( transferD, transferH);
            err = nc_put_vara_double( ncid, dataIDs[j], start, count, transferH.data());
        }
        transfer = asela.uparallel()[0];
        dg::blas2::symv( interpolate, transfer, transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[2], start, count, transferH.data() );
        transfer = asela.uparallel()[1];
        dg::blas2::symv( interpolate, transfer, transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[3], start, count, transferH.data() );
        transfer = asela.potential()[0];
        dg::blas2::symv( interpolate, transfer, transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[4], start, count, transferH.data() );
        transfer = asela.aparallel();
        dg::blas2::symv( interpolate, transfer, transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[5], start, count, transferH.data() );
        transfer = asela.potential()[0];
        dg::blas2::symv( interpolate, transfer, helperD);
        dg::blas2::symv( rolkar.laplacianM(), helperD, transferD);
        dg::blas1::scal(transferD,-1.0);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[6], start, count, transferH.data() );
        transfer = asela.aparallel();
        dg::blas2::symv( interpolate, transfer, helperD);
        dg::blas2::symv( rolkar.laplacianM(), helperD, transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[7], start, count, transferH.data() );
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        err = nc_close(ncid);
#ifdef DG_BENCHMARK
        ti.toc();
        std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
    }
    t.toc(); 
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";

    return 0;

}
