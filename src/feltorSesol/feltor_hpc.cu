#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG
#include "dg/file/file.h"

#include "feltor.cuh"
#include "parameters.h"


int main( int argc, char* argv[])
{
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Value js;
    if( argc != 3 && argc != 4)
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n"; 
        std::cerr << "Usage: "<<argv[0]<<" [input.txt] [output.nc] [input.nc] \n";
        return -1;
    }
    else 
        dg::file::file2Json( argv[1], js, dg::file::comments::are_forbidden);
    std::string input = js.toStyledString(); 
    const eule::Parameters p( js);
    p.display( std::cout);

    //Make grid
    dg::Grid2d grid( 0., p.lx, 0., p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::Grid2d grid_out( 0., p.lx, 0., p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);  
    // Grid for radial probe location. This is used only in netcdf output, probe positioning is still hard-coded
    dg::Grid1d grid_probe(0, p.lx, 1, 8, p.bc_x);
    //create RHS 
    std::cout << "Constructing Explicit...\n";
    eule::Explicit<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > feltor( grid, p); //initialize before rolkar!
    std::cout << "Constructing Implicit...\n";
    eule::Implicit<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > rolkar( grid, p);
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //initial perturbation
    //dg::Gaussian3d init0(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
    dg::Gaussian init0(p.posX * p.lx, p.posY * p.ly, p.sigma, p.sigma, p.amp);
//     dg::BathRZ init0(16, 16, 1, 2.0, 2.0, 30.0, 5.0, p.amp);
//     solovev::ZonalFlow init0(p, gp);
//     dg::CONSTANT init0( 0.);
    
    //background profile
//     solovev::Nprofile prof(p, gp); //initial background profile
//     dg::CONSTANT prof(p.bgprofamp );
    //
//     dg::LinearX prof(-p.nprofileamp/((double)p.lx), p.bgprofamp + p.nprofileamp);
//     dg::SinProfX prof(p.nprofileamp, p.bgprofamp,M_PI/(2.*p.lx));
     dg::ExpProfX prof(p.nprofileamp, p.bgprofamp,p.invkappa);
//     dg::TanhProfX prof(p.lx*p.solb,p.invkappa,-1.0,p.bgprofamp,p.nprofileamp); //<n>
//     dg::TanhProfX prof(p.lx*p.solb,p.lx/10.,-1.0,p.bgprofamp,p.nprofileamp); //<n>

//     const dg::DVec prof =  dg::LinearX( -p.nprofileamp/((double)p.lx), p.bgprofamp + p.nprofileamp);

    std::vector<dg::DVec> y0(2, dg::evaluate(prof, grid)), y1(y0); 
    dg::HVec temp(dg::evaluate(dg::zero,grid));
    double time = 0;
    
    if (argc ==3){
      y1[1] = dg::evaluate( init0, grid);
      dg::blas1::pointwiseDot(y1[1], y0[1], y1[1]);

      dg::blas1::axpby(1., y1[1], 1., y0[1]); //initialize ni
      dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); //initialize ni-1
  //     dg::blas1::pointwiseDot(rolkar.damping(),y0[1], y0[1]); //damp with gaussprofdamp
      std::cout << "intiialize ne" << std::endl;
      feltor.initializene(y0[1], y0[0]);    
      std::cout << "Done!\n";
    }
    if (argc==4) {
        dg::file::NC_Error_Handle errIN;
        int ncidIN;
        errIN = nc_open( argv[3], NC_NOWRITE, &ncidIN);
        ///////////////////read in and show inputfile und geomfile//////////////////
        size_t length;
        errIN = nc_inq_attlen( ncidIN, NC_GLOBAL, "inputfile", &length);
        std::string inputIN(length, 'x');
        errIN = nc_get_att_text( ncidIN, NC_GLOBAL, "inputfile", &inputIN[0]);
        Json::Value jsIN;
        dg::file::string2Json(inputIN, jsIN, dg::file::comments::are_forbidden);

        const eule::Parameters pIN(  jsIN);
        std::cout << "[input.nc] file parameters" << std::endl;
        pIN.display( std::cout);    

        dg::Grid2d grid_IN( 0., pIN.lx, 0., pIN.ly, pIN.n_out, pIN.Nx_out, pIN.Ny_out, pIN.bc_x, pIN.bc_y);  
        dg::HVec transferINH( dg::evaluate(dg::zero, grid_IN));
        size_t count2dIN[3]  = {1, grid_IN.n()*grid_IN.Ny(), grid_IN.n()*grid_IN.Nx()};
        size_t start2dIN[3]  = {0, 0, 0};
        std::string namesIN[2] = {"electrons", "ions"}; 

        int dataIDsIN[2];     
        int timeIDIN;
        double  timeIN;
        size_t stepsIN;
        /////////////////////The initial field///////////////////////////////////////////
        /////////////////////Get time length and initial data///////////////////////////
        errIN = nc_inq_varid(ncidIN, namesIN[0].data(), &dataIDsIN[0]);
        errIN = nc_inq_dimlen(ncidIN, dataIDsIN[0], &stepsIN);
        stepsIN-=1;
        start2dIN[0] = stepsIN/pIN.itstp;
        std::cout << "stepsIN= "<< stepsIN <<  std::endl;
        std::cout << "start2dIN[0]= "<< start2dIN[0] <<  std::endl;
        errIN = nc_inq_varid(ncidIN, "time", &timeIDIN);
        errIN = nc_get_vara_double( ncidIN, timeIDIN,start2dIN, count2dIN, &timeIN);
        std::cout << "timeIN= "<< timeIN <<  std::endl;
        time=timeIN;
        dg::IHMatrix interpolateIN = dg::create::interpolation( grid,grid_IN); 
        errIN = nc_get_vara_double( ncidIN, dataIDsIN[0], start2dIN, count2dIN, transferINH.data());
        dg::blas2::gemv( interpolateIN, transferINH,temp);
        dg::assign(temp,y0[0]);
        errIN = nc_inq_varid(ncidIN, namesIN[1].data(), &dataIDsIN[1]);
        errIN = nc_get_vara_double( ncidIN, dataIDsIN[1], start2dIN, count2dIN, transferINH.data());
        dg::blas2::gemv( interpolateIN, transferINH,temp);
        dg::assign(temp,y0[1]);      
        errIN = nc_close(ncidIN);

    }

    dg::DefaultSolver< std::vector<dg::DVec> > solver( rolkar, y0,
            y0[0].size(), p.eps_time);
    dg::ImExMultistep< std::vector<dg::DVec> > karniadakis( "ImEx-BDF-3-3", y0);
    karniadakis.init( std::tie(feltor, rolkar, solver), 0., y0, p.dt);
//     feltor.energies( y0);//now energies and potential are at time 0
    /////////////////////////////set up netcdf/////////////////////////////////////
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_create( argv[2], NC_NETCDF4|NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    int dim_ids_field[3], tvarID_field;
    err = dg::file::define_dimensions(ncid, dim_ids_field, &tvarID_field, grid_out);
    err = nc_enddef(ncid);
    err = nc_redef(ncid);

    //field IDs
    std::vector<std::string> varname_fields;
    varname_fields.push_back("electrons"); varname_fields.push_back("ions"); varname_fields.push_back("potential"); varname_fields.push_back("vor");
    int dataIDs[4]; 
    for(unsigned i = 0; i < varname_fields.size(); i++)
        err = nc_def_var(ncid, varname_fields[i].data(), NC_DOUBLE, 3, dim_ids_field, &dataIDs[i]);
    //energy IDs, used for small time-step diagnostic
    int EtimeID, EtimevarID;
    err = dg::file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, energyIDs[3], dissID, dEdtID, accuracyID, couplingID; 

    err = nc_def_var(ncid, "energy", NC_DOUBLE, 1, &EtimeID, &energyID);
    err = nc_def_var(ncid, "mass", NC_DOUBLE, 1, &EtimeID, &massID);
    err = nc_def_var(ncid, "dissipation", NC_DOUBLE, 1, &EtimeID, &dissID);
    err = nc_def_var(ncid, "dEdt", NC_DOUBLE, 1, &EtimeID, &dEdtID);
    err = nc_def_var(ncid, "accuracy", NC_DOUBLE, 1, &EtimeID, &accuracyID);
    err = nc_def_var(ncid, "Coupling", NC_DOUBLE, 1, &EtimeID, &couplingID);  

    std::string energies[3] = {"Se", "Si", "Uperp"}; 
    for(unsigned i = 0; i < 3; i++)
    {
        err = nc_def_var( ncid, energies[i].data(), NC_DOUBLE, 1, &EtimeID, &energyIDs[i]);
    }

    //// Probe IDs
    std::vector<std::string> varname_probes;
    varname_probes.push_back("probe_ne"); varname_probes.push_back("probe_phi"); varname_probes.push_back("probe_Gamma_x");
    // Create x-dimension for probe 

    int ID_probes[3];
    int dim_ids_probe[2];
    dim_ids_probe[0] = EtimeID;
    //dim_ids_probe[1] = 
    dg::file::define_dimension(ncid, &dim_ids_probe[1],  grid_probe, "X_probe" );
    for(unsigned i = 0; i < varname_probes.size(); i++)
    {
        err = nc_def_var(ncid, varname_probes[i].data(), NC_DOUBLE, 2, dim_ids_probe, &ID_probes[i]);
    } 
    
    err = nc_enddef(ncid);
    ///////////////////////////////////first output/////////////////////////
    std::cout << "First output ... \n";
    size_t start[3] = {0, 0, 0};
    size_t count[3] = {1, grid_out.n() * grid_out.Ny(), grid_out.n() * grid_out.Nx()};
    dg::DVec transfer( dg::evaluate(dg::zero, grid));
    dg::DVec transferD( dg::evaluate(dg::zero, grid_out));
    dg::HVec transferH( dg::evaluate(dg::zero, grid_out));
    dg::IDMatrix interpolate = dg::create::interpolation( grid_out, grid); 
    for( unsigned i=0; i<2; i++)
    {
        dg::blas2::gemv( interpolate, y0[i], transferD);
        dg::assign( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[i], start, count, transferH.data() );
    }
    //pot
    transfer = feltor.potential()[0];
    dg::blas2::symv( interpolate, transfer, transferD);
    dg::assign( transferD, transferH);

    err = nc_put_vara_double( ncid, dataIDs[2], start, count, transferH.data() );
    //Vor
    transfer = feltor.potential()[0];
    dg::blas2::gemv( rolkar.laplacianM(), transfer, y1[1]);            
    dg::blas2::symv( interpolate,y1[1], transferD);
    dg::assign( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[3], start, count, transferH.data() );



    err = nc_put_vara_double( ncid, tvarID_field, start, count, &time);
    err = nc_put_vara_double( ncid, EtimevarID, start, count, &time);

    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    double energy0 = feltor.energy(), mass0 = feltor.mass(), E0 = energy0, mass = mass0, E1 = 0.0, dEdt = 0., diss = 0., accuracy=0.;
    double radtrans = feltor.radial_transport();
    double coupling = feltor.coupling();
    std::vector<double> evec = feltor.energy_vector();
    err = nc_put_vara_double( ncid, energyID, Estart, Ecount, &energy0);
    err = nc_put_vara_double( ncid, massID, Estart, Ecount, &mass0);
    for( unsigned i=0; i<3; i++)
        err = nc_put_vara_double( ncid, energyIDs[i], Estart, Ecount, &evec[i]);

    err = nc_put_vara_double( ncid, dissID, Estart, Ecount, &diss);
    err = nc_put_vara_double( ncid, dEdtID, Estart, Ecount, &dEdt);
    err = nc_put_vara_double( ncid, couplingID, Estart, Ecount, &coupling);
    err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount, &accuracy);

    err = nc_close(ncid);
    std::cout << "First write successful!\n";

    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Timer t;
    t.tic();
    unsigned step = 0;
    for( unsigned i=1; i<=p.maxout; i++)
    {

#ifdef DG_BENCHMARK
        dg::Timer ti;
        ti.tic();
#endif//DG_BENCHMARK
        for( unsigned j=0; j<p.itstp; j++)
        {
            try{ karniadakis.step( std::tie(feltor, rolkar, solver), time, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                err = nc_close(ncid);
                return -1;
            }
            step++;
            Estart[0] = step;
            E1 = feltor.energy(), mass = feltor.mass(), diss = feltor.energy_diffusion();
            dEdt = (E1 - E0)/p.dt; 
            E0 = E1;
            accuracy = 2.*fabs( (dEdt-diss)/(dEdt + diss));
            evec = feltor.energy_vector();
            radtrans = feltor.radial_transport();
            coupling= feltor.coupling();
            err = nc_open(argv[2], NC_WRITE, &ncid);
            err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
            err = nc_put_vara_double( ncid, energyID, Estart, Ecount, &E1);
            err = nc_put_vara_double( ncid, massID,   Estart, Ecount, &mass);
            for( unsigned i=0; i<3; i++)
            {
                err = nc_put_vara_double( ncid, energyIDs[i], Estart, Ecount, &evec[i]);
            }
            err = nc_put_vara_double( ncid, dissID,     Estart, Ecount,&diss);
            err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount,&dEdt);
            err = nc_put_vara_double( ncid, couplingID, Estart, Ecount,&coupling);    
            err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount,&accuracy);

            std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass0<<"\t";
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout <<" d E/dt = " << dEdt <<" Lambda = " << diss << " -> Accuracy: "<< accuracy << "\n";
            err = nc_close(ncid);
        }
#ifdef DG_BENCHMARK
        ti.toc();
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
        //////////////////////////write fields////////////////////////
        start[0] = i;
        err = nc_open(argv[2], NC_WRITE, &ncid);
        for(unsigned j = 0; j < 2; j++)
        {
            dg::blas2::symv( interpolate, y0[j], transferD);
            dg::assign( transferD, transferH);
            err = nc_put_vara_double( ncid, dataIDs[j], start, count, transferH.data());
        }
        transfer = feltor.potential()[0];
        dg::blas2::symv( interpolate, transfer, transferD);
        dg::assign( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[2], start, count, transferH.data() );
        transfer = feltor.potential()[0];
        dg::blas2::gemv( rolkar.laplacianM(), transfer, y1[1]);            
        dg::blas2::symv( interpolate,y1[1], transferD);
        dg::assign( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[3], start, count, transferH.data());

        err = nc_put_vara_double(ncid, tvarID_field, start, count, &time);
        err = nc_close(ncid);
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

