#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>

#ifdef WITH_GLFW
#include "draw/host_window.h"
#endif
#ifdef WITH_MPI
#include <mpi.h> //activate mpi
#endif

#include "dg/file/file.h"

#include "diag.h"
#include "feltor.h"
#include "parameters.h"


int main( int argc, char* argv[])
{
#ifdef WITH_MPI
    dg::mpi_init( argc, argv);
    MPI_Comm comm;
    dg::mpi_init2d( dg::DIR, dg::PER, comm, std::cin, true);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
    ////////////////////////Parameter initialisation//////////////////////////
    std::string input;
    if( argc == 1)
        input = "input.json";
    else
        input = argv[1];
    dg::file::WrappedJsonValue js( dg::file::error::is_throw);
    eule::Parameters p;
    try{
        js = dg::file::file2Json( input);
        p = { js};
    } catch( std::exception& e) {
        DG_RANK0 std::cerr << "ERROR in input file "<<input<<std::endl;
        DG_RANK0 std::cerr << e.what()<<std::endl;
        dg::abort_program();
    }
    DG_RANK0 std::cout << js.toStyledString() << std::endl;
    DG_RANK0 p.display(std::cout);
    //////////////////////////////////////////////////////////////////////////

    //Make grid
    dg::x::Grid2d grid(     0., p.lx, 0.,p.ly, p.n,     p.Nx,     p.Ny,     p.bc_x, p.bc_y
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
        );
    dg::x::Grid2d grid_out( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
        );
    //create RHS
    DG_RANK0 std::cout << "Constructing Explicit...\n";
    eule::Explicit<dg::x::CartesianGrid2d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec > feltor( grid, p); //initialize before rolkar!
    DG_RANK0 std::cout << "Constructing Implicit...\n";
    eule::Implicit<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec > rolkar( grid, p);
    DG_RANK0 std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //initial perturbation
    //dg::Gaussian3d init0(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
    dg::Gaussian init0( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);
//     dg::BathRZ init0(8, 8, 1, 0.0, 0.0, 30., 2., p.amp);
//     solovev::ZonalFlow init0(p, gp);
//     dg::CONSTANT init0( 0.);
//      dg::Vortex init0(  p.posX*p.lx, p.posY*p.ly, 0, p.sigma, p.amp);
    //background profile
//     solovev::Nprofile prof(p, gp); //initial background profile
//     dg::CONSTANT prof(p.bgprofamp );
    //
//     dg::LinearX prof(-p.nprofileamp/((double)p.lx), p.bgprofamp + p.nprofileamp);
//     dg::SinProfX prof(p.nprofileamp, p.bgprofamp,M_PI/(2.*p.lx));
    dg::ExpProfX prof(p.nprofileamp, p.bgprofamp, p.invkappa);
//     const dg::x::DVec prof =  dg::LinearX( -p.nprofileamp/((double)p.lx), p.bgprofamp + p.nprofileamp);
//     dg::TanhProfX prof(p.lx*p.solb,p.lx/10.,-1.0,p.bgprofamp,p.nprofileamp); //<n>
    std::vector<dg::x::DVec> y0(2, dg::evaluate( prof, grid)), y1(y0);


    //no field aligning
    y1[1] = dg::evaluate( init0, grid);
    dg::blas1::pointwiseDot(y1[1], y0[1],y1[1]); //<n>*ntilde

    dg::blas1::axpby( 1., y1[1], 1., y0[1]); //initialize ni = <n> + <n>*ntilde
    dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); //initialize ni-1
//     dg::blas1::pointwiseDot(rolkar.damping(),y0[1], y0[1]); //damp with gaussprofdamp
    DG_RANK0 std::cout << "intiialize ne" << std::endl;
    feltor.initializene( y0[1], y0[0]);
    DG_RANK0 std::cout << "Done!\n";

    dg::DefaultSolver< std::vector<dg::x::DVec> > solver( rolkar, y0,
            y0[0].size(), p.eps_time);
    dg::ImExMultistep< std::vector<dg::x::DVec> > karniadakis( "ImEx-BDF-3-3", y0);
    std::cout << "intiialize karniadakis" << std::endl;
    karniadakis.init( std::tie(feltor, rolkar, solver), 0., y0, p.dt);
    DG_RANK0 std::cout << "Done!\n";

    double time = 0;
    unsigned step = 0;

    const double mass0 = feltor.mass();
    double E0 = feltor.energy(), energy0 = E0, E1 = 0.;

    DG_RANK0 std::cout << "Begin computation \n";
    DG_RANK0 std::cout << std::scientific << std::setprecision( 2);


    std::string output = "netcdf";
    //create timer
    dg::Timer t;
    t.tic();
#ifdef WITH_GLFW
    output = "glfw";
    if( "glfw" == output)
    {
        const double mass_blob0 = mass0 - grid.lx()*grid.ly();
        dg::DVec dvisual( grid.size(), 0.);
        dg::DVec dvisual2( grid.size(), 0.);
        dg::HVec hvisual( grid.size(), 0.), visual(hvisual),avisual(hvisual);
        dg::IHMatrix equi = dg::create::backscatter( grid);
        draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);
        /////////glfw initialisation ////////////////////////////////////////////
        std::stringstream title;
        dg::file::WrappedJsonValue js = dg::file::file2Json( "window_params.json");
        GLFWwindow* w = draw::glfwInitAndCreateWindow( js["cols"].asUInt()*js["width"].asUInt()*p.lx/p.ly, js["rows"].asUInt()*js["height"].asUInt(), "");
        draw::RenderHostData render(js["rows"].asUInt(), js["cols"].asUInt());
        while ( !glfwWindowShouldClose( w ))
        {

            dg::assign( y0[0], hvisual);
            dg::blas2::gemv( equi, hvisual, visual);
            colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(), (double)-1e14, thrust::maximum<double>() );
    //         colors.scalemin() = -colors.scalemax();
            //colors.scalemin() = 1.0;
            colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );

            title << std::setprecision(2) << std::scientific;
            //title <<"ne / "<<(double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() )<<"  " << colors.scalemax()<<"\t";
            title <<"ne-1 / " << colors.scalemin()<<"\t";

            render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

            //draw ions
            //thrust::transform( y1[1].begin(), y1[1].end(), dvisual.begin(), dg::PLUS<double>(-0.));//ne-1
            dg::assign( y0[1], hvisual);
            dg::blas2::gemv( equi, hvisual, visual);
            colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );
            //colors.scalemin() = 1.0;
    //         colors.scalemin() = -colors.scalemax();
            colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );

            title << std::setprecision(2) << std::scientific;
            //title <<"ni / "<<(double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() )<<"  " << colors.scalemax()<<"\t";
            title <<"ni-1 / " << colors.scalemin()<<"\t";

            render.renderQuad(visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


            //draw potential
            //transform to Vor
    //        dvisual=feltor.potential()[0];
    //        dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
    //        hvisual = y1[1];
             dg::assign( feltor.potential()[0], hvisual);
            dg::blas2::gemv( equi, hvisual, visual);
            colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );

            colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax() ,thrust::minimum<double>() );

    //         //colors.scalemin() = 1.0;
    //          colors.scalemin() = -colors.scalemax();
    //          colors.scalemin() = -colors.scalemax();
            //colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
            title <<"Potential / "<< colors.scalemax() << " " << colors.scalemin()<<"\t";

            render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
            //draw potential
            //transform to Vor
            dvisual=feltor.potential()[0];
            dg::blas2::gemv( rolkar.laplacianM(), dvisual, y1[1]);
            dg::assign( y1[1], hvisual);
             //hvisual = feltor.potential()[0];
            dg::blas2::gemv( equi, hvisual, visual);
            colors.scalemax() = (double)thrust::reduce( visual.begin(), visual.end(),  (double)-1e14, thrust::maximum<double>() );
            //colors.scalemin() = 1.0;
    //          colors.scalemin() = -colors.scalemax();
            colors.scalemin() =  (double)thrust::reduce( visual.begin(), visual.end(), colors.scalemax()  ,thrust::minimum<double>() );
            title <<"Omega / "<< colors.scalemax()<< " "<< colors.scalemin()<<"\t";

            render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


            title << std::fixed; 
            title << " &&   time = "<<time;
            glfwSetWindowTitle(w,title.str().c_str());
            title.str("");
            glfwPollEvents();
            glfwSwapBuffers( w);

            //step 
            t.tic();
            for( unsigned i=0; i<p.itstp; i++)
            {
                try{ karniadakis.step( std::tie(feltor, rolkar, solver), time, y0);}
                catch( dg::Fail& fail) { 
                    std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                    std::cerr << "Does Simulation respect CFL condition?\n";
                    glfwSetWindowShouldClose( w, GL_TRUE);
                    break;
                }
                step++;
                std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass_blob0<<"\t";
                E1 = feltor.energy();
                double diff = (E1 - E0)/p.dt; //
                double diss = feltor.energy_diffusion( );
                double coupling = feltor.coupling();
                std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                std::cout << 
                             " Ga_nex= " << feltor.radial_transport() <<
                             " Coupling= " << coupling <<
                             " Accuracy: "<< 2.*fabs((diff-diss)/(diff+diss))<<
                             " d E/dt = " << diff <<
                             " Lambda =" << diss <<  std::endl;
                E0 = E1;
            }
            dg::blas1::transform( y0[0], dvisual, dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //npe = N+1
            dvisual2 = feltor.potential()[0];
    //         p.profiles
            t.toc();
            std::cout << "\n\t Step "<<step;
            std::cout << "\n\t Average time for one step: "<<t.diff()/(double)p.itstp<<"s\n\n";
        }
        glfwTerminate();
    }
#endif //WITH_GLFW
    if( "netcdf" == output)
    {
        if( argc != 3 && argc != 4)
        {
            DG_RANK0 std::cerr << "ERROR: Wrong number of arguments for netcdf output!\nUsage: "
                    << argv[0]<<" [input.json] [output.nc]\n OR \n"
                    << argv[0]<<" [input.json] [output.nc] [initial.nc] "<<std::endl;
            dg::abort_program();
        }
        /////////////////////////////set up netcdf/////////////////////////////////////
        dg::file::NcFile file;
        std::string outputfile = argv[2];
        try{
            file.open( outputfile, dg::file::nc_clobber);
        }catch( std::exception& e)
        {
            DG_RANK0 std::cerr << "ERROR creating file "<<argv[1]<<std::endl;
            DG_RANK0 std::cerr << e.what() << std::endl;
            dg::abort_program();
        }
        std::map<std::string, dg::file::nc_att_t> att;
        att["title"] = "Output file of feltor/src/feltorSesol/feltor.cpp";
        att["Conventions"] = "CF-1.8";
        att["history"] = dg::file::timestamp( argc, argv);
        att["comment"] = "Find more info in feltor/src/feltorShw/feltorSesol.tex";
        att["source"] = "FELTOR";
        att["references"] = "https://github.com/feltor-dev/feltor";
        // Here we put the inputfile as a string without comments so that it can be read later by another parser
        att["inputfile"] = js.toStyledString();
        file.put_atts( att);

        dg::x::DMatrix dy = dg::create::dy( grid, p.bc_y, dg::centered);
        eule::Variables var = { feltor, rolkar, y0, dy, time, 0, 0};
        dg::x::IHMatrix interpolate = dg::create::interpolation( grid_out, grid);
        dg::x::HVec resultH = dg::evaluate( dg::zero, grid);
        dg::x::DVec resultD( resultH);
        dg::x::HVec resultP = dg::evaluate( dg::zero, grid_out);

        file.def_dimvar_as<double>( "time", NC_UNLIMITED, {{"axis", "T"}});
        file.def_dimvar_as<double>( "energy_time", NC_UNLIMITED, {{"axis", "T"}});
        file.defput_dim( "x", {{"axis", "X"},
            {"long_name", "x-coordinate in Cartesian system"}},
            grid_out.abscissas(0));
        file.defput_dim( "y", {{"axis", "Y"},
            {"long_name", "y-coordinate in Cartesian system"}},
            grid_out.abscissas(1));

        file.put_var( "time", {0}, time);

        for( auto& record : eule::records)
        {
            record.function ( resultD, var);
            dg::assign( resultD, resultH);
            dg::blas2::symv( interpolate, resultH, resultP);
            file.def_var_as<double>( record.name, {"time", "y","x"}, record.atts);
            file.put_var( record.name, {0, grid_out}, resultP);
        }

        for( auto& record : eule::records0d)
        {
            double data = record.function ( var);
            file.def_var_as<double>( record.name, {"energy_time"}, {record.atts});
            file.put_var( record.name, {0}, data);
        }

        std::vector<double> xprobecoords(7,1.);
        for (unsigned i=0;i<7; i++) {
            xprobecoords[i] = p.lx/8.*(1+i) ;
        }
        const std::vector<double> yprobecoords(7,p.ly/2.);
        std::vector<std::vector<double>> coords = {xprobecoords, yprobecoords};
        dg::file::ProbesParams probes_params = {
            coords, {"xprobe", "yprobe"}, "none", true
        };
        dg::file::Probes probes( file, grid, probes_params);
        probes.write( time, eule::probe_list, var);
        file.close();
        DG_RANK0 std::cout << "First write successful!\n";
        for( unsigned i=1; i<=p.maxout; i++)
        {
            dg::Timer ti;
            ti.tic();
            for( unsigned j=0; j<p.itstp; j++)
            {
                try{ karniadakis.step( std::tie(feltor, rolkar, solver), time, y0);}
                catch( dg::Fail& fail) {
                    DG_RANK0 std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                    DG_RANK0 std::cerr << "Does Simulation respect CFL condition?\n";
                    dg::abort_program();
                }
                step++;
                E1 = feltor.energy();
                var.dEdt = (E1 - E0)/p.dt;
                double diss = feltor.energy_diffusion();
                E0 = E1;
                var.accuracy = 2.*fabs( (var.dEdt-diss)/(var.dEdt + diss));
                DG_RANK0 std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass0<<"\t";
                DG_RANK0 std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                DG_RANK0 std::cout <<" d E/dt = " << var.dEdt <<" Lambda = " << diss << " -> Accuracy: "<< var.accuracy << "\n";
                file.open( outputfile, dg::file::nc_write);
                probes.write( time, eule::probe_list, var);
                for( auto& record : eule::records0d)
                {
                    double data = record.function ( var);
                    file.put_var( record.name, {0}, data);
                }
                file.close();
            }
            ti.toc();
            DG_RANK0 std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
            DG_RANK0 std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
            file.open( outputfile, dg::file::nc_write);
            for( auto& record : eule::records)
            {
                record.function ( resultD, var);
                dg::assign( resultD, resultH);
                dg::blas2::symv( interpolate, resultH, resultP);
                file.put_var( record.name, {0, grid_out}, resultP);
            }
            file.put_var("time", {0}, time);
            file.close();
        }
    }
    ////////////////////////////////////////////////////////////////////
    t.toc();
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    DG_RANK0 std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    DG_RANK0 std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    DG_RANK0 std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";
#ifdef WITH_MPI
    MPI_Finalize();
#endif //WITH_MPI
    ////////////////////////////////////////////////////////////////////

    return 0;

}
