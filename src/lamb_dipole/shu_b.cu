#include <iostream>
#include <iomanip>
#include <sstream>
#include <thrust/host_vector.h>

#include "dg/algorithm.h"

#ifndef WITHOUT_GLFW
#include "draw/host_window.h"
#endif
#include "dg/file/file.h"

#include "init.h"
#include "diag.h"
#include "shu.h"


int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js;
    if( argc == 1)
        dg::file::file2Json( "input/default.json", js, dg::file::comments::are_discarded);
    else
        dg::file::file2Json( argv[1], js);
    std::cout << js <<std::endl;
    dg::file::WrappedJsonValue ws( js, dg::file::error::is_throw);

    /////////////////////////////////////////////////////////////////
    dg::CartesianGrid2d grid = shu::createGrid( ws["grid"]);
    dg::DVec w2d( dg::create::weights(grid));
    /////////////////////////////////////////////////////////////////

    dg::HVec omega;
    try{
        omega = shu::initial_conditions( ws["init"], grid);
    }
    catch ( std::exception& e) {
        std::cerr << e.what()<<"\n";
        std::cerr << "Exit now!\n";
        return -1;
    }

    dg::DVec y0( omega ), y1( y0);
    //subtract mean mass
    if( grid.bcx() == dg::PER && grid.bcy() == dg::PER)
    {
        double meanMass = dg::blas1::dot( y0, w2d)/(double)(grid.lx()*grid.ly());
        dg::blas1::axpby( -meanMass, 1., 1., y0);
    }
    //make solver and stepper
    shu::Shu<dg::CartesianGrid2d, dg::DMatrix, dg::DVec>
        shu( grid, ws);
    shu::Diffusion<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> diffusion( shu);
    shu::Implicit<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> shu_implicit( grid, ws);
    shu::Filter<dg::IDMatrix, dg::DVec> shu_filter( grid, ws) ;
    if( "mms" == ws["init"]["type"].asString())
    {
        double sigma = ws["init"].get( "sigma", 0.2).asDouble();
        double velocity = ws["init"].get( "velocity", 0.1).asDouble();
        shu.set_mms_source( sigma, velocity, grid.ly());
    }

    double time = 0;
    shu::Variables var = {shu, grid, y0, time, w2d, 0., ws};
    dg::Timer t;
    t.tic();
    std::cout << "Trigger potential computation\n";
    shu( 0., y0, y1);
    t.toc();
    var.duration = t.diff();
    for( auto record : shu::diagnostics1d_list)
    {
        double result = record.function( var);
        std::cout  << "Diagnostics "<<record.name<<" "<<result<<"\n";
    }

    /// ////////////// Initialize timestepper ///////////////////////
    std::string stepper = ws[ "timestepper"].get( "type", "FilteredExplicitMultistep").asString();
    std::string tableau = ws[ "timestepper"].get( "tableau", "ImEx-BDF-3-3").asString();
    std::string regularization = ws[ "regularization"].get( "type", "modal").asString();
    dg::DefaultSolver<dg::DVec> solver;
    dg::ImExMultistep<dg::DVec> imex;
    dg::ImplicitMultistep<dg::DVec> implicit;
    dg::ShuOsher<dg::DVec> shu_osher;
    dg::FilteredExplicitMultistep<dg::DVec> multistep;
    dg::Adaptive<dg::FilteredERKStep<dg::DVec> > adaptive;
    dg::Adaptive<dg::ARKStep<dg::DVec> > adaptive_imex;
    dg::Adaptive<dg::DIRKStep<dg::DVec> > adaptive_implicit;
    if( regularization == "modal" || regularization == "swm" || regularization == "median")
    {
        if( stepper != "FilteredExplicitMultistep" && stepper != "Shu-Osher" && stepper != "ERK" )
        {
            throw std::runtime_error( "Error: Limiter regularization only works with either FilteredExplicitMultistep or Shu-Osher or ERK!");
            return -1;
        }
    }
    else if( regularization != "viscosity" && regularization != "none")
        throw std::runtime_error( "ERROR: Unkown regularization type "+regularization);

    double dt = 0.;
    double rtol = 1., atol = 1.;
    auto odeint = std::unique_ptr<dg::aTimeloop<dg::DVec>>();
    if( "ImExMultistep" == stepper)
    {
        dt = ws[ "timestepper"].get( "dt", 2e-3).asDouble();
        double eps_time = ws[ "timestepper"].get( "eps_time", 1e-10).asDouble();
        solver.construct( diffusion, y0, y0.size(), eps_time);
        imex.construct( tableau, y0);
        odeint = std::make_unique<dg::MultistepTimeloop<dg::DVec>>( imex,
            std::tie( shu, diffusion, solver), time, y0, dt);
    }
    else if( "ImplicitMultistep" == stepper)
    {
        dt = ws[ "timestepper"].get( "dt", 2e-3).asDouble();
        implicit.construct( tableau, y0);
        odeint = std::make_unique<dg::MultistepTimeloop<dg::DVec>>( implicit,
            std::tie( shu_implicit, shu_implicit), time, y0, dt);
    }
    else if( "Shu-Osher" == stepper)
    {
        dt = ws[ "timestepper"].get( "dt", 2e-3).asDouble();
        shu_osher.construct( tableau, y0);
        odeint = std::make_unique<dg::SinglestepTimeloop<dg::DVec>>( shu_osher,
            std::tie( shu, shu_filter), dt);
    }
    else if( "FilteredExplicitMultistep" == stepper)
    {
        dt = ws[ "timestepper"].get( "dt", 2e-3).asDouble();
        multistep.construct( tableau, y0);
        odeint = std::make_unique<dg::MultistepTimeloop<dg::DVec>>( multistep,
            std::tie( shu, shu_filter), time, y0, dt);
    }
    else if( "ERK" == stepper)
    {
        rtol = ws["timestepper"].get("rtol", 1e-5).asDouble();
        atol = ws["timestepper"].get("atol", 1e-5).asDouble();
        adaptive.construct( tableau, y0);
        odeint = std::make_unique<dg::AdaptiveTimeloop<dg::DVec>>( adaptive,
            std::tie( shu, shu_filter), dg::pid_control, dg::l2norm, rtol, atol);
    }
    else if( "ARK" == stepper)
    {
        double eps_time = ws[ "timestepper"].get( "eps_time", 1e-10).asDouble();
        rtol = ws["timestepper"].get("rtol", 1e-5).asDouble();
        atol = ws["timestepper"].get("atol", 1e-5).asDouble();
        solver.construct( diffusion, y0, y0.size(), eps_time);
        adaptive_imex.construct( tableau, y0);
        odeint = std::make_unique<dg::AdaptiveTimeloop<dg::DVec>>( adaptive_imex,
            std::tie( shu, diffusion, solver), dg::pid_control, dg::l2norm, rtol, atol);
    }
    else if( "DIRK" == stepper)
    {
        rtol = ws["timestepper"].get("rtol", 1e-5).asDouble();
        atol = ws["timestepper"].get("atol", 1e-5).asDouble();
        adaptive_implicit.construct( tableau, y0);
        odeint = std::make_unique<dg::AdaptiveTimeloop<dg::DVec>>( adaptive_implicit,
            std::tie( shu_implicit, shu_implicit), dg::i_control, dg::l2norm, rtol, atol);
    }
    else
    {
        throw std::runtime_error( "Error! Timestepper "+stepper+" not recognized!\n");

        return -1;
    }
    unsigned maxout =    ws[ "output"].get( "maxout", 100).asUInt();
    double tend = ws["output"].get( "tend", 100).asDouble();
    double deltaT = tend/(double)maxout;
    std::string output = ws[ "output"].get( "type", "glfw").asString();
#ifndef WITHOUT_GLFW
    if( "glfw" == output)
    {
        ////////////////////////////////glfw//////////////////////////////
        //create visualisation vectors
        dg::DVec visual( grid.size());
        dg::HVec hvisual( grid.size());
        //transform vector to an equidistant grid
        std::stringstream title;
        GLFWwindow* w = draw::glfwInitAndCreateWindow(600, 600, "");
        draw::RenderHostData render( 1,1);
        draw::ColorMapRedBlueExt colors( 1.);
        dg::IDMatrix equidistant = dg::create::backscatter( grid );
        while (!glfwWindowShouldClose(w) && time < tend)
        {
            dg::blas2::symv( equidistant, y0, visual);
            colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
            //draw and swap buffers
            dg::assign( visual, hvisual);
            render.renderQuad( hvisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
            title << "Time "<<time<< " \ttook "<<t.diff()<<"\t per step";
            glfwSetWindowTitle(w, title.str().c_str());
            title.str("");
            glfwPollEvents();
            glfwSwapBuffers(w);
            //step
            t.tic();
            try{
                odeint->integrate( time, y0, time+deltaT, y0, dg::to::exact);
            } catch( dg::Fail& fail) {
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                return -1;
            }
            t.toc();
            var.duration = t.diff();
            std::cout << "\n\n\tTime "<<time<<" Current timestep: "<<odeint->get_dt()<<"\n\n";
        }
        glfwTerminate();
    }
#endif //WITHOUT_GLFW
    if( "netcdf" == output)
    {
        std::string inputfile = js.toStyledString(); //save input without comments, which is important if netcdf file is later read by another parser
        std::string outputfile;
        if( argc == 1 || argc == 2)
            outputfile = "shu.nc";
        else
            outputfile = argv[2];
        /// //////////////////////set up netcdf/////////////////////////////////////
        dg::file::NC_Error_Handle err;
        int ncid=-1;
        try{
            err = nc_create( outputfile.c_str(),NC_NETCDF4|NC_CLOBBER, &ncid);
        }catch( std::exception& e)
        {
            std::cerr << "ERROR creating file "<<outputfile<<std::endl;
            std::cerr << e.what()<<std::endl;
           return -1;
        }
        /// Set global attributes
        std::map<std::string, std::string> att;
        att["title"] = "Output file of feltor/src/lamb_dipole/shu_b.cu";
        att["Conventions"] = "CF-1.7";
        ///Get local time and begin file history
        auto ttt = std::time(nullptr);
        auto tm = *std::localtime(&ttt);

        std::ostringstream oss;
        ///time string  + program-name + args
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        for( int i=0; i<argc; i++) oss << " "<<argv[i];
        att["history"] = oss.str();
        att["comment"] = "Find more info in feltor/src/lamb_dipole/shu.tex";
        att["source"] = "FELTOR";
        att["references"] = "https://github.com/feltor-dev/feltor";
        att["inputfile"] = inputfile;
        for( auto pair : att)
            err = nc_put_att_text( ncid, NC_GLOBAL,
                pair.first.data(), pair.second.size(), pair.second.data());

        int dim_ids[3], tvarID;
        std::map<std::string, int> id1d, id3d;
        err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid,
                {"time", "y", "x"});

        //Create field IDs
        for( auto& record : shu::diagnostics2d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id3d[name] = 0;
            err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
                    &id3d.at(name));
            err = nc_put_att_text( ncid, id3d.at(name), "long_name", long_name.size(),
                long_name.data());
        }
        for( auto& record : shu::diagnostics1d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id1d[name] = 0;
            err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1, &dim_ids[0],
                &id1d.at(name));
            err = nc_put_att_text( ncid, id1d.at(name), "long_name", long_name.size(),
                long_name.data());
        }
        // Output static vars
        dg::DVec resultD = dg::evaluate( dg::zero, grid);
        dg::HVec resultH( resultD);
        for( auto& record : shu::diagnostics2d_static_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            int staticID = 0;
            err = nc_def_var( ncid, name.data(), NC_DOUBLE, 2, &dim_ids[1],
                &staticID);
            err = nc_put_att_text( ncid, staticID, "long_name", long_name.size(),
                long_name.data());
            record.function( resultD, var);
            dg::assign( resultD, resultH);
            dg::file::put_var_double( ncid, staticID, grid, resultH);
        }
        size_t start[3] = {0, 0, 0};
        size_t count[3] = {1, grid.n()*grid.Ny(), grid.n()*grid.Nx()};
        ///////////////////////////////////first output/////////////////////////
        for( auto& record : shu::diagnostics2d_list)
        {
            record.function( resultD, var);
            dg::assign( resultD, resultH);
            dg::file::put_vara_double( ncid, id3d.at(record.name), start[0], grid, resultH);
        }
        for( auto& record : shu::diagnostics1d_list)
        {
            double result = record.function( var);
            nc_put_vara_double( ncid, id1d.at(record.name), start, count, &result);
        }
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        ///////////////////////////////////timeloop/////////////////////////
        try{
        for( unsigned u=1; u<=maxout; u++)
        {

            dg::Timer ti;
            ti.tic();
            odeint->integrate( time, y0, time+deltaT, y0, dg::to::exact);
            ti.toc();
            var.duration = ti.diff();
            //output all fields
            start[0] = u;
            for( auto& record : shu::diagnostics2d_list)
            {
                record.function( resultD, var);
                dg::assign( resultD, resultH);
                dg::file::put_vara_double( ncid, id3d.at(record.name), start[0], grid, resultH);
            }
            for( auto& record : shu::diagnostics1d_list)
            {
                double result = record.function( var);
                nc_put_vara_double( ncid, id1d.at(record.name), start, count, &result);
            }
            err = nc_put_vara_double( ncid, tvarID, start, count, &time);
            std::cout << "\n\t Step "<<u <<" of "<<maxout <<" at time "<<time;
            std::cout << "\n\t Average time for one step: "<<var.duration<<"s\n\n"<<std::flush;
        }
        }
        catch( dg::Fail& fail) {
            std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
            std::cerr << "Does Simulation respect CFL condition?\n";
            err = nc_close(ncid);
            return -1;
        }
        err = nc_close(ncid);
    }
    if( !("netcdf" == output) && !("glfw" == output))
    {
        std::cerr<<"Error: Wrong value for output type "<<output<<" Must be glfw or netcdf! Exit now!";
        return -1;
    }
    ////////////////////////////////////////////////////////////////////
    std::cout << "Time "<<time<<std::endl;
    for( auto record : shu::diagnostics1d_list)
    {
        double result = record.function( var);
        std::cout  << "Diagnostics "<<record.name<<" "<<result<<"\n";
    }

    return 0;

}
