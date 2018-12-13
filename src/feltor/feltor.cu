#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>

#include "draw/host_window.h"

#include "feltor.cuh"

int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js, gs;
    if( argc == 1)
    {
        std::ifstream is("input.json");
        std::ifstream ks("geometry_params.json");
        is >> js;
        ks >> gs;
    }
    else if( argc == 3)
    {
        std::ifstream is(argv[1]);
        std::ifstream ks(argv[2]);
        is >> js;
        ks >> gs;
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "
                  << argv[0]<<" [inputfile] [geomfile] \n";
        return -1;
    }
    const feltor::Parameters p( js);
    const dg::geo::solovev::Parameters gp(gs);
    p.display( std::cout);
    gp.display( std::cout);
    /////////////////////////////////////////////////////////////////////////
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grid
    dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);

    //create RHS
    std::cout << "Constructing Explicit...\n";
    feltor::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> feltor( grid, p, mag);
    std::cout << "Constructing Implicit...\n";
    feltor::Implicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> im( grid, p, mag);
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //First the profile and the source (on the host since we want to output those)
    dg::HVec profile = dg::pullback( dg::geo::Compose<dg::LinearX>( mag.psip(),
        p.nprofamp/mag.psip()(mag.R0(), 0.), 0.), grid);
    dg::HVec xpoint_damping = dg::evaluate( dg::one, grid);
    if( gp.hasXpoint() )
        xpoint_damping = dg::pullback(
            dg::geo::ZCutter(-1.1*gp.elongation*gp.a), grid);
    dg::HVec source_damping = dg::pullback(dg::geo::TanhDamping(
        //first change coordinate from psi to (psi_0 - psip)/psi_0
        dg::geo::Compose<dg::LinearX>( mag.psip(), -1./mag.psip()(mag.R0(), 0.),1.),
        //then shift tanh
        p.rho_source-3.*p.alpha, p.alpha, -1.), grid);
    dg::blas1::pointwiseDot( xpoint_damping, source_damping, source_damping);
    if( p.omega_source != 0)
        feltor.set_source( p.omega_source, profile, source_damping);

    dg::HVec profile_damping = dg::pullback( dg::geo::TanhDamping(
        mag.psip(), -3.*p.alpha, p.alpha, -1), grid);
    dg::blas1::pointwiseDot( xpoint_damping, profile_damping, profile_damping);
    dg::blas1::pointwiseDot( profile_damping, profile, profile);

    //Now perturbation
    dg::HVec ntilde = dg::evaluate(dg::zero,grid);
    if( p.initne == "blob" || p.initne == "straight blob")
    {
        dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
        dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
        if( p.symmetric)
            ntilde = dg::pullback( init0, grid);
        else if( p.initne == "blob")//rounds =3 ->2*3-1
            ntilde = feltor.fieldalignedn( init0, gaussianZ, (unsigned)p.Nz/2, 3);
        else if( p.initne == "straight blob")//rounds =1 ->2*1-1
            ntilde = feltor.fieldalignedn( init0, gaussianZ, (unsigned)p.Nz/2, 1);
    }
    else if( p.initne == "turbulence")
    {
        dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
        dg::BathRZ init0(16,16,Rmin,Zmin, 30.,5.,p.amp);
        if( p.symmetric)
            ntilde = dg::pullback( init0, grid);
        else
            ntilde = feltor.fieldalignedn( init0, gaussianZ, (unsigned)p.Nz/2, 1);
        dg::blas1::pointwiseDot( profile_damping, ntilde, ntilde);
    }
    else if( p.initne == "zonal")
    {
        dg::geo::ZonalFlow init0(mag.psip(), p.amp, 0., p.k_psi);
        ntilde = dg::pullback( init0, grid);
        dg::blas1::pointwiseDot( profile_damping, ntilde, ntilde);
    }
    else
        std::cerr <<"WARNING: Unknown initial condition!\n";
    std::array<std::array<dg::DVec,2>,2> y0;
    y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<dg::DVec>(profile);
    dg::blas1::axpby( 1., dg::construct<dg::DVec>(ntilde), 1., y0[0][0]);
    std::cout << "initialize ni" << std::endl;
    if( p.initphi == "zero")
        feltor.initializeni( y0[0][0], y0[0][1]);
    else if( p.initphi == "balance")
        dg::blas1::copy( y0[0][0], y0[0][1]); //set N_i = n_e
    else
        std::cerr <<"WARNING: Unknown initial condition for phi!\n";

    dg::blas1::copy( 0., y0[1][0]); //set we = 0
    dg::blas1::copy( 0., y0[1][1]); //set Wi = 0

    ////////////////////////create timer and timestepper
    //
    dg::Timer t;
    double time = 0, dt_new = p.dt, dt =0;
    unsigned step = 0;
    dg::Adaptive< dg::ARKStep<std::array<std::array<dg::DVec,2>,2>> > adaptive(
        "ARK-4-2-3", y0, grid.size(), p.eps_time);

    //since we map pointers we don't need to update those later

    std::map<std::string, const dg::DVec* > v4d;
    v4d["ne-1 / "] = &y0[0][0],               v4d["ni-1 / "] = &y0[0][1];
    v4d["Ue / "]   = &feltor.fields()[1][0],  v4d["Ui / "]   = &feltor.fields()[1][1];
    v4d["Omega / "] = &feltor.potential()[0]; v4d["Apar / "] = &feltor.induction();
    const feltor::Quantities& q = feltor.quantities();
    double dEdt = 0, accuracy = 0, dMdt = 0, accuracyM  = 0;
    std::map<std::string, const double*> v0d{
        {"energy", &q.energy}, {"ediff", &q.ediff},
        {"mass", &q.mass}, {"diff", &q.diff}, {"Apar", &q.Apar},
        {"Se", &q.S[0]}, {"Si", &q.S[1]}, {"Uperp", &q.Tperp},
        {"Upare", &q.Tpar[0]}, {"Upari", &q.Tpar[1]},
        {"dEdt", &dEdt}, {"accuracy", &accuracy},
        {"aligned", &q.aligned}
    };

    //first, update quantities in feltor

    {
        std::array<std::array<dg::DVec,2>,2> y1(y0);
        try{
            feltor( time, y0, y1);
        } catch( dg::Fail& fail) {
            std::cerr << "CG failed to converge in first step to "
                      << fail.epsilon()<<"\n";
            return -1;
        }
        feltor.update_quantities();
    }
    double energy0 = q.energy, mass0 = q.mass, E0 = energy0, M0 = mass0;
    /////////////////////////set up transfer for glfw
    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual), avisual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);
    //perp laplacian for computation of vorticity

    dg::Elliptic3d<dg::CylindricalGrid3d, dg::DMatrix, dg::DVec>
        laplacianM(grid, p.bcxP, p.bcyP, dg::PER, dg::normed, dg::centered);
    auto bhatF = dg::geo::createEPhi();
    if( p.curvmode == "true")
        bhatF = dg::geo::createBHat( mag);
    dg::SparseTensor<dg::DVec> hh = dg::geo::createProjectionTensor( bhatF, grid);
    laplacianM.set_chi( hh);

    /////////glfw initialisation ////////////////////////////////////////////
    //
    std::stringstream title;
    std::ifstream is( "window_params.js");
    is >> js;
    is.close();
    unsigned red = js.get("reduction", 1).asUInt();
    GLFWwindow* w = draw::glfwInitAndCreateWindow( (p.Nz/red+1)*js["width"].asDouble(), js["rows"].asDouble()*js["height"].asDouble(), "");
    draw::RenderHostData render(js["rows"].asDouble(), p.Nz/red + 1);

    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    dg::Average<dg::HVec> toroidal_average( grid, dg::coo3d::z);
    title << std::setprecision(2) << std::scientific;
    while ( !glfwWindowShouldClose( w ))
    {
        for( auto pair : v4d)
        {
            if(pair.first == "Omega / ")
            {
                dg::blas2::gemv( laplacianM, *pair.second, dvisual);
                dg::assign( dvisual, hvisual);
            }
            else
                dg::assign( *pair.second, hvisual);
            dg::blas2::gemv( equi, hvisual, visual);
            colors.scalemax() = (double)thrust::reduce(
                visual.begin(), visual.end(), 0., thrust::maximum<double>() );
            colors.scalemin() = -colors.scalemax();
            title <<pair.first << colors.scalemax()<<"\t";
            for( unsigned k=0; k<p.Nz/red;k++)
            {
                unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
                dg::HVec part( visual.begin() +  k*red   *size,
                               visual.begin() + (k*red+1)*size);
                render.renderQuad( part, grid.n()*grid.Nx(),
                                         grid.n()*grid.Ny(), colors);
            }
            dg::blas1::scal(avisual,0.);
            toroidal_average(visual,avisual);
            render.renderQuad( avisual, grid.n()*grid.Nx(),
                                        grid.n()*grid.Ny(), colors);
        }
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
            try{
                do
                {
                    dt = dt_new;
                    adaptive.step( feltor, im, time, y0, time, y0, dt_new,
                        dg::pid_control, dg::l2norm, p.rtol, 1e-10);
                    if( adaptive.failed())
                        std::cout << "FAILED STEP! REPEAT!\n";
                }while ( adaptive.failed());
            }
            catch( dg::Fail& fail) {
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;
            }
            step++;
            feltor.update_quantities();
            std::cout << "Timestep "<<dt<<"\n";
            dEdt = (*v0d["energy"] - E0)/dt, dMdt = (*v0d["mass"] - M0)/dt;
            E0 = *v0d["energy"], M0 = *v0d["mass"];
            accuracy  = 2.*fabs( (dEdt - *v0d["ediff"])/( dEdt + *v0d["ediff"]));
            accuracyM = 2.*fabs( (dMdt - *v0d["diff"])/( dMdt + *v0d["diff"]));

            q.display(std::cout);
            std::cout << "(m_tot-m_0)/m_0: "<< (*v0d["mass"]-mass0)/mass0<<"\t";
            std::cout << "(E_tot-E_0)/E_0: "<< (*v0d["energy"]-energy0)/energy0<<"\t";
            std::cout <<" d E/dt = " << dEdt
              <<" Lambda = " << *v0d["ediff"]
              <<" -> Accuracy: " << accuracy << "\n";
            std::cout <<" d M/dt = " << dMdt
                      <<" Lambda = " << *v0d["diff"]
                      <<" -> Accuracy: " << accuracyM << "\n";

        }
        t.toc();
        std::cout << "\n\t Step "<<step;
        std::cout << "\n\t Average time for one step: "<<t.diff()/(double)p.itstp<<"s\n\n";
    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////
    return 0;

}
