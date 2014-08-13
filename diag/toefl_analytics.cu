#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"
#include "dg/arakawa.h"
#include "dg/backend/xspacelib.cuh"
#include "dg/backend/timer.cuh"
#include "dg/functors.h"
#include "file/read_input.h"
#include "file/file.h"

#include "galerkin/parameters.h"

template <class Matrix, class container>
struct Vesqr
{
    Vesqr( const dg::Grid2d<double>& grid, double kappa): dx( grid.size()), dy(dx), one( grid.size(), 1.), w2d( dg::create::w2d(grid)), binv( evaluate( dg::LinearX( kappa, 1.), grid)), arakawa(grid){}
    const container& operator()( const container& phi)
    {
        dg::blas2::gemv( arakawa.dx(), phi, dx);
        dg::blas2::gemv( arakawa.dy(), phi, dy);
        dg::blas1::pointwiseDot( binv, dx, dx);
        dg::blas1::pointwiseDot( binv, dy, dy);
        dg::blas1::pointwiseDot( dx, dx, dx);
        dg::blas1::pointwiseDot( dy, dy, dy);
        dg::blas1::axpby( 1., dx, 1.,  dy);
        return dy;

    }
  private:
    container dx, dy, one, w2d, binv;    
    dg::ArakawaX<Matrix, container> arakawa;

};

template <class Matrix, class container>
struct Nonlinearity 
{
    Nonlinearity( const dg::Grid2d<double>& grid): dxn( grid.size()), dyn(dxn), dxphi( dxn), dyphi( dyn), logni(dyn), one( grid.size(), 1.), w2d( dg::create::w2d(grid)), arakawa(grid){}
    const container& operator()( const container& ni, const container& phi)
    {
        thrust::transform( ni.begin(), ni.end(), logni.begin(), dg::LN<double>());
        dg::blas2::gemv( arakawa.dx(), logni , dxn);
        dg::blas2::gemv( arakawa.dx(), phi , dxphi);
        dg::blas2::gemv( arakawa.dy(), logni , dxn);
        dg::blas2::gemv( arakawa.dy(), phi , dxphi);

        dg::blas1::pointwiseDot( dxn, dxphi, dxn);
        dg::blas1::pointwiseDot( dyn, dyphi, dyn);
        dg::blas1::axpby( 1., dxn, 1.,  dyn);
        return dyn; 
    }
  private:
    container dxn, dyn, dxphi, dyphi, logni, one, w2d;    
    dg::ArakawaX<Matrix, container> arakawa;

};


int main( int argc, char* argv[])
{
    std::stringstream title;
    std::vector<double> v = file::read_input( "window_params.txt");
    GLFWwindow* w = draw::glfwInitAndCreateWindow( v[3]*v[2], v[4]*v[1], "");
    draw::RenderHostData render( v[1], v[2]);

    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [inputfile]\n";
        return -1;
    }

    std::string in;
    file::T5rdonly t5file( argv[1], in);
    unsigned nlinks = t5file.get_size();

    int layout = 0;
    if( in.find( "TOEFL") != std::string::npos)
        layout = 0;
    else if( in.find( "INNTO") != std::string::npos)
        layout = 1;
    else 
        std::cerr << "Unknown input file format: default to 0"<<std::endl;
    const Parameters p( file::read_input( in), layout);
    p.display();
    dg::Grid2d<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::HVec visual(  grid.size(), 0.), input( visual), input2( visual);
    dg::HVec w2d = dg::create::w2d( grid);
    dg::HMatrix equi = dg::create::backscatter( grid);
    dg::HMatrix laplacianM = dg::create::laplacianM( grid, dg::normed);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    bool running = true;
    unsigned index = 1;
    std::cout << "PRESS N FOR NEXT FRAME!\n";
    std::cout << "PRESS P FOR PREVIOUS FRAME!\n";
    std::vector<double> mass, energy, diffusion, dissipation, massAcc, energyAcc;
    if( p.global)
    {
        t5file.get_xfile( mass, "mass");
        t5file.get_xfile( energy, "energy");
        t5file.get_xfile( diffusion, "diffusion");
        t5file.get_xfile( dissipation, "dissipation");
        massAcc.resize(mass.size()), energyAcc.resize(mass.size());
        mass.insert(mass.begin(), 0), mass.push_back(0);
        energy.insert( energy.begin(), 0), energy.push_back(0);
        for(unsigned i=0; i<massAcc.size(); i++ )
        {
            massAcc[i] = (mass[i+2]-mass[i])/2./p.dt; //first column
            //if( i < 10 || i > num_entries - 20)
            //    std::cout << "i "<<i<<"\t"<<massAcc[i]<<"\t"<<mass[i+1]<<std::endl;
            massAcc[i] = fabs(2.*(massAcc[i]-diffusion[i])/(massAcc[i]+diffusion[i]));
            energyAcc[i] = (energy[i+2]-energy[i])/2./p.dt;
            energyAcc[i] = fabs(2.*(energyAcc[i]-dissipation[i])/(energyAcc[i]+dissipation[i]));
            //energyAcc[i] = fabs(energyAcc[i]-dissipation[i])/p.nu;
        }
    }

    std::cout << std::scientific << std::setprecision( 2);
    /*
        bool waiting = true;
        do
        {
            glfwPollEvents();
            if( glfwGetKey( 'S')){
                waiting = false;
            }
        }while( waiting && !glfwGetKey( GLFW_KEY_ESC) && glfwGetWindowParam( GLFW_OPENED));
        */
    Vesqr<dg::HMatrix, dg::HVec> vesqr(grid, p.kappa);
    Nonlinearity<dg::HMatrix, dg::HVec> nonlinear( grid);
    while (!glfwWindowShouldClose(w) && index < nlinks + 1 )
    {
        t5file.get_field( input, "potential", index);

        //dg::blas2::gemv( laplacianM, input, visual);
        //input.swap( visual);
        dg::blas2::gemv( equi, input, visual);

        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        if( v[6] > 0)
            colors.scale() = v[6];
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"potential / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //transform phi
        //t5file.get_field( input, "potential", index);
        t5file.get_field( input2, "ions", index);
        visual = nonlinear( input2, input);
        //visual = vesqr( input);
        input.swap( visual);
        std::cout << " U_phi "<< 0.5*dg::blas2::dot( input, w2d, input2)<<std::endl;
        dg::blas2::gemv( equi, input, visual);
        //compute the color scale (take the same as for omega)
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        if(colors.scale() == 0) { colors.scale() = 1;}
        if( v[7] > 0 )
            colors.scale() = v[7];
        //draw phi and swap buffers
        title <<"Nonlinearity / "<<colors.scale()<<"\t";
        title << std::fixed; 
        title << " &&  time = "<<t5file.get_time( index); //read time as double from string
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        if(p.global)
        {
            std::cout <<"(m_tot-m_0)/m_0: "<<(mass[(index-1)*p.itstp]-mass[1])/(mass[1]-grid.lx()*grid.ly()) //blob mass is mass[] - Area
                      <<"\t(E_tot-E_0)/E_0: "<<(energy[(index-1)*p.itstp]-energy[1])/energy[1]
                      <<"\tAccuracy: "<<energyAcc[(index-1)*p.itstp]<<std::endl;
        }
        bool waiting = true;
        do
        {
            glfwPollEvents();
            if( glfwGetKey(w, 'B')||glfwGetKey(w, 'P') ){
                index -= v[5];
                waiting = false;
            }
            else if( glfwGetKey(w, 'N') ){
                index +=v[5];
                waiting = false;
            }
            //glfwWaitEvents();
        }while( waiting && !glfwGetKey(w, GLFW_KEY_ESCAPE) );

    }
    glfwTerminate();
    return 0;
}
