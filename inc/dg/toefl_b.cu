#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/timer.cuh"
//#include "draw/device_window.cuh"
#include "draw/host_window.h"

#include "algorithm.h"
#include "toefl.cuh"
#include "backend/xspacelib.cuh"
#include "backend/typedefs.cuh"



const unsigned n = 3;
const unsigned Nx = 100;
const unsigned Ny = 20;
const double lx = 5.;
const double ly = 1.;

const double Pr = 10;
const double Ra = 5e5;

const unsigned k = 3;
const double dt = 2e-7;
const unsigned N = 5; //steps between output

double eps = 1e-3;


double groundState( double x, double y) { return ly/2. - y;}
/**
 * @brief Functor returning a gaussian
 * \f[
   f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}} 
   \f]
 */
struct Gaussian
{
    /**
     * @brief Functor returning a gaussian
     *
     * @param x0 x-center-coordinate
     * @param y0 y-center-coordinate
     * @param sigma_x x - variance
     * @param sigma_y y - variance 
     * @param amp Amplitude
     */
    Gaussian( float x0, float y0, float sigma_x, float sigma_y, float amp)
        : x00(x0), y00(y0), sigma_x(sigma_x), sigma_y(sigma_y), amplitude(amp){}
    /**
     * @brief Return the value of the gaussian
     *
     * \f[
       f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}} 
       \f]
     * @param x x - coordinate
     * @param y y - coordinate
     *
     * @return gaussian
     */
    float operator()(float x, float y)
    {
        return  amplitude*
                   exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                          (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
  private:
    float  x00, y00, sigma_x, sigma_y, amplitude;

};

int main()
{
    std::stringstream title;
    title << " temperature / "<<1. <<" time 0";
    GLFWwindow* w = draw::glfwInitAndCreateWindow(1000, 200, title.str().c_str());
    title.str("");
    draw::RenderHostData render(1,1);


    ///////////////////////////////////////////////////////////////////////
    std::cout << "# of Legendre coefficients: " << n<<std::endl;
    std::cout << "# of grid cells:            " << Nx*Ny<<std::endl;
    std::cout << "Timestep                    " << dt << std::endl;

    //create initial vector
    const dg::Grid2d<double> grid( 0, lx, 0, ly,n, Nx, Ny, dg::PER, dg::DIR);
    dg::Gaussian gaussian( 1., ly/2., .1, .1, 1);
    dg::DVec theta = dg::evaluate ( gaussian, grid);
    std::vector<dg::DVec> y0(2, theta);
    y0[1] = dg::DVec( grid.size(), 0.); //omega is zero

    //create RHS and AB
    dg::Toefl< dg::cartesian::Grid2d, dg::DMatrix, dg::DVec> test( grid, Ra, Pr, eps); 
    dg::AB< k, std::vector<dg::DVec> > ab( y0);

    //create visualisation vectors
    dg::DVec visual(  grid.size());
    dg::DVec ground = dg::evaluate ( groundState, grid), temperature( ground);
    dg::IDMatrix equidistant = dg::create::backscatter( grid );
    draw::ColorMapRedBlueExt colors( 1.);
    colors.scale() =  1.;
    ab.init( test, y0, dt);
    double time = 0;
    while (!glfwWindowShouldClose(w))
    {
        //compute the total temperature
        dg::blas1::axpby( 1., y0[0],  0., temperature);
        dg::blas1::axpby( 1., ground, 1., temperature);
        //transform field to an equidistant grid
        dg::blas2::symv( equidistant, temperature, visual);

        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        render.renderQuad( visual, n*grid.Nx(), n*grid.Ny(), colors);
        title << " temperature / "<<colors.scale() <<" time "<<std::setw(3)<<time;
        glfwSetWindowTitle( w, title.str().c_str());
        title.str("");
        glfwPollEvents();
        glfwSwapBuffers( w);
        //step 
        for( unsigned i=0; i<N; i++)
        {
            ab( test, y0);
            time += (double)N*dt;
        }
    }
    ////////////////////////////////////////////////////////////////////
    glfwTerminate();

    return 0;

}

