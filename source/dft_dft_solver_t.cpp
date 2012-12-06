#include <iostream>
#include <iomanip>
#include <GL/glfw.h>
#include <sstream>
#include "toefl.h"
#include "dft_dft_solver.h"
#include "blueprint.h"

using namespace std;
using namespace toefl;
    
unsigned N;
double slit = 2./500.; //half distance between pictures in units of width

/*! @brief Adds a gaussian to a given matrix
 *
 * The function interprets the given points as inner, cell centered points of a 
 * square box. [0,1]x[0,1]
 * , where the first index is the y and the second index is the x point. 
 * (0,0) corresponds  to the lower left corner.
 * It adds the values of the smooth function
 * \f[
   f(x) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}} 
   \f]
   where A is a constant and \f$ x,y = 0...1 \f$.
 * \param m the matrix
 * @param x0 x-position of maximum 0<x0<1
 * @param y0 y-position of maximum 0<y0<1
 * @param sigma_x Varianz in x (FWHM = 2.35*sigma_x)
 * @param sigma_y Varianz in y (FWHM = 2.35*sigma_y)
 * @param amplitude Value of maximum
 */
template< class M>
void init_gaussian( M& m, const double x0, const double y0, 
                          const double sigma_x, const double sigma_y,
                          const double amplitude)
{
    const size_t rows = m.rows(), cols = m.cols();
    const double hx = 1./(double)(cols), hy = 1./(double)(rows); 
    double x,y;
    for( unsigned i=0; i<rows; i++)
        for( unsigned j=0; j<cols; j++)
        {
            x = (j+0.5)*hx;
            y = (i+0.5)*hy;
            m(i,j) += amplitude*
                   exp( -(double)((x-x0)*(x-x0)/2./sigma_x/sigma_x+
                                  (y-y0)*(y-y0)/2./sigma_y/sigma_y) );
        }
}

void init( Physical& phys, Algorithmic& alg, Boundary& bound)
{
    vector<double> para;
    try{ para = read_input( "input.test"); }
    catch (Message& m) {  m.display(); return ;}
    phys.d = para[1];
    phys.g_e = phys.g[0] = para[2];
    phys.g[1] = para[3];
    phys.tau[0] = para[4];
    phys.tau[1] = para[22];
    phys.nu = para[8];
    phys.mu[1] = para[23];
    phys.a[1] = para[24];
    phys.kappa = para[6];

    phys.a[0] = 1. -phys.a[1];
    phys.g[1] = (phys.g_e - phys.a[1] * phys.g[1])/(1.-phys.a[1]);
    phys.mu[0] = 1.0;

    bound.ly = para[12];
    alg.nx = para[13];
    alg.ny = para[14];
    alg.dt = para[15];
    N = para[16];

    alg.h = bound.ly / (double)alg.ny;
    bound.lx = (double)alg.nx * alg.h;
    bound.bc_x = TL_PERIODIC;
}

void drawScene( const DFT_DFT_Solver<2>& solver, unsigned nx, unsigned ny)
{
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);
    double temp;
    static Texture_RGBf tex( ny, nx);
    static Matrix<double, TL_DFT> field( ny, nx);
    field = solver.getField( TL_ELECTRONS);
    temp = 0;
    for( unsigned i=0; i<field.rows(); i++)
        for( unsigned j=0; j<field.cols(); j++)
            if( abs(field(i,j)) > temp) temp = field(i,j);
#ifdef TL_DEBUG
    cout <<"ne temp "<<temp<<endl;
#endif
    gentexture_RGBf( tex, field, temp);
    // image comes from texarray on host
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.cols(), tex.rows(), 0, GL_RGB, GL_FLOAT, tex.getPtr());
    glLoadIdentity();
    //Draw a textured quad
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( -1.0/3.0-slit, -1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( -1.0/3.0-slit, 1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0, 1.0);
    glEnd();
    field = solver.getField( TL_IONS);
    temp = 0;
    for( unsigned i=0; i<field.rows(); i++)
        for( unsigned j=0; j<field.cols(); j++)
            if( abs(field(i,j)) > temp) temp = field(i,j);
#ifdef TL_DEBUG
    cout <<"ni temp "<<temp<<endl;
#endif
    gentexture_RGBf( tex, field, temp);
    // image comes from texarray on host
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.cols(), tex.rows(), 0, GL_RGB, GL_FLOAT, tex.getPtr());
    glLoadIdentity();
    //Draw a textured quad
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0/3.0+slit, -1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0/3.0-slit, -1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0/3.0-slit, 1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0/3.0+slit, 1.0);
    glEnd();
    field = solver.getField( TL_POTENTIAL);
    temp = 0;
    for( unsigned i=0; i<field.rows(); i++)
        for( unsigned j=0; j<field.cols(); j++)
            if( abs(field(i,j)) > temp) temp = field(i,j);
#ifdef TL_DEBUG
    cout <<"phi temp "<<temp<<endl;
#endif
    gentexture_RGBf( tex, field, temp);
    // image comes from texarray on host
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.cols(), tex.rows(), 0, GL_RGB, GL_FLOAT, tex.getPtr());
    glLoadIdentity();
    //Draw a textured quad
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f( 1.0/3.0+slit, -1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0, 1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f( 1.0/3.0+slit, 1.0);
    glEnd();
}

int main()
{
    //Parameter initialisation
    Physical phys; 
    Algorithmic alg;
    Boundary bound;
    init( phys, alg, bound);

    Blueprint bp( phys, bound, alg);
    bp.enable(TL_COUPLING);
    //bp.enable(TL_CURVATURE);
    try{ bp.consistencyCheck();}
    catch( Message& m) {m.display(); bp.display(); return -1;}
    bp.display(cout);

    //construct solver
    DFT_DFT_Solver<2> solver( bp);

    //init solver
    Matrix<double, TL_DFT> ne{ alg.ny, alg.nx, 0.}, ni{ alg.ny, alg.nx, 0.};
    init_gaussian( ne,  0.5,0.5, 0.05, 0.05, 0.1);
    init_gaussian( ni, 0.5,0.5, 0.05, 0.05, 0.1);
    std::array< Matrix<double, TL_DFT>,2> arr{{ ne, ni}};
    try{
        solver.init( arr, TL_POTENTIAL);
    }catch( Message& m){m.display();}

    ////////////////////////////////glfw//////////////////////////////
    {
    int running = GL_TRUE;
    if( !glfwInit()) { cerr << "ERROR: glfw couldn't initialize.\n";}
    unsigned width = 1800, height = 600;
    if( !glfwOpenWindow( width, height,  0,0,0,  0,0,0, GLFW_WINDOW))
    { 
        cerr << "ERROR: glfw couldn't open window!\n";
    }
    glEnable( GL_TEXTURE_2D);
    glfwEnable( GLFW_STICKY_KEYS);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    double t = 3*alg.dt;
    while( running)
    {
        glfwPollEvents();
        if( glfwGetKey( 'S')) 
        {
            do
            {
                glfwWaitEvents();
            } while( !glfwGetKey('R'));
        }
        stringstream str; 
        str << setprecision(2) << fixed;
        str << "ne, ni and phi ... time = "<<t;
        glfwSetWindowTitle( (str.str()).c_str() );

        drawScene( solver, alg.nx, alg.ny);
        glfwSwapBuffers();
#ifdef TL_DEBUG
        glfwWaitEvents();
        if( glfwGetKey('N'))
        {
#else
        for(unsigned i=0; i<N; i++)
        {
#endif
            solver.step();
            t+= alg.dt;
#ifndef TL_DEBUG
        }
#else   
            cout << "Next Step\n";
        }
#endif
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    glfwTerminate();
    }
    //////////////////////////////////////////////////////////////////
    return 0;

}
