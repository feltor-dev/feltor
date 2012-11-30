#include <iostream>
#include "toefl.h"
#include "dft_dft_solver.h"
#include "blueprint.h"

using namespace std;
using namespace toefl;
    

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
int main()
{

    Physical phys; 
    Algorithmic alg;
    Boundary bound;
    vector<double> para;
    try{ para = read_input( "input.dat"); }
    catch (Message& m) {  m.display(); return -1;}
    phys.d = para[1];
    phys.g_e = phys.g[0] = para[2];
    phys.g[2] = para[3];
    phys.tau[0] = para[4];
    phys.tau[1] = para[22];
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

    alg.h = bound.ly / (double)alg.ny;
    bound.lx = (double)alg.nx * alg.h;
    bound.bc_x = TL_PERIODIC;


    Blueprint bp( phys, bound, alg);
    try{ bp.consistencyCheck();}
    catch( Message& m) {m.display(); bp.display(); return -1;}
    bp.display(cout);
    DFT_DFT_Solver<2> solver( bp);

    Matrix<double, TL_DFT> ne{ alg.ny, alg.nx, 0.}, phi{ alg.ny, alg.nx, 0.};
    init_gaussian( ne, 0.5,0.5, 0.1, 0.1, 0.2);
    init_gaussian( phi, 0.5,0.5, 0.1, 0.1, 0.2);
    std::array< Matrix<double, TL_DFT>,2> arr{{ ne, phi}};
    try{
    solver.init( arr, TL_IONS);
    }catch( Message& m){m.display();}
    ////////////////////////////////glfw//////////////////////////////
    int running = GL_TRUE;
    if( !glfwInit()) { cerr << "ERROR: glfw couldn't initialize.\n";}
    if( !glfwOpenWindow( 600, 600,  0,0,0,  0,0,0, GLFW_WINDOW))
    { 
        cerr << "ERROR: glfw couldn't open window!\n";
    }
    glfwSetWindowTitle( "Gaussian test");
    Texture_RGBf tex( nz, nx);
    glEnable( GL_TEXTURE_2D);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    while( running)
    {
        //generate a texture
        solver.getField( ne, TL_ELECTRONS);
        gentexture_RGBf( tex, m, 1);
        glLoadIdentity();
        glClearColor(0.f, 0.f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);
        // image comes from texarray on host
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.cols(), tex.rows(), 0, GL_RGB, GL_FLOAT, tex.getPtr());
        glLoadIdentity();
        //Draw a textured quad
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0, 1.0);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0, 1.0);
        glEnd();
        glfwSwapBuffers();
        solver.step();///////////<--
        glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    glfwTerminate();
    //cout << setprecision(2)<<fixed;
    //cout << m<<endl;
    //////////////////////////////////////////////////////////////////


}
