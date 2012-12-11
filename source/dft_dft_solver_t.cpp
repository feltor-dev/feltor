#include <iostream>
#include <iomanip>
#include <GL/glfw.h>
#include <sstream>
#include "toefl.h"
#include "dft_dft_solver.h"
#include "drt_dft_solver.h"
#include "blueprint.h"

using namespace std;
using namespace toefl;
    
unsigned N; //initialized by init function
double amp; //
const double slit = 2./500.; //half distance between pictures in units of width
unsigned width = 960, height = 1080; //initial window width & height

void GLFWCALL WindowResize( int w, int h)
{
    glViewport( 0, 0, (GLsizei) w, (GLsizei) h);
    width = w;
    height = h;
}

void init( Blueprint& bp)
{
    Physical& phys = bp.physical();
    Algorithmic& alg = bp.algorithmic();
    Boundary& bound = bp.boundary();
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
    amp = para[10];
    if( para[29])
        bp.enable( TL_GLOBAL);

    alg.h = bound.ly / (double)alg.ny;
    bound.lx = (double)alg.nx * alg.h;
    bound.bc_x = TL_PERIODIC;
}

template< class M>
double abs_max( const M& field)
{
    double temp = 0;
    for( unsigned i=0; i<field.rows(); i++)
        for( unsigned j=0; j<field.cols(); j++)
            if( fabs(field(i,j)) > temp) temp = fabs(field(i,j));
    return temp;
}
    

template<class M>
void loadTexture( const M& field, double max)
{
    static Texture_RGBf tex( field.rows(), field.cols());
    gentexture_RGBf( tex, field, max);
    // image comes from texarray on host
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.cols(), tex.rows(), 0, GL_RGB, GL_FLOAT, tex.getPtr());
}

//todo correct aspect ratio 
template<class M>
void drawScene( const DFT_DFT_Solver<2>& solver)
{
    glClear(GL_COLOR_BUFFER_BIT);
    double scale_y = 1.;
    double max;
    M const * field;
    
    {
    field = &solver.getField( TL_ELECTRONS);
    max = abs_max( *field);
    glLoadIdentity();
    loadTexture( *field, max);
#ifdef TL_DEBUG
    cout <<"max densitiy = "<<max<<endl;
#endif
    //Draw a textured quad
    //upper left
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f( -1.0, +slit);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( -slit, +slit);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( -slit, 1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f( -1.0, 1.0);
    glEnd();
    }
    {
    field = &solver.getField( TL_IONS);
    loadTexture( *field, max);
    glLoadIdentity();
    //upper right
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f( +slit, +slit);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, +slit);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0, 1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f( +slit, 1.0);
    glEnd();
    }
    {
    field = &solver.getField( TL_POTENTIAL); 
    max = abs_max(*field);
    loadTexture( *field, max);
#ifdef TL_DEBUG
    cout <<"max potential = "<<max<<endl;
#endif
    glLoadIdentity();
    //lower left
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( -slit, -1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( -slit, -slit);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0, -slit);
    glEnd();
    }
    //lower right
    //glBegin(GL_QUADS);
    //    glTexCoord2f(0.0f, 0.0f); glVertex2f( +slit, -1.0);
    //    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
    //    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0, 0);
    //    glTexCoord2f(0.0f, 1.0f); glVertex2f(+slit, -slit);
    //glEnd();
        
}

int main()
{
    //Parameter initialisation
    Blueprint bp;
    init( bp);

    try{ bp.consistencyCheck();}
    catch( Message& m) {m.display(); bp.display(); return -1;}
    bp.display(cout);

    //construct solver
    DFT_DFT_Solver<2> solver( bp);
    bp.boundary().bc_x = TL_DST10;
    DRT_DFT_Solver<2> drt_solver( bp);

    //init solver
    const Algorithmic& alg = bp.algorithmic();
    Matrix<double, TL_DFT> ne{ alg.ny, alg.nx, 0.}, ni{ alg.ny, alg.nx, 0.};
    init_gaussian( ne,  0.5,0.5, 0.05, 0.05, amp);
    init_gaussian( ni, 0.5,0.5, 0.05, 0.05, amp);
    std::array< Matrix<double, TL_DFT>,2> arr{{ ne, ni}};
    try{
        solver.init( arr, TL_POTENTIAL);
    }catch( Message& m){m.display();}

    ////////////////////////////////glfw//////////////////////////////
    {
    int running = GL_TRUE;
    if( !glfwInit()) { cerr << "ERROR: glfw couldn't initialize.\n";}
    if( !glfwOpenWindow( width, height,  0,0,0,  0,0,0, GLFW_WINDOW))
    { 
        cerr << "ERROR: glfw couldn't open window!\n";
    }
    glfwSetWindowSizeCallback( WindowResize);

    glEnable( GL_TEXTURE_2D);
    glfwEnable( GLFW_STICKY_KEYS);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glClearColor(0.f, 0.f, 0.f, 0.f);

    double t = 3*alg.dt;
    Timer timer;
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

        drawScene<Matrix<double, TL_DFT>>( solver);
        glfwSwapBuffers();
#ifdef TL_DEBUG
        glfwWaitEvents();
        if( glfwGetKey('N'))
        {
#endif
        timer.tic();
        for(unsigned i=0; i<N; i++)
        {
            solver.step();
            t+= alg.dt;
        }
        timer.toc();
#ifdef TL_DEBUG
            cout << "Next "<<N<<" Steps\n";
        }
#endif
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    glfwTerminate();
    cout << "Average time for one step = "<<timer.diff()/(double)N<<"s\n";
    }
    //////////////////////////////////////////////////////////////////
    return 0;

}
