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
double amp, imp_amp; //
const double slit = 2./500.; //half distance between pictures in units of width
double field_ratio;
unsigned width = 960, height = 1080; //initial window width & height
stringstream window_str;  //window name

void GLFWCALL WindowResize( int w, int h)
{
    // map coordinates to the whole window
    double win_ratio = (double)w/(double)h;
    GLint ww = (win_ratio<field_ratio) ? w : h*field_ratio ;
    GLint hh = (win_ratio<field_ratio) ? w/field_ratio : h;
    glViewport( 0, 0, (GLsizei) ww, hh);
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
    phys.g[0] = (phys.g_e - phys.a[1] * phys.g[1])/(1.-phys.a[1]);
    phys.mu[0] = 1.0;//single charged ions

    bound.ly = para[12];
    alg.nx = para[13];
    alg.ny = para[14];
    alg.dt = para[15];
    N = para[16];
    amp = para[10];
    imp_amp = para[11];
    if( para[29])
        bp.enable( TL_GLOBAL);
    if( para[30])
        bp.enable( TL_IMPURITY);

    alg.h = bound.ly / (double)alg.ny;
    bound.lx = (double)alg.nx * alg.h;
    switch( (unsigned)para[21])
    {
        case( 0): bound.bc_x = TL_PERIODIC; break;
        case( 1): bound.bc_x = TL_DST10; break;
        case( 2): bound.bc_x = TL_DST01; break;
    }

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

// The solver has to have the getField( target) function returing M
// and the blueprint() function
template<class M, class Solver>
void drawScene( const Solver& solver)
{
    glClear(GL_COLOR_BUFFER_BIT);
    double max;
    M const * field;
    
    { //draw electrons
    field = &solver.getField( TL_ELECTRONS);
    max = abs_max( *field);
    glLoadIdentity();
    loadTexture( *field, max);
    window_str << scientific;
    window_str <<"ne / "<<max<<"\t";
    //Draw a textured quad
    //upper left
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f( -1.0, +slit);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( -slit, +slit);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( -slit, 1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f( -1.0, 1.0);
    glEnd();
    }

    { //draw Ions
    field = &solver.getField( TL_IONS);
    loadTexture( *field, max);
    window_str <<" ni / "<<max<<"\t";
    glLoadIdentity();
    //upper right
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f( +slit, +slit);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, +slit);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0, 1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f( +slit, 1.0);
    glEnd();
    }

    if( solver.blueprint().isEnabled( TL_IMPURITY))
    {
        field = &solver.getField( TL_IMPURITIES); 
        max = abs_max(*field);
        loadTexture( *field, max);
        window_str <<" nz / "<<max<<"\t";
        glLoadIdentity();
        //lower left
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( -slit, -1.0);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( -slit, -slit);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0, -slit);
        glEnd();
    }

    { //draw potential
    field = &solver.getField( TL_POTENTIAL); 
    max = abs_max(*field);
    loadTexture( *field, max);
    window_str <<" phi / "<<max<<"\t";
    glLoadIdentity();
    //lower right
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f( +slit, -1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0, 0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(+slit, -slit);
    glEnd();
    }
        
}

int main()
{
    //Parameter initialisation
    Blueprint bp_mod;
    init( bp_mod);
    const Blueprint bp{ bp_mod};
    field_ratio = bp.boundary().lx/bp.boundary().ly;

    bp.display(cout);
    //construct solvers 
    DFT_DFT_Solver<2> solver2( bp);
    DFT_DFT_Solver<3> solver3( bp);
    bp_mod.boundary().bc_x = TL_DST10;
    DRT_DFT_Solver<2> drt_solver2( bp_mod);
    DRT_DFT_Solver<3> drt_solver3( bp_mod);

    //init solver
    const Algorithmic& alg = bp.algorithmic();
    Matrix<double, TL_DFT> ne{ alg.ny, alg.nx, 0.}, ni{ ne}, nz{ ne};
    try{
        init_gaussian( ne, 0.5,0.5, 0.05/field_ratio, 0.05, amp);
        init_gaussian( ni, 0.5,0.5, 0.05/field_ratio, 0.05, amp);
        if( bp.isEnabled( TL_IMPURITY))
            init_gaussian( nz, 0.5,0.5, 0.05/field_ratio, 0.05, imp_amp);
        std::array< Matrix<double, TL_DFT>,2> arr2{{ ne, ni}};
        std::array< Matrix<double, TL_DFT>,3> arr3{{ ne, ni, nz}};
        Matrix<double, TL_DRT_DFT> ne_{ alg.ny, alg.nx, 0.}, ni_{ ne_}, nz_{ ne_};
        init_gaussian( ne_, 0.5,0.5, 0.05/field_ratio, 0.05, amp);
        init_gaussian( ni_, 0.5,0.5, 0.05/field_ratio, 0.05, amp);
        if( bp.isEnabled( TL_IMPURITY))
            init_gaussian( nz_, 0.5,0.5, 0.05/field_ratio, 0.05, imp_amp);
        std::array< Matrix<double, TL_DRT_DFT>,2> arr2_{{ ne_, ni_}};
        std::array< Matrix<double, TL_DRT_DFT>,3> arr3_{{ ne_, ni_, nz_}};
        if( !bp.isEnabled( TL_IMPURITY))
        {
            if( bp.boundary().bc_x == TL_PERIODIC)
                solver2.init( arr2, TL_POTENTIAL);
            else
                drt_solver2.init( arr2_, TL_POTENTIAL);
        }
        else
        {
            if( bp.boundary().bc_x == TL_PERIODIC)
                solver3.init( arr3, TL_POTENTIAL);
            else
                drt_solver3.init( arr3_, TL_POTENTIAL);
        }
    }catch( Message& m){m.display();}

    ////////////////////////////////glfw//////////////////////////////
    {
    int running = GL_TRUE;
    if( !glfwInit()) { cerr << "ERROR: glfw couldn't initialize.\n";}

    height = width/field_ratio;
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
        
        //draw scene
        if( !bp.isEnabled( TL_IMPURITY))
        {
            if( bp.boundary().bc_x == TL_PERIODIC)
                drawScene<Matrix<double, TL_DFT>>( solver2);
            else
                drawScene<Matrix<double, TL_DRT_DFT>>( drt_solver2);
        }
        else
        {
            if( bp.boundary().bc_x == TL_PERIODIC)
                drawScene<Matrix<double, TL_DFT>>( solver3);
            else
                drawScene<Matrix<double, TL_DRT_DFT>>( drt_solver3);
        }
        window_str << setprecision(2) << fixed;
        window_str << " &&   time = "<<t;
        glfwSetWindowTitle( (window_str.str()).c_str() );
        window_str.str("");
        glfwSwapBuffers();
#ifdef TL_DEBUG
        glfwWaitEvents();
        if( glfwGetKey('N'))
        {
#endif
        timer.tic();
        for(unsigned i=0; i<N; i++)
        {
            if( !bp.isEnabled( TL_IMPURITY))
            {
                if( bp.boundary().bc_x == TL_PERIODIC)
                    solver2.step( );
                else
                    drt_solver2.step();
            }
            else
            {
                if( bp.boundary().bc_x == TL_PERIODIC)
                    solver3.step( );
                else
                    drt_solver3.step( );
            }
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
