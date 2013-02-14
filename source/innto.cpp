#include <iostream>
#include <iomanip>
#include <GL/glfw.h>
#include <sstream>
#include <omp.h>

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

Blueprint read( char const * file)
{
    Blueprint bp;
    Physical& phys = bp.physical();
    Algorithmic& alg = bp.algorithmic();
    Boundary& bound = bp.boundary();
    vector<double> para;
    try{ para = read_input( file); }
    catch (Message& m) 
    {  
        m.display(); 
        throw m;
    }
    alg.nx = para[1];
    alg.ny = para[2];
    alg.dt = para[3];

    bound.ly = para[4];
    switch( (unsigned)para[5])
    {
        case( 0): bound.bc_x = TL_PERIODIC; break;
        case( 1): bound.bc_x = TL_DST10; break;
        case( 2): bound.bc_x = TL_DST01; break;
    }
    if( para[6])
        bp.enable( TL_MHW);

    phys.d = para[7];
    phys.nu = para[8];
    phys.kappa = para[9];

    amp = para[10];
    phys.g_e = phys.g[0] = para[11];
    phys.tau[0] = para[12];
    if( para[13])
        bp.enable( TL_IMPURITY);
    imp_amp = para[14];
    phys.g[1] = para[15];
    phys.a[1] = para[16];
    phys.mu[1] = para[17];
    phys.tau[1] = para[18];

    phys.a[0] = 1. -phys.a[1];
    phys.g[0] = (phys.g_e - phys.a[1] * phys.g[1])/(1.-phys.a[1]);
    phys.mu[0] = 1.0;//single charged ions

    N = para[19];
    omp_set_num_threads( para[20]);
    cout<< "With "<<omp_get_max_threads()<<" threads\n";

    alg.h = bound.ly / (double)alg.ny;
    bound.lx = (double)alg.nx * alg.h;
    return bp;
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

int main( int argc, char* argv[])
{
    //Parameter initialisation
    Blueprint bp_mod;
    if( argc == 1)
    {
        bp_mod = read("input.txt");
    }
    else if( argc == 2)
    {
        bp_mod = read( argv[1]);
    }
    else
    {
        cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const Blueprint bp = bp_mod;
    field_ratio = bp.boundary().lx/bp.boundary().ly;
    

    bp.display(cout);
    //construct solvers 
    DFT_DFT_Solver<2> solver2( bp);
    DFT_DFT_Solver<3> solver3( bp);
    if( bp.boundary().bc_x == TL_PERIODIC)
        bp_mod.boundary().bc_x = TL_DST10;
    DRT_DFT_Solver<2> drt_solver2( bp_mod);
    DRT_DFT_Solver<3> drt_solver3( bp_mod);

    const Algorithmic& alg = bp.algorithmic();
    Matrix<double, TL_DFT> ne{ alg.ny, alg.nx, 0.}, nz{ ne}, phi{ ne};
    // place some gaussian blobs in the field
    try{
        //init_gaussian( ne, 0.1,0.2, 10./128./field_ratio, 10./128., amp);
        //init_gaussian( ne, 0.1,0.4, 10./128./field_ratio, 10./128., -amp);
        init_gaussian( ne, 0.8,0.4, 10./128./field_ratio, 10./128., amp);
        //init_gaussian( ne, 0.1,0.8, 10./128./field_ratio, 10./128., -amp);
        //init_gaussian( ni, 0.1,0.5, 0.05/field_ratio, 0.05, amp);
        if( bp.isEnabled( TL_IMPURITY))
        {
            init_gaussian( nz, 0.8,0.4, 0.05/field_ratio, 0.05, -imp_amp);
            //init_gaussian_column( nz, 0.2, 0.05/field_ratio, imp_amp);
        }
        std::array< Matrix<double, TL_DFT>,2> arr2{{ ne, phi}};
        std::array< Matrix<double, TL_DFT>,3> arr3{{ ne, nz, phi}};
        Matrix<double, TL_DRT_DFT> ne_{ alg.ny, alg.nx, 0.}, nz_{ ne_}, phi_{ ne_};
        init_gaussian( ne_, 0.5,0.5, 0.05/field_ratio, 0.05, amp);
        //init_gaussian( ne_, 0.2,0.2, 0.05/field_ratio, 0.05, -amp);
        //init_gaussian( ne_, 0.6,0.6, 0.05/field_ratio, 0.05, -amp);
        //init_gaussian( ni_, 0.5,0.5, 0.05/field_ratio, 0.05, amp);
        if( bp.isEnabled( TL_IMPURITY))
        {
            init_gaussian( nz_, 0.5,0.5, 0.05/field_ratio, 0.05, -imp_amp);
            //init_gaussian( nz_, 0.2,0.2, 0.05/field_ratio, 0.05, -imp_amp);
            //init_gaussian( nz_, 0.6,0.6, 0.05/field_ratio, 0.05, -imp_amp);
        }
        std::array< Matrix<double, TL_DRT_DFT>,2> arr2_{{ ne_, phi_}};
        std::array< Matrix<double, TL_DRT_DFT>,3> arr3_{{ ne_, nz_, phi_}};
        //now set the field to be computed
        if( !bp.isEnabled( TL_IMPURITY))
        {
            if( bp.boundary().bc_x == TL_PERIODIC)
                solver2.init( arr2, TL_IONS);
            else
                drt_solver2.init( arr2_, TL_IONS);
        }
        else
        {
            if( bp.boundary().bc_x == TL_PERIODIC)
                solver3.init( arr3, TL_IONS);
            else
                drt_solver3.init( arr3_, TL_IONS);
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
    Timer overhead;
    cout<< "HIT ESC to terminate program \n"
        << "HIT S   to stop simulation \n"
        << "HIT R   to continue simulation!\n";
    while( running)
    {
        overhead.tic();
        //ask if simulation shall be stopped
        glfwPollEvents();
        if( glfwGetKey( 'S')) 
        {
            do
            {
                glfwWaitEvents();
            } while( !glfwGetKey('R') && 
                     !glfwGetKey( GLFW_KEY_ESC) && 
                      glfwGetWindowParam( GLFW_OPENED) );
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
        overhead.toc();
    }
    glfwTerminate();
    cout << "Average time for one step = "<<timer.diff()/(double)N<<"s\n";
    cout << "Overhead for visualisation, etc. per step = "<<(overhead.diff()-timer.diff())/(double)N<<"s\n";
    }
    //////////////////////////////////////////////////////////////////
    fftw_cleanup();
    return 0;

}
