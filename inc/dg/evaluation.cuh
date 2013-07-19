#ifndef _DG_EVALUATION_
#define _DG_EVALUATION_

#include <cassert> 
#include "grid.cuh"
#include "matrix_traits_thrust.h"
#include "operator_dynamic.h"

//#include "arrvec1d.cuh"
//#include "arrvec2d.cuh"
//#include "dlt.h"

/*! @file function discretization routines
  */
namespace dg
{


///@addtogroup evaluation
///@{



///**
// * @brief Evaluate a function on gaussian abscissas
// *
// * Evaluates f(x) on the intervall (a,b)
// * @tparam Function Model of Unary Function
// * @tparam n number of Legendre nodes per cell
// * @param f The function to evaluate
// * @param a Left border
// * @param b Right border
// * @param num_int Number of intervalls between a and b 
// *
// * @return  A DG Host Vector with values
// */
//template< class Function, size_t n>
//ArrVec1d< double, n> evaluate( Function& f, double a, double b, unsigned num_int)
//{
//    assert( b > a && num_int > 0) ;
//    ArrVec1d< double, n> v(num_int);
//    const double h = (b-a)/2./(double)num_int;
//    /* x = (b-a)/2N x' +a  maps the function to [0;2N]
//      then x' goes through 1,3,5,...,2N-1
//     */
//    double xp=1.;
//    for( unsigned i=0; i<num_int; i++)
//    {
//        for( unsigned j=0; j<n; j++)
//            v(i,j) = f( h*(xp + DLT<n>::abscissa[j])+a);
//        xp+=2.;
//    }
//    return v;
//}

/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the intervall (a,b)
 * @tparam Function Model of Unary Function
 * @param f The function to evaluate
 * @param g The grid on which to evaluate f
 *
 * @return  A DG Host Vector with values
 */
template< class Function>
thrust::host_vector<double> evaluate( Function f, const Grid1d<double>& g)
{
    thrust::host_vector<double> abs = create::abscissas( g);
    for( unsigned i=0; i<g.size(); i++)
        abs[i] = f( abs[i]);
    return abs;
};
///@cond
thrust::host_vector<double> evaluate( double (f)(double), const Grid1d<double>& g)
{
    thrust::host_vector<double> v = evaluate<double (double)>( f, g);
    return v;
};
///@endcond





///**
// * @brief Evaluate a function on gaussian abscissas
// *
// * Evaluates f(x) on the field (x0,y0), (x1,y1)
// * @tparam Function Model of Binary Function
// * @param f The function to evaluate: f = f(x,y)
// * @param x0 x-position of lower left corner
// * @param x1 x-position of upper right corner
// * @param y0 y-position of lower left corner
// * @param y1 y-position of upper right corner
// * @param Nx Number of intervalls between x0 and x1 
// * @param Ny Number of intervalls between y0 and y1 
// *
// * @return  A DG Host Vector with values
// */
//template< class BinaryOp, size_t n>
//ArrVec2d< double, n> evaluate( BinaryOp& f, double x0, double x1, double y0, double y1, unsigned Nx, unsigned Ny)
//{
//    assert( x1 > x0 && y1 > y0);
//    assert( Nx > 0  && Ny > 0);
//    ArrVec2d< double,n > v( Ny, Nx);
//    const double hx = (x1-x0)/2./(double)Nx;
//    const double hy = (y1-y0)/2./(double)Ny;
//    double x , y;
//    double xp=1., yp = 1.;
//    //access of v in dg order
//    for( unsigned i=0; i<Ny; i++)
//    {
//        xp = 1.;
//        for( unsigned j=0; j<Nx; j++)
//        {
//            for( unsigned k=0; k<n; k++) //y-index
//            {
//                y = y0 + hy*( yp + DLT<n>::abscissa[k]) ;
//                for( unsigned l=0; l<n; l++) //x-index
//                {
//                    x  = x0 + hx*( xp + DLT<n>::abscissa[l]);
//                    v(i,j,k,l) = f( x, y);
//                }
//            }
//            xp+=2.;
//        }
//        yp+=2.;
//    }
//    return v;
//}

/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the given grid
 * @tparam Function Model of Binary Function
 * @tparam n number of Legendre nodes per cell per dimension
 * @param f The function to evaluate: f = f(x,y)
 * @param g The 2d grid on which to evaluate f
 *
 * @return  A DG Host Vector with values
 * @note Copies the binary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class BinaryOp>
thrust::host_vector<double> evaluate( BinaryOp f, const Grid<double>& g)
{
    unsigned n= g.n();
    Grid1d<double> gx( g.x0(), g.x1(), n, g.Nx());
    Grid1d<double> gy( g.y0(), g.y1(), n, g.Ny());
    thrust::host_vector<double> absx = create::abscissas( gx);
    thrust::host_vector<double> absy = create::abscissas( gy);

    thrust::host_vector<double> v( g.size());
    for( unsigned i=0; i<gy.N(); i++)
        for( unsigned j=0; j<gx.N(); j++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    v[ i*g.Nx()*n*n + j*n*n + k*n + l] = f( absx[j*n+l], absy[i*n+k]);
    return v;
};
///@cond
thrust::host_vector<double> evaluate( double(f)(double, double), const Grid<double>& g)
{
    //return evaluate<double(&)(double, double), n>( f, g );
    return evaluate<double(double, double)>( f, g);
};
///@endcond



///**
// * @brief Evaluate and dlt transform a function 
// *
// * Evaluates f(x) on the intervall (a,b)
// * @tparam Function Model of Unary Function
// * @param f The function to evaluate: f = f(x)
// * @param a Left border
// * @param b Right border
// * @param num_int Number of intervalls between a and b 
// *
// * @return  A DG Host Vector with dlt transformed values
// */
//template< class Function, size_t n>
//ArrVec1d<double, n> expand( Function& f, double a, double b, unsigned num_int)
//{
//    ArrVec1d<double, n> v = evaluate<Function,n> ( f, a, b, num_int);
//    //std::cout << "Evaluation: 1D\n" << v<<std::endl;
//    //multiply elements by forward
//    double temp[n];
//    for( unsigned k=0; k<num_int; k++)
//    {
//        for(unsigned i=0; i<n; i++)
//        {
//            temp[i] = 0;
//            for( unsigned j=0; j<n; j++)
//                temp[i] += DLT<n>::forward[i][j]*v(k,j);
//        }
//        for( unsigned j=0; j<n; j++)
//            v(k,j) = temp[j];
//    }
//    return v;
//}
/**
 * @brief Evaluate and dlt transform a function 
 *
 * Evaluates f(x) on the given grid
 * @tparam Function Model of Unary Function
 * @param f The function to evaluate: f = f(x)
 * @param g The grid on which to evaluate f
 *
 * @return  A DG Host Vector with dlt transformed values
 */
template< class Function>
thrust::host_vector<double> expand( Function f, const Grid1d<double>& g)
{
    thrust::host_vector<double> v = evaluate( f, g);
    Operator<double> forward( g.dlt().forward());
    double temp[g.n()];
    for( unsigned k=0; k<g.N(); k++)
    {
        for(unsigned i=0; i<g.n(); i++)
        {
            temp[i] = 0;
            for( unsigned j=0; j<g.n(); j++)
                temp[i] += forward(i,j)*v[k*g.n()+j];
        }
        for( unsigned j=0; j<g.n(); j++)
            v[k*g.n()+j] = temp[j];
    }
    return v;
};
///@cond
thrust::host_vector<double> expand( double(f)(double), const Grid1d<double>& g)
{
    return expand<double(double)>( f, g);
};

///@endcond




///**
// * @brief Evaluate and dlt transform a function
// *
// * Evaluates and dlt-transforms f(x) on the field (x0,y0), (x1,y1)
// * @tparam BinaryOp Model of Binary Function
// * @tparam n number of Legendre nodes per cell per dimension
// * @param f The function to evaluate: f = f(x,y)
// * @param x0 x-position of lower left corner
// * @param x1 x-position of upper right corner
// * @param y0 y-position of lower left corner
// * @param y1 y-position of upper right corner
// * @param Nx Number of intervalls between x0 and x1 
// * @param Ny Number of intervalls between y0 and y1 
// *
// * @return  A DG Host Vector with dlt transformed values
// */
//template< class BinaryOp, size_t n>
//ArrVec2d< double, n> expand( BinaryOp& f, double x0, double x1, double y0, double y1, unsigned Nx, unsigned Ny)
//{
//    ArrVec2d<double, n> v = evaluate<BinaryOp,n> ( f, x0, x1, y0, y1, Nx, Ny);
//    //std::cout << "Evaluation: 2D\n" << v<<std::endl;
//    double temp[n][n];
//    //DLT each dg-Box 
//    for( unsigned i=0; i<Ny; i++)
//        for( unsigned j=0; j<Nx; j++)
//        {
//            //first transform each row
//            for( unsigned k=0; k<n; k++) 
//                for( unsigned l=0; l<n; l++)
//                {
//                    //multiply forward-matrix with each row k
//                    temp[k][l] = 0;
//                    for(  unsigned ll=0; ll<n; ll++)
//                        temp[k][l] += DLT<n>::forward[l][ll]*v( i,j,k,ll);
//                }
//            //then transform each col
//            for( unsigned k=0; k<n; k++) 
//                for( unsigned l=0; l<n; l++)
//                {
//                    //multiply forward-matrix with each col 
//                    v(i,j,k,l) = 0;
//                    for(  unsigned kk=0; kk<n; kk++)
//                        v(i,j,k,l) += DLT<n>::forward[k][kk]*temp[kk][l];
//                }
//        }
//    return v;
//}

/**
 * @brief Evaluate and dlt transform a function
 *
 * Evaluates and dlt-transforms f(x) on the given grid
 * @tparam Function Model of Binary Function
 * @tparam n number of Legendre nodes per cell per dimension
 * @param f The function to evaluate: f = f(x,y)
 * @param g The 2d grid on which to evaluate f
 *
 * @return  A DG Host Vector with values
 * @note Copies the binary Operator. This function is meant for small function objects.
 */
template< class BinaryOp>
thrust::host_vector<double> expand( BinaryOp f, const Grid<double>& g)
{
    thrust::host_vector<double> v = evaluate( f, g);
    unsigned n = g.n();
    Operator<double> forward( g.dlt().forward());
    double temp[n][n];
    //DLT each dg-Box 
    for( unsigned i=0; i<g.Ny(); i++)
        for( unsigned j=0; j<g.Nx(); j++)
        {
            //first transform each row
            for( unsigned k=0; k<n; k++) 
                for( unsigned l=0; l<n; l++)
                {
                    //multiply forward-matrix with each row k
                    temp[k][l] = 0;
                    for(  unsigned ll=0; ll<n; ll++)
                        temp[k][l] += forward(l,ll)*v[ i*n*n*g.Nx() + j*n*n + k*n + ll];
                }
            //then transform each col
            for( unsigned k=0; k<n; k++) 
                for( unsigned l=0; l<n; l++)
                {
                    //multiply forward-matrix with each col 
                    v[i*n*n*g.Nx() + j*n*n + k*n + l] = 0;
                    for(  unsigned kk=0; kk<n; kk++)
                        v[i*n*n*g.Nx() + j*n*n + k*n + l] += forward(k,kk)*temp[kk][l];
                }
        }

    return v;
};

///@cond
thrust::host_vector<double> expand( double(f)(double, double), const Grid<double>& g)
{
    return expand<double(double, double)>( f, g);
};

///@endcond


///@}
}//namespace dg

#endif //_DG_EVALUATION
