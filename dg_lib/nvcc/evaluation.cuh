#ifndef _DG_EVALUATION_
#define _DG_EVALUATION_

#include <cassert> 
#include "grid.cuh"

#include "arrvec1d.cuh"
#include "arrvec2d.cuh"
#include "dlt.h"

/*! @file function discretization routines
  */
namespace dg
{


///@addtogroup evaluation
///@{



/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the intervall (a,b)
 * @tparam Function Model of Unary Function
 * @tparam n number of Legendre nodes per cell
 * @param f The function to evaluate
 * @param a Left border
 * @param b Right border
 * @param num_int Number of intervalls between a and b 
 *
 * @return  A DG Host Vector with values
 */
template< class Function, size_t n>
ArrVec1d< double, n> evaluate( Function& f, double a, double b, unsigned num_int)
{
    assert( b > a && num_int > 0) ;
    ArrVec1d< double, n> v(num_int);
    const double h = (b-a)/2./(double)num_int;
    /* x = (b-a)/2N x' +a  maps the function to [0;2N]
      then x' goes through 1,3,5,...,2N-1
     */
    double xp=1.;
    for( unsigned i=0; i<num_int; i++)
    {
        for( unsigned j=0; j<n; j++)
            v(i,j) = f( h*(xp + DLT<n>::abscissa[j])+a);
        xp+=2.;
    }
    return v;
}
/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the intervall (a,b)
 * @tparam Function Model of Unary Function
 * @tparam n number of Legendre nodes per cell
 * @param f The function to evaluate
 * @param g The grid on which to evaluate f
 *
 * @return  A DG Host Vector with values
 */
template< class Function, size_t n>
thrust::host_vector<double> evaluate( Function& f, const Grid1d<double,n>& g)
{
    return (evaluate<Function, n>( f, g.x0(), g.x1(), g.Nx())).data();
};
///@cond
template< size_t n>
thrust::host_vector<double> evaluate( double (*f)(double), const Grid1d<double,n>& g)
{
    return (evaluate<double(&)(double), n>( *f, g.x0(), g.x1(), g.N())).data();
};
///@endcond





/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the field (x0,y0), (x1,y1)
 * @tparam Function Model of Binary Function
 * @tparam n number of Legendre nodes per cell per dimension
 * @param f The function to evaluate: f = f(x,y)
 * @param x0 x-position of lower left corner
 * @param x1 x-position of upper right corner
 * @param y0 y-position of lower left corner
 * @param y1 y-position of upper right corner
 * @param Nx Number of intervalls between x0 and x1 
 * @param Ny Number of intervalls between y0 and y1 
 *
 * @return  A DG Host Vector with values
 */
template< class BinaryOp, size_t n>
ArrVec2d< double, n> evaluate( BinaryOp& f, double x0, double x1, double y0, double y1, unsigned Nx, unsigned Ny)
{
    assert( x1 > x0 && y1 > y0);
    assert( Nx > 0  && Ny > 0);
    ArrVec2d< double,n > v( Ny, Nx);
    const double hx = (x1-x0)/2./(double)Nx;
    const double hy = (y1-y0)/2./(double)Ny;
    double x , y;
    double xp=1., yp = 1.;
    //access of v in dg order
    for( unsigned i=0; i<Ny; i++)
    {
        xp = 1.;
        for( unsigned j=0; j<Nx; j++)
        {
            for( unsigned k=0; k<n; k++) //y-index
            {
                y = y0 + hy*( yp + DLT<n>::abscissa[k]) ;
                for( unsigned l=0; l<n; l++) //x-index
                {
                    x  = x0 + hx*( xp + DLT<n>::abscissa[l]);
                    v(i,j,k,l) = f( x, y);
                }
            }
            xp+=2.;
        }
        yp+=2.;
    }
    return v;
}

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
 */
template< class BinaryOp, size_t n>
thrust::host_vector<double> evaluate( BinaryOp& f, const Grid<double,n>& g)
{
    return (evaluate<BinaryOp, n>( f, g.x0(), g.x1(), g.y0(), g.y1(), g.Nx(), g.Ny() )).data();
};
///@cond
template< size_t n>
thrust::host_vector<double> evaluate( double(f)(double, double), const Grid<double,n>& g)
{
    //return evaluate<double(&)(double, double), n>( f, g );
    return (evaluate<double(&)(double, double), n>( *f, g.x0(), g.x1(), g.y0(), g.y1(), g.Nx(), g.Ny() )).data();
};
///@endcond



/**
 * @brief Evaluate and dlt transform a function 
 *
 * Evaluates f(x) on the intervall (a,b)
 * @tparam Function Model of Unary Function
 * @param f The function to evaluate: f = f(x)
 * @param a Left border
 * @param b Right border
 * @param num_int Number of intervalls between a and b 
 *
 * @return  A DG Host Vector with dlt transformed values
 */
template< class Function, size_t n>
ArrVec1d<double, n> expand( Function& f, double a, double b, unsigned num_int)
{
    ArrVec1d<double, n> v = evaluate<Function,n> ( f, a, b, num_int);
    //std::cout << "Evaluation: 1D\n" << v<<std::endl;
    //multiply elements by forward
    double temp[n];
    for( unsigned k=0; k<num_int; k++)
    {
        for(unsigned i=0; i<n; i++)
        {
            temp[i] = 0;
            for( unsigned j=0; j<n; j++)
                temp[i] += DLT<n>::forward[i][j]*v(k,j);
        }
        for( unsigned j=0; j<n; j++)
            v(k,j) = temp[j];
    }
    return v;
}
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
template< class Function, size_t n>
thrust::host_vector<double> expand( Function& f, const Grid1d<double,n>& g)
{
    return (expand<Function, n>( f, g.x0(), g.x1(), g.Nx())).data();
};
///@cond
template< size_t n>
thrust::host_vector<double> expand( double(*f)(double), const Grid1d<double,n>& g)
{
    return (expand<double(&)(double), n>( *f, g.x0(), g.x1(), g.N())).data();
};

///@endcond




/**
 * @brief Evaluate and dlt transform a function
 *
 * Evaluates and dlt-transforms f(x) on the field (x0,y0), (x1,y1)
 * @tparam BinaryOp Model of Binary Function
 * @tparam n number of Legendre nodes per cell per dimension
 * @param f The function to evaluate: f = f(x,y)
 * @param x0 x-position of lower left corner
 * @param x1 x-position of upper right corner
 * @param y0 y-position of lower left corner
 * @param y1 y-position of upper right corner
 * @param Nx Number of intervalls between x0 and x1 
 * @param Ny Number of intervalls between y0 and y1 
 *
 * @return  A DG Host Vector with dlt transformed values
 */
template< class BinaryOp, size_t n>
ArrVec2d< double, n> expand( BinaryOp& f, double x0, double x1, double y0, double y1, unsigned Nx, unsigned Ny)
{
    ArrVec2d<double, n> v = evaluate<BinaryOp,n> ( f, x0, x1, y0, y1, Nx, Ny);
    //std::cout << "Evaluation: 2D\n" << v<<std::endl;
    double temp[n][n];
    //DLT each dg-Box 
    for( unsigned i=0; i<Ny; i++)
        for( unsigned j=0; j<Nx; j++)
        {
            //first transform each row
            for( unsigned k=0; k<n; k++) 
                for( unsigned l=0; l<n; l++)
                {
                    //multiply forward-matrix with each row k
                    temp[k][l] = 0;
                    for(  unsigned ll=0; ll<n; ll++)
                        temp[k][l] += DLT<n>::forward[l][ll]*v( i,j,k,ll);
                }
            //then transform each col
            for( unsigned k=0; k<n; k++) 
                for( unsigned l=0; l<n; l++)
                {
                    //multiply forward-matrix with each col 
                    v(i,j,k,l) = 0;
                    for(  unsigned kk=0; kk<n; kk++)
                        v(i,j,k,l) += DLT<n>::forward[k][kk]*temp[kk][l];
                }
        }
    return v;
}

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
 */
template< class Function, size_t n>
thrust::host_vector<double> expand( Function& f, const Grid<double,n>& g)
{
    return (expand<Function, n>( f, g.x0(), g.x1(), g.y0(), g.y1(), g.Nx(), g.Ny() )).data();
};

///@cond
template< size_t n>
thrust::host_vector<double> expand( double(f)(double, double), const Grid<double,n>& g)
{
    return (expand<double(&)(double, double), n>( *f, g.x0(), g.x1(), g.y0(), g.y1(), g.Nx(), g.Ny() )).data();
};

///@endcond


    /*
    //not tested in practical use yet
template< class T, size_t n>
thrust::host_vector<T> positions( T a, T b, unsigned num_int)
{
    assert( b > a && num_int > 0) ;
    thrust::host_vector< T> v(n*num_int);
    const double h = (b-a)/2./(double)num_int;
    // x = (b-a)/2N x' +a  maps the function to [0;2N]
    //  then x' goes through 1,3,5,...,2N-1
     
    double xp=1.;
    for( unsigned i=0; i<num_int; i++)
    {
        for( unsigned j=0; j<n; j++)
            v[i*n+j] = a + h*(xp + DLT<n>::abscissa[j]);
        xp+=2.;
    }
    return v;
}

    //not tested in practical use
template< class Function, class Vector>
Vector evaluate( Function& f, Vector& grid)
{
    Vector v(grid);
    thrust::transform( grid.begin(), grid.end(), v.begin(), f);
    return v;
}

    //not tested in practical use
template< class Vector>
Vector evaluate( double(f)(double), Vector& grid)
{
    return evaluate< double(&)(double, double), Vector>( f, grid);
}

    //not tested in practical use
template< class Function, class Vector>
Vector evaluate( Function& f, Vector& gridx, Vector& gridy)
{
    Vector v(gridx.size()*gridy.size());
    for( unsigned i=0; i<gridy.size(); i++)
        for( unsigned j=0; j<gridx.size(); j++)
            v[i*gridx.size() + j] = f( gridx[j], gridy[i]);
    return v;
}
*/
///@}
}//namespace dg

#endif //_DG_EVALUATION
