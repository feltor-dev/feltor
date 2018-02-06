/*! @file 
 * @brief Root finding method
 * @author Matthias Wiesenberger
 * @date 12.4.2010
 */
#ifndef _NULLSTELLE_
#define _NULLSTELLE_

#include <exception>
#include <math.h>
#include "backend/exceptions.h"
namespace dg{

/*! @brief Exception class, that stores boundaries for 1D root finding
 *
 * @ingroup root
 */
class NoRoot1d: public std::exception
{
  private:
    double x1, x2;
  public:
    /*! @brief construct
     *
     * \param x_min left boundary
     * \param x_max right boundary
     */
    NoRoot1d(double x_min, double x_max): x1(x_min), x2(x_max){}
    /*! @brief display left and right boundary on std::cerr
     */
    void display() const
    {
      std::cerr << "Between " <<x1 << " and "<<x2<<" is no root!\n";
    }
    /**
     * @brief what string
     */
    char const* what() const throw(){ return "There is no root!";}
};

/*! @brief Find a root of a 1d function in given boundaries using bisection
 *
 * @ingroup root
 * It is assumed that a sign change occurs at the root.
 * Function jumps closer to the root by checking the sign. 
 * \tparam UnaryOp unary function operator
 * \param op Function or Functor
 * \param x_min left boundary, contains new left boundary on execution
 * \param x_max right boundary, contains new right boundary on execution
 * \param eps accuracy of the root finding	
 * \return number of used steps to reach the desired accuracy
 * \throw NoRoot1d if no root lies between the given boundaries
 * \throw std::runtime_error if after 60 steps the accuracy wasn't reached
 *
 * \code nullstelle_1D(funk, x_min, x_max, eps); \endcode
 * \note If the root is found exactly the x_min = x_max 
 */
template <typename UnaryOp>           
int bisection1d (UnaryOp& op, double& x_min, double& x_max, const double eps) 
{
    double  mitte;
    double wert_max, wert_mitte, wert_min;
    wert_max=op(x_max);
    wert_min=op(x_min);

    if(wert_max*wert_min>=0) 
        throw NoRoot1d(x_min, x_max);
    
    int j_max = 60;
    for(int j=0; j<j_max; j++)
    {
        wert_mitte = op( mitte =(x_min+x_max)/2.0 );
        if(wert_mitte==0) 			    {x_min=x_max=mitte; return j+3;}
        else if(wert_mitte*wert_max>0) 	x_max = mitte;
        else 				            x_min = mitte;
        if((x_max-x_min)<eps)           return j+3; 
    }
    throw std::runtime_error("Too many steps in root finding!");
}

//@}
}//namespace dg
#endif //_NULLSTELLE_  
  
  
  
