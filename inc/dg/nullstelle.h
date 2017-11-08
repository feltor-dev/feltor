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
 * @ingroup numerical0
 */
class KeineNST_1D: public std::exception
{
  private:
    double x1, x2;
  public:
    /*! @brief construct
     *
     * \param x_min left boundary
     * \param x_max right boundary
     */
    KeineNST_1D(double x_min, double x_max): x1(x_min), x2(x_max){}
    /*! @brief display on std::cerr
     *
     * meldet %Fehler und gibt linke und rechte Grenze aus
     */
    void anzeigen() const
    {
      std::cerr << "Between " <<x1 << " and "<<x2<<" is no root!\n";
    }
    /**
     * @brief what string
     *
     * @return 
     */
    char const* what() const throw(){ return "Failed to find root!";}
};

/*! @brief Find a root of a 1d function in given boundaries
 *
 * @ingroup root
 * It is assumed that a sign change occurs at the root.
 * Function jumps closer to the root by checking the sign. 
 * \tparam UnaryOp unary function operator
 * \param funktion Function or Functor
 * \param x_min left boundary, contains new left boundary on execution
 * \param x_max right boundary, contains new right boundary on execution
 * \param aufloesung accuracy of the root finding	
 * \return number of used steps to reach the desired accuracy
 * \throw KeineNST_1D if no root lies between the given boundaries
 * \throw std::runtime_error if after 60 steps the accuracy wasn't reached
 *
 * \code nullstelle_1D(funk, x_min, x_max, aufloesung); \endcode
 * \note If the root is found exactly the x_min = x_max 
 */
template <typename UnaryOp>           
int bisection1d (UnaryOp& funktion, double& x_min, double& x_max, const double aufloesung) 
{
    double  mitte;
    double wert_oben, wert_mitte, wert_unten;
    wert_oben=funktion(x_max);
    wert_unten=funktion(x_min);

    if(wert_oben*wert_unten>=0) 
        throw KeineNST_1D(x_min, x_max);
    
    int j_max = 60;
    for(int j=0; j<j_max; j++)
    {
        wert_mitte = funktion( mitte=(x_min+x_max)/2.0 );
        if(wert_mitte==0) 			        {x_min=x_max=mitte; return j+3;}
        else if(wert_mitte*wert_oben>0) 	x_max = mitte;
        else 				                x_min = mitte;
        if((x_max-x_min)<aufloesung)        return j+3; 
    }
    throw std::runtime_error("Zu viele Schritte bei Nullstellensuche! evtl. j_max aendern");
}

      
//@}
}//namespace dg
#endif //_NULLSTELLE_  
  
  
  
