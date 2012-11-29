/*!
 * \file
 * \brief Implementation of the arakawa scheme
 * \author Matthias Wiesenberger
 * \email Matthias.Wiesenberger@uibk.ac.at
 * 
 */
#ifndef _TL_ARAKAWA_
#define _TL_ARAKAWA_

#include "quadmat.h"

namespace toefl{


template< class M>
static double interior( const size_t i0, const size_t j0, const M& lhs, const M& rhs);
template< class M>
static double boundary( const size_t i0, const size_t j0, const M& lhs, const M& rhs);

/*! @brief Implements the arakawa scheme 
 *
 * The 2D Jacobian is defined as the Poisson bracket:
 \f[ 
 j(x,y) := \{l(x,y), r(x,y)\} = \partial_x l \partial_y r - \partial_y l \partial_x r 
 \f] 
 * In the following algorithm the matrices are assumed to be the points
 \f[ left_{ij} = l(x_j, y_i) \f]
 \f[ right_{ij} = r(x_j, y_i) \f]
 \f[ x_j = x_0 + hj \f]
 \f[ y_i = z_0 + hi \f]
 * i.e the first index is the y - direction.  
 */
class Arakawa
{
  private:
    const double c;
  public:
    /*! @brief constructor
     *
     * @param h the physical grid constant
     */
    Arakawa( const double h): c(1.0/(12.0*h*h)){}
    /*! @brief Arakawa scheme working with ghostcells
     *
     * This function takes less than 0.03s for 1e6 elements
     * and is of O(N).
     * But could be twice as fast if only the interior function 
     * and no GhostMatrix were used! (At least on some processors, 
     *  with older compilers...)
     * @tparam GhostM the type of the GhostMatrix
     * @tparam M    the type of the Matrix
     * @param lhs the left function in the Poisson bracket
     * @param rhs the right function in the Poisson bracket
     * @param jac the Poisson bracket contains solution on output
     */
    template< class GhostM, class M>
    void operator()( const GhostM& lhs, const GhostM& rhs, M& jac);
};


template< class GhostM, class M>
void Arakawa::operator()(const GhostM& lhs, 
                         const GhostM& rhs, 
                         M& jac)
{
    const size_t rows = jac.rows(), cols = jac.cols();

    for( size_t j0 = 0; j0 < cols; j0++)
        jac(0,j0)       = c*boundary( 0, j0, lhs, rhs);
    for( size_t i0 = 1; i0 < rows-1; i0++)
    {
        jac(i0,0)       = c*boundary( i0, 0, lhs, rhs);
        for( size_t j0 = 1; j0 < cols-1; j0++)
            jac(i0,j0)  = c*interior( i0, j0, lhs, rhs);
        jac(i0,cols-1)  = c*boundary( i0, cols-1, lhs, rhs);
    }
    for( size_t i0 = 1; i0 < rows-1; i0++)
    {
        jac(i0,0)       = c*boundary( i0, 0, lhs, rhs);
        jac(i0,cols-1)  = c*boundary( i0, cols-1, lhs, rhs);
    }
    
    for( size_t j0 = 0; j0 < cols; j0++)
        jac(rows-1,j0)  = c*boundary( rows-1, j0, lhs, rhs);
}


/******************Access pattern of interior************************
 * xo.   
 * o .    andere Ecken analog (4 mal 2 Mult)
 * ...
 *
 * oo.
 * x .    andere Teile analog ( 4 mal 4 Mult)
 * oo.
 */
/*! @brief computes an interior point in the Arakawa scheme
 *
 *  @tparam M M class that has to provide m(i, j) access, a rows() and a cols() method.
 *      (type is normally inferred by the compiler)
 *  @param i0 row index of the interior point
 *  @param j0 col index of the interior point
 *  @param lhs left hand side 
 *  @param rhs right hand side 
 *  @return the unnormalized value of the Arakawa bracket
 */
template< class M>
double interior( const size_t i0, const size_t j0, const M& lhs, const M& rhs) 
{
    double jacob;
    const size_t ip = i0 + 1;
    const size_t jp = j0 + 1;
    const size_t im = i0 - 1;
    const size_t jm = j0 - 1;
    jacob  = rhs(i0,jm) * ( lhs(ip,j0) -lhs(im,j0) -lhs(im,jm) +lhs(ip,jm) );
    jacob += rhs(i0,jp) * (-lhs(ip,j0) +lhs(im,j0) -lhs(ip,jp) +lhs(im,jp) );
    jacob += rhs(ip,j0) * ( lhs(i0,jp) -lhs(i0,jm) +lhs(ip,jp) -lhs(ip,jm) );
    jacob += rhs(im,j0) * (-lhs(i0,jp) +lhs(i0,jm) +lhs(im,jm) -lhs(im,jp) );
    jacob += rhs(ip,jm) * ( lhs(ip,j0) -lhs(i0,jm) );
    jacob += rhs(ip,jp) * ( lhs(i0,jp) -lhs(ip,j0) );
    jacob += rhs(im,jm) * ( lhs(i0,jm) -lhs(im,j0) );
    jacob += rhs(im,jp) * ( lhs(im,j0) -lhs(i0,jp) );
    return jacob;
}

/*! @brief calculates a boundary point in the Arakawa scheme
 *
 *  It assumes periodic BC on the edges!
 *  @tparam M M class that has to provide m.at(i, j) access (e.g. GhostMatrix)
 *      (type is normally inferred by the compiler)
 *  @param i0 row index of the edge point
 *  @param j0 col index of the edge point
 *  @param lhs left hand side M
 *  @param rhs right hand side M
 *  @return the unnormalized value of the Arakawa bracket
 */
template< class M>
double boundary( const size_t i0, const size_t j0, const M& lhs, const M& rhs) 
{
    static QuadMat<double, 3> l, r;
    //assignment
    for( size_t i = 0; i < 3; i++)
        for( size_t j = 0; j < 3; j++)
        {
            l(i,j) = lhs.at( i0 -1 + i, j0 - 1 + j);
            r(i,j) = rhs.at( i0 -1 + i, j0 - 1 + j);
        }
    return interior( 1, 1, l, r);
}
} //namespace toefl
#endif// _TL_ARAKAWA_





