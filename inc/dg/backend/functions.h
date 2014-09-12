
#pragma once

namespace dg{
///@cond
/**
* @brief The delta function
*
* @param i Index 
* @param j Index
*
* @return delta_{ij}
*/
inline double delta( unsigned i, unsigned j)
{
    if( i==j ) 
        return 1.;
    return 0.;
}
/**
* @brief Scalar product of two Legendre functions
*
* Corresponds to the S - Matrix
* @param i Index of Legendre function
* @param j Index of Legendre function
*
* @return L2 scalar product
*/
inline double pipj( unsigned i, unsigned j)
{
    if( i==j) 
        return 2./(2.*(double)i+1.);
    return 0;
}

/**
* @brief Inverse of the scalar product matrix
*
* Corresponds to the T - Matrix
* @param i Index of Legendre function
* @param j Index of Legendre function
*
* @return Inverse of L2 scalar product
*/
inline double pipj_inv( unsigned i, unsigned j)
{
    if( i==j)
        return (2.*(double)i+1.)/2.;
    return 0;
}

/**
* @brief Scalar product between Legendre function and a derivative
*
* Corresponds to the D - Matrix
* @param i Index of first Legendre function
* @param j Index of second Legendre function
*
* @return Scalar product of p_i(x) and partial_x p_j(x)
*/
inline double pidxpj( unsigned i, unsigned j)
{
    if( i < j)
    {
        if( (i+j)%2 == 0)
            return 0;
        else
            return 2;
    }
    return 0;
}


/**
* @brief Evaluation of Legendre-functions left and right of an edge point
*
* Corresponds to the R - Matrix
* @param i Index of first Legendre function
* @param j Index of second Legendre function
*
* @return 1
*/
inline double rirj( unsigned i, unsigned j)
{
    return 1.;
}
/**
* @brief Evaluation of Legendre-functions left and right of an edge point
*
* Corresponds to the RL - Matrix
* @param i Index of first Legendre function
* @param j Index of second Legendre function
*
* @return (-1)^j
*/
inline double rilj( unsigned i, unsigned j)
{
    if( j%2==0)
        return 1.;
    return -1.;
}
/**
* @brief Evaluation of Legendre-functions left and right of an edge point
*
* Corresponds to the LR - Matrix
* @param i Index of first Legendre function
* @param j Index of second Legendre function
*
* @return (-1)^i 
*/
inline double lirj( unsigned i, unsigned j)
{
    return rilj(j,i);
}
/**
* @brief Evaluation of Legendre-functions left and right of an edge point
*
* Corresponds to the L - Matrix
* @param i Index of first Legendre function
* @param j Index of second Legendre function
*
* @return (-1)^{i+j}
*/
inline double lilj( unsigned i, unsigned j)
{
    if( (i+j)%2 ==0)
        return 1.;
    return -1.;
}
///@endcond

///@addtogroup functions
///@{

/**
 * @brief The constant 
 *
 * Useful for integration.
 * @param x x-value
 *
 * @return 1
 */
inline double one( double x) {return 1;}

/**
 * @brief The constant 
 *
 * Useful for integration.
 * @param x x-value
 * @param y y-value
 *
 * @return 1
 */
inline double one( double x, double y) {return 1;}

/**
 * @brief The constant 
 *
 * Useful for integration.
 * @param x x-value
 * @param y x-value
 * @param z x-value
 *
 * @return 1
 */
inline double one( double x, double y, double z) {return 1;}

/**
 * @brief The constant 
 *
 * Useful for integration.
 * @param x x-value
 *
 * @return 1
 */
inline double zero( double x) {return 0.;}

/**
 * @brief The constant 
 *
 * Useful for integration.
 * @param x x-value
 * @param y y-value
 *
 * @return 1
 */
inline double zero( double x, double y) {return 0.;}

/**
 * @brief The constant 
 *
 * Useful for integration.
 * @param x x-value
 * @param y x-value
 * @param z x-value
 *
 * @return 1
 */
inline double zero( double x, double y, double z) {return 0.;}

double coo1( double x) {return x;}
double coo1( double x, double y) {return x;}
double coo1( double x, double y, double z) {return x;}
double coo2( double x, double y) {return y;}
double coo2( double x, double y, double z) {return y;}
double coo3( double x, double y, double z) {return z;}
} //namespace dg

///@}

