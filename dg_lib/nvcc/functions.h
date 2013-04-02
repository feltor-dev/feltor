#ifndef _DG_PROJECTION_
#define _DG_PROJECTION_

namespace dg{

///@addtogroup utilities
///@{
//namespace create{
//namespace detail{
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
//} //namespace detail
//} //namespace create
} //namespace dg

///@}

#endif //_DG_PROJECTION_
