#pragma once

#include "gridX.h"
#include "dxX.h"

/*! @file 
  
  Convenience functions to create 2D derivatives
  */
namespace dg{

template<class Matrix>
struct Composite
{
    Composite( ):m1(), m2(), dual(false){ }
    template<class Matrix2>
    Composite( const Composite<Matrix2>& src):m1(src.m1), m2(src.m2), dual(src.dual){}
    Composite( const Matrix& m):m1(m), m2(m), dual(false){ }
    Composite( const Matrix& m1, const Matrix& m2):m1(m1), m2(m2), dual(true){ }
    template<class Matrix2>
    Composite& operator=( const Composite<Matrix2>& src){ Composite c(src); 
        *this = c; return *this;}
    template< class container>
    void symv( const  container& v1, container& v2)
    {
        m1.symv( v1, v2);
        if( dual)
            m2.symv( v1, v2);
    }
    void display( std::ostream& os = std::cout) const
    {
        if( dual)
        {
            os << " dual matrix: \n";
            os << " INNER MATRIX\n";
            m1.display( os);
            os << " OUTER MATRIX\n";
            m2.display( os);
        }
        else
        {
            os << "single matrix: \n";
            m1.display(os);
        }
    }
    Matrix m1, m2;
    bool dual;
};
///@cond
template <class Matrix>
struct MatrixTraits<Composite<Matrix> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <class Matrix>
struct MatrixTraits<const Composite<Matrix> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond


/**
 * @brief Contains functions used for matrix creation
 */
namespace create{

///@addtogroup creation
///@{

//dx, dy, jumpX, jumpY

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx
 * @param bcx The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat<double> > dx( const aTopologyX2d& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat<double>  dx;
    dx = dx_normed( g.n(), g.Nx(), g.hx(), bcx, dir);
    dx.left_size = g.n()*g.Ny();
    dx.set_default_range();
    return dx;
}

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
Composite<EllSparseBlockMat<double> > dx( const aTopologyX2d& g, direction dir = centered) { return dx( g, g.bcx(), dir);}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
Composite<EllSparseBlockMat<double> > dy( const aTopologyX2d& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat<double>  dy_inner, dy_outer;
    GridX1d g1d_inner( g.y0(), g.y1(), g.fy(), g.n(), g.Ny(), bcy);
    Grid1d g1d_outer( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    dy_inner = dx( g1d_inner, bcy, dir);
    dy_outer = dx( g1d_outer, bcy, dir);
    dy_inner.right_size = g.n()*g.Nx();
    dy_inner.right_range[0] = 0;
    dy_inner.right_range[1] = g.n()*g.inner_Nx();
    dy_outer.right_range[0] = g.n()*g.inner_Nx();
    dy_outer.right_range[1] = g.n()*g.Nx();
    dy_outer.right_size = g.n()*g.Nx();

    Composite<EllSparseBlockMat<double> > c( dy_inner, dy_outer);
    return c;
}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat<double> > dy( const aTopologyX2d& g, direction dir = centered){ return dy( g, g.bcy(), dir);}

/**
 * @brief Matrix that contains 2d jump terms in X direction
 *
 * @param g grid
 * @param bcx boundary condition in x
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat<double> > jumpX( const aTopologyX2d& g, bc bcx)
{
    EllSparseBlockMat<double>  jx;
    jx = jump( g.n(), g.Nx(), g.hx(), bcx);
    jx.left_size = g.n()*g.Ny();
    jx.set_default_range();
    return jx;
}

/**
 * @brief Matrix that contains 2d jump terms in Y direction
 *
 * @param g grid
 * @param bcy boundary condition in y
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat<double> > jumpY( const aTopologyX2d& g, bc bcy)
{
    EllSparseBlockMat<double>  jy_inner, jy_outer;
    GridX1d g1d_inner( g.y0(), g.y1(), g.fy(), g.n(), g.Ny(), bcy);
    Grid1d g1d_outer( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    jy_inner = jump( g1d_inner, bcy);
    jy_outer = jump( g1d_outer, bcy);
    jy_inner.right_size = g.n()*g.Nx();
    jy_inner.right_range[0] = 0;
    jy_inner.right_range[1] = g.n()*g.inner_Nx();
    jy_outer.right_range[0] = g.n()*g.inner_Nx();
    jy_outer.right_range[1] = g.n()*g.Nx();
    jy_outer.right_size = g.n()*g.Nx();

    Composite<EllSparseBlockMat<double> > c( jy_inner, jy_outer);
    return c;
}

/**
 * @brief Matrix that contains 2d jump terms in X direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat<double> > jumpX( const aTopologyX2d& g)
{
    return jumpX( g, g.bcx());
}

/**
 * @brief Matrix that contains 2d jump terms in Y direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat<double> > jumpY( const aTopologyX2d& g)
{
    return jumpY( g, g.bcy());
}

///@}

} //namespace create

} //namespace dg

