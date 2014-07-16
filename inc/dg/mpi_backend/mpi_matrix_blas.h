#pragma once

namespace dg
{
namespace blas2
{
namespace detail
{

    //Deconstifies x!!!!!
template< class Matrix, class Vector>
inline void doSymv( const Matrix& m, const Vector& x_, Vector& y, MPIMatrixTag, MPIVectorTag  )
{
#ifdef DG_DEBUG
    assert( x.data.size() == y.data.size() );
    assert( m.data.rows == m.data.cols );
    assert( m.data.rows == x.data.size() );
#endif //DG_DEBUG
    Vector& x = const_cast<Vector&>( x_);//!!!!!!!!!!!!!!!!!!!!!!!!!
    if( m.state == 0) //x-direction
    {
        m.update_boundaryX( x);
        for( unsigned i=1; i<m.rows-1; i++)
            for( unsigned j=1; j<m.cols-1; j++)
            {
                for( unsigned k=0; k<n*n; k++)
                    y.data[(i*m.cols+j)*n*n +k] = 0;
                multiplyAdd( n, m.data[0], &x.data[(i*m.cols+j-1)*n*n], &y.data[(i*m.cols+j)*n*n]);
                multiplyAdd( n, m.data[1], &x.data[(i*m.cols+j  )*n*n], &y.data[(i*m.cols+j)*n*n]);
                multiplyAdd( n, m.data[2], &x.data[(i*m.cols+j+1)*n*n], &y.data[(i*m.cols+j)*n*n]);
            }
    }
    else if( m.state == 1) //y-direction
    {
        m.update_boundaryY( x);
        for( unsigned i=1; i<m.rows-1; i++)
            for( unsigned j=1; j<m.cols-1; j++)
            {
                for( unsigned k=0; k<n*n; k++)
                    y.data[(i*m.cols+j)*n*n +k] = 0;
                multiplyAdd( m.data[0], n, &x.data[((i-1)*m.cols+j)*n*n], &y.data[(i*m.cols+j)*n*n]);
                multiplyAdd( m.data[1], n, &x.data[( i   *m.cols+j)*n*n], &y.data[(i*m.cols+j)*n*n]);
                multiplyAdd( m.data[2], n, &x.data[((i+1)*m.cols+j)*n*n], &y.data[(i*m.cols+j)*n*n]);
            }
    }
    else //laplace
    {
        m.update_boundaryX( x);
        m.update_boundaryY( x);
        for( unsigned i=1; i<m.rows-1; i++)
            for( unsigned j=1; j<m.cols-1; j++)
            {
                for( unsigned k=0; k<n*n; k++)
                    y.data[(i*m.cols+j)*n*n +k] = 0;
                multiplyAdd( n, m.data[0], &x.data[(i*m.cols+j-1)*n*n],   &y.data[(i*m.cols+j)*n*n]);
                multiplyAdd( m.data[0], n, &x.data[((i-1)*m.cols+j)*n*n], &y.data[(i*m.cols+j)*n*n]);
                multiplyAdd( m.data[1], n, &x.data[( i   *m.cols+j)*n*n], &y.data[(i*m.cols+j)*n*n]);
                multiplyAdd( m.data[2], n, &x.data[((i+1)*m.cols+j)*n*n], &y.data[(i*m.cols+j)*n*n]);
                multiplyAdd( n, m.data[2], &x.data[(i*m.cols+j+1)*n*n],   &y.data[(i*m.cols+j)*n*n]);
            }

    }
}

template< class Matrix, class Vector>
inline void doGemv( const Matrix& m, const Vector&x, Vector& y, MPIMatrixTag, MPIVectorTag  )
{
    doSymv( m, x, y, MPIMatrixTag(), MPIVectorTag());
}

} //namespace detail
} //namespace blas2
} //namespace dg
