#ifndef _DG_CG_
#define _DG_CG_

namespace dg{

/*!@brief class managing ressources for the conjugate gradient method

 The Matrix and Vector class are assumed to be double valued
 @tparam Matrix The matrix class no requirements except for the 
            CG_BLAS routines
 @tparam Vector The Vector class: needs to be assignable, copyable, 
            v(unsigned) constructible, and deletable. 
 The following 3 pseudo - CG_BLAS routines need to be callable
 double ddot( const Vector& v1, const Vector& v2)
 void daxpby( double alpha, const Vector& x, double beta, Vector& y)
 void dsymv( double alpha, const Matrix& m, const Vector& x, double beta, Vector& y)
 by specializing the corresponding template traits functions


 TO DO: check for better stopping criteria using condition number estimates
*/
template < class Vector>
class CG_BLAS1
{
    double ddot( const Vector& x, const Vector& y);
    void daxpby( double alpha, const Vector& x, double beta, Vector& y);
};

template < class Matrix, class Vector>
class CG_BLAS2
{
    void dsymv( double alpha, const Matrix& m, const Vector& x, double beta, Vector& y);
};

template< class Matrix, class Vector>
class CG
{
  public:
    CG( unsigned size):r(size), p(r), ap(r), eps(1e-10), max_iter(size){}
    void set_eps( double eps_rel) {eps = eps_rel;}
    double get_eps( ) {return eps;}
    void set_max( unsigned new_max) {max_iter = new_max;}
    unsigned get_max() {return max_iter;}
    unsigned solve( const Matrix& A, Vector& x, const Vector& b);
  private:
    Vector r, p, ap;
    double eps;
    unsigned max_iter;
};

template< class Matrix, class Vector>
unsigned CG::solve( const Matrix& A, Vector& x, const Vector& b)
{
    double nrm2b = CG_BLAS1<Vector>::ddot(b,b);
    p = r = b;
    CG_BLAS2<Matrix, Vector>::dsymv( -1., A, x, 1.,r); //compute r_0 
    double nrm2r_old = ddot( r, r); //and store the norm of it
    double alpha, nrm2r_new;
    for( unsigned i=1; i<max_iter; i++)
    {
        CG_BLAS2<Matrix, Vector>::dsymv( 1., A, p, 0., ap);
        alpha = nrm2r_old /ddot( p, ap);
        CG_BLAS1<Vector>::daxpby( alpha, p, 1.,x);
        CG_BLAS1<Vector>::daxpby( -alpha, ap, 1., r);
        nrm2r_new = ddot( r,r);
        if( sqrt( nrm2r_new/nrm2b) < eps) return i;
        CG_BLAS1<Vector>::daxpby(1., r, nrm2r_new/nrm2r_old, p );
    }
    return max_iter;
}



#endif //_DG_CG_
