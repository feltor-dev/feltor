#ifndef _DG_CG_
#define _DG_CG_

#include <cmath>
#include "blas.h"
namespace dg{


// TO DO: check for better stopping criteria using condition number estimates
/*!@brief class managing ressources for the conjugate gradient method

 The Matrix and Vector class are assumed to be double valued
 @tparam Matrix The matrix class no requirements except for the 
            BLAS routines
 @tparam Vector The Vector class: needs to be assignable, copyable, 
            v(unsigned) constructible, and deletable. 
 The following 3 pseudo - BLAS routines need to be callable
 double ddot( const Vector& v1, const Vector& v2)
 void daxpby( double alpha, const Vector& x, double beta, Vector& y)
 void dsymv( double alpha, const Matrix& m, const Vector& x, double beta, Vector& y)
 by specializing the corresponding template traits functions
*/
template< class Matrix, class Vector>
class CG
{
  public:
    CG( const Vector& copy, unsigned max_iter):r(copy), p(r), ap(r), max_iter(max_iter){}
    void set_max( unsigned new_max) {max_iter = new_max;}
    unsigned get_max() {return max_iter;}
    unsigned operator()( const Matrix& A, Vector& x, const Vector& b, double eps = 1e-12);//solve?
  private:
    Vector r, p, ap; //could contain solutions of previous iterations?
    unsigned max_iter;
};

template< class Matrix, class Vector>
unsigned CG<Matrix, Vector>::operator()( const Matrix& A, Vector& x, const Vector& b, double eps)
{
    double nrm2b = BLAS1<Vector>::ddot(b,b);
    r = b; BLAS2<Matrix, Vector>::dsymv( -1., A, x, 1.,r); //compute r_0 
    p = r;
    double nrm2r_old = BLAS1<Vector>::ddot( r, r); //and store the norm of it
    double alpha, nrm2r_new;
    for( unsigned i=1; i<max_iter; i++)
    {
        BLAS2<Matrix, Vector>::dsymv( 1., A, p, 0., ap);
        alpha = nrm2r_old /BLAS1<Vector>::ddot( p, ap);
        BLAS1<Vector>::daxpby( alpha, p, 1.,x);
        BLAS1<Vector>::daxpby( -alpha, ap, 1., r);
        nrm2r_new = BLAS1<Vector>::ddot( r,r);
        if( sqrt( nrm2r_new/nrm2b) < eps) 
            return i;
        BLAS1<Vector>::daxpby(1., r, nrm2r_new/nrm2r_old, p );
        nrm2r_old=nrm2r_new;
    }
    return max_iter;
}

template< class Matrix, class Vector, class Preconditioner>
class PCG
{
  public:
    PCG( const Vector& copy, unsigned max_iter):r(copy), p(r), ap(r), max_iter(max_iter){}
    void set_max( unsigned new_max) {max_iter = new_max;}
    unsigned get_max() {return max_iter;}
    unsigned operator()( const Matrix& A, Vector& x, const Vector& b, const Preconditioner& P, double eps = 1e-12);//solve?
  private:
    Vector r, p, ap; //could contain solutions of previous iterations?
    unsigned max_iter;
};

/*
//compared to unpreconditioned compare
//ddot(r,r), axpby()
//to 
//ddot( r,P,r), dsymv(P)
//i.e. it will be slower, if P needs to be stored
//(but in our case P_{ii} can be computed directly
//compared to normal preconditioned compare
//ddot(r,P,r), dsymv(P)
//to
//ddot(r,z), dsymv(P), axpby(), (storage for z)
//i.e. it's surely faster if P contains no more elements than z 
//(which is the case for diagonal scaling)
//NOTE: the same comparison hold for A with the result that A contains 
//significantly more elements than z whence ddot(r,A,r) is far slower than ddot(r,z)
*/
template< class Matrix, class Vector, class Preconditioner>
unsigned PCG< Matrix, Vector, Preconditioner>::operator()( const Matrix& A, Vector& x, const Vector& b, const Preconditioner& P, double eps)
{
    double nrm2b = BLAS2<Preconditioner, Vector>::ddot( b,P,b);
    r = b; BLAS2<Matrix, Vector>::dsymv( -1., A, x, 1.,r); //compute r_0 
    BLAS2<Preconditioner, Vector>::dsymv(1.,P, r, 0., p );//<-- compute p_0
    double nrm2r_old = BLAS2<Preconditioner, Vector>::ddot( r,P,r); //and store the norm of it
    double alpha, nrm2r_new;
    for( unsigned i=1; i<max_iter; i++)
    {
        BLAS2<Matrix, Vector>::dsymv( 1., A, p, 0., ap);
        alpha = nrm2r_old /BLAS1<Vector>::ddot( p, ap);
        BLAS1<Vector>::daxpby( alpha, p, 1.,x);
        BLAS1<Vector>::daxpby( -alpha, ap, 1., r);
        nrm2r_new = BLAS2<Preconditioner, Vector>::ddot( r,P, r); //<--
        if( sqrt( nrm2r_new/nrm2b) < eps) 
            return i;
        BLAS2<Preconditioner, Vector>::dsymv(1.,P, r, nrm2r_new/nrm2r_old, p );//<--
        nrm2r_old=nrm2r_new;
    }
    return max_iter;
}


} //namespace dg



#endif //_DG_CG_
