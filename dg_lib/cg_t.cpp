#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>

#include "operators.h"
#include "evaluation.h"
#include "laplace.h"
#include "dlt.h"
#include "cg.h"


#define P 2
typedef std::vector<std::array<double, P>> Vector;
typedef dg::Laplace<P> Matrix;

namespace dg{
template < >
struct CG_BLAS1<Vector>
{
    static double ddot( const Vector& x, const Vector& y)
    {
        double sum = 0;
        double s = 0;
        for( unsigned i=0; i<x.size(); i++)
        {
            s=0;
            for( unsigned j=0; j<P; j++)
                s+= x[i][j]*y[i][j];
            sum +=s;
        }
        return sum;
    }
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        for( unsigned i=0; i<x.size(); i++)
            for( unsigned j=0; j<P; j++)
                y[i][j] = alpha*x[i][j]+beta*y[i][j];
    }

};



template <>
struct CG_BLAS2< Laplace<P>, Vector>
{
    static void dsymv( double alpha, const dg::Laplace<P>& m, const Vector& x, double beta, Vector& y)
    {
        /*
            y[0] = alpha*( B^T*x[N-1] + A*x[0] + B*x[1]  ) + beta*y[0];
            y[k] = alpha*( B^T*x[k-1] + A*x[k] + B*x[k+1]) + beta*y[k];
            y[N] = alpha*( B^T*x[N-1] + A*x[N] + B*x[0]  ) + beta*y[N];
        */
        const dg::Operator<double, P> & a = m.get_a();
        const dg::Operator<double, P> & b = m.get_b();
        const unsigned N = x.size();

        for( unsigned i=0; i<P; i++)
        {
            y[0][i] = beta*y[0][i];
            for( unsigned j=0; j<P; j++)
                y[0][i] += alpha*(b(j,i)*x[N-1][j] + a(i,j)*x[0][j] + b(i,j)*x[1][j]);
        }
        for( unsigned k=1; k<N-1; k++)
            for( unsigned i=0; i<P; i++)
            {
                y[k][i] = beta*y[k][i];
                for( unsigned j=0; j<P; j++)
                    y[k][i] += alpha*(b(j,i)*x[k-1][j] + a(i,j)*x[k][j] + b(i,j)*x[k+1][j]);
            }
        for( unsigned i=0; i<P; i++)
        {
            y[N-1][i] = beta*y[N-1][i];
            for( unsigned j=0; j<P; j++)
                y[N-1][i] += alpha*(b(j,i)*x[N-2][j] + a(i,j)*x[N-1][j] + b(i,j)*x[0][j]);
        }
    }
};

template <>
struct CG_BLAS2< Laplace_Dir<P>, Vector>
{
    static void dsymv( double alpha, const Laplace_Dir<P>& m, const Vector& x, double beta, Vector& y)
    {
        /*
            y[0] = alpha*(              Ap*x[0]+ Bp*x[1] ) + beta*y[0];
            y[1] = alpha*(  Bp^T*x[0] + Ap*x[1]+ B*x[2]  ) + beta*y[1];
            y[k] = alpha*( B^T*x[k-1] + A*x[k] + B*x[k+1]) + beta*y[k];
          y[N-1] = alpha*( B^T*x[N-2] + A*x[N-1]         ) + beta*y[N-1];
        */
        const dg::Operator<double, P> & a = m.get_a();
        const dg::Operator<double, P> & b = m.get_b();
        const unsigned N = x.size();

        const dg::Operator<double, P> & ap = m.get_ap();
        const dg::Operator<double, P> & bp = m.get_bp();
        for( unsigned i=0; i<P; i++)
        {
            y[0][i] = beta*y[0][i];
            for( unsigned j=0; j<P; j++)
                y[0][i] += alpha*( ap(i,j)*x[0][j] + bp(i,j)*x[1][j]);
        }
        for( unsigned i=0; i<P; i++)
        {
            y[1][i] = beta*y[1][i];
            for( unsigned j=0; j<P; j++)
                y[1][i] += alpha*( bp(j,i)*x[0][j] + a(i,j)*x[1][j] + b(i,j)*x[2][j]);
        }
        for( unsigned k=2; k<N-1; k++)
            for( unsigned i=0; i<P; i++)
            {
                y[k][i] = beta*y[k][i];
                for( unsigned j=0; j<P; j++)
                    y[k][i] += alpha*(b(j,i)*x[k-1][j] + a(i,j)*x[k][j] + b(i,j)*x[k+1][j]);
            }
        for( unsigned i=0; i<P; i++)
        {
            y[N-1][i] = beta*y[N-1][i];
            for( unsigned j=0; j<P; j++)
                y[N-1][i] += alpha*(b(j,i)*x[N-2][j] + a(i,j)*x[N-1][j] );
        }
    }
};

} //namespace dg
using namespace std;
using namespace dg;

double sinus(double x){ return sin(2*M_PI*x);}
double secondsinus(double x){ return 4.*M_PI*M_PI*sin(2*M_PI*x);}


int main()
{
    const unsigned num_int = 1000;
    const double h = 1./(double)num_int;
    Matrix l(h); //the constant makes all projection operators correct
    cout << l.get_a()<<endl;
    cout << l.get_b()<<endl;
    Operator<double,P> forward( DLT<P>::forward);

    Vector x = evaluate< double(&)(double), P>( sinus, 0,1, num_int);
    Vector solution = evaluate< double(&)(double), P>( secondsinus, 0,1, num_int);
    cout << "Square norm of sine is : "<<square_norm( x, XSPACE)*h/2.<<endl;
    cout << "Square norm of solution (779.273) is : "<<square_norm( solution, XSPACE)*h/2.<<endl;
    
    for( unsigned i=0; i<num_int; i++)
        x[i] = forward*x[i];
    for( unsigned i=0; i<num_int; i++)
        solution[i] = forward*solution[i];
    double s_norm2 = CG_BLAS2<S,Vector>::ddot(x,S(h),x); 

    cout << "Square norm of sine is : "<<s_norm2 <<endl;    cout << "Square norm of solution is : "<<CG_BLAS2<S, Vector>::ddot( solution, S(h), solution) <<endl;
    //cout << "Sine approximation\n";
    //for( unsigned i=0; i<num_int; i++)
    //{
    //    for( unsigned j=0; j<P; j++)
    //        cout << x[i][j]<<" ";
    //    cout << "\n";
    //}
    //cout << "Solution: \n";
    //for( unsigned i=0; i<num_int; i++)
    //{
    //    for( unsigned j=0; j<P; j++)
    //        cout << solution[i][j]<<" ";
    //    cout << "\n";
    //}
    Vector w(num_int);
    dg::CG_BLAS2<Matrix, Vector>::dsymv( 1., l, x, 0, w);
    dg::CG_BLAS2<T, Vector>::dsymv( 1., T(h), w, 0, w);
    double w_norm2 = CG_BLAS2<S, Vector>::ddot(w, S(h), w);
    //cout << "Approximation: \n";
    //for( unsigned i=0; i<num_int; i++)
    //{
    //    for( unsigned j=0; j<P; j++)
    //        cout << w[i][j]<<" ";
    //    cout << "\n";
    //}
    cout << "Square norm of w is: "<< w_norm2 << endl;
    dg::CG_BLAS1<Vector>::daxpby( 1., solution, -1., w);
    cout << "Relative error in L2 norm is \n";

    ofstream os( "error.dat");
    for( unsigned i=0; i<num_int; i++)
        os << (double)i*h << " "<< w[i][0] << " "<<solution[i][0]<<" "
           << w[i][1]<< " "<<solution[i][1]<<"\n";
    
    cout << sqrt(CG_BLAS2<S, Vector>::ddot(w, S(h), w)/s_norm2)<<endl;


    return 0;
}

