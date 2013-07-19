#ifndef _DLT_CUH_
#define _DLT_CUH_

#include <fstream>
#include <stdexcept>
#include <vector>

namespace dg{

template< class T>
class DLT
{
  public:
    DLT( unsigned n);
    const std::vector<T>& weights()const {return w_;}
    const std::vector<T>& abscissas()const {return a_;}
    const std::vector<T>& forward()const {return forw_;}
    const std::vector<T>& backward()const {return back_;}
    const std::vector<T>& backwardEQ()const {return backEQ_;}

  private:
    std::vector<T> a_, w_, forw_, back_, backEQ_;
};

template<class T>
DLT<T>::DLT( unsigned n):a_(n), w_(n), forw_(n*n), back_(n*n),backEQ_(n*n)
{
    std::ifstream stream( "dlt.dat");
    if( stream.fail()) 
        throw "File 'dlt.dat' corrupted!";
    double x;
    for( unsigned i=1; i<n; i++)
    {
        for( unsigned j=0; j<2*i+3*i*i ;j++)
            stream >> x;
    }
    for( unsigned j=0; j<n; j++) stream >> a_[j];
    for( unsigned j=0; j<n; j++) stream >> w_[j];
    for( unsigned j=0; j<n*n; j++) stream >> back_[j];
    for( unsigned j=0; j<n*n; j++) stream >> forw_[j];
    for( unsigned j=0; j<n*n; j++) stream >> backEQ_[j];
    if( !stream.good())
        throw "Error reading file dlt.dat! Is n > 20?";
    stream.close();
}


} //namespace dg
#endif//_DLT_CUH_
