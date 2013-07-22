#ifndef _DLT_CUH_
#define _DLT_CUH_

#include <fstream>
#include <vector>

namespace dg{

class DLT
{
  public:
    DLT( unsigned n);
    const std::vector& weights() {return w_;}
    const std::vector& abscissas() {return a_;}
    const std::vector& forward() {return forw_;}
    const std::vector& backward() {return back_;}
    const std::vector& backwardEQ() {return backEQ_;}

  private:
    std::vector a_, w_, forw_, back_, backEQ_;
};

DLT::DLT( unsigned n):a_(n), w_(n), forw_(n*n), back_(n*n),backEQ_(n*n)
{
    ifstream stream( "dlt.dat");

}

} //namespace dg
#endif//_DLT_CUH_
