#ifndef _DG_VECTOR_
#define _DG_VECTOR_

#include <thrust/host_vector.h>

namespace dg
{

template< size_t n>
class ArrVec1d
{
    public:
    typedef thrust::host_vector<double> container;
    ArrVec1d(){}
    ArrVec1d( unsigned size, double value=0): hv( n*size, value){}
    double& operator()( unsigned i, unsigned k) {return hv[ i*n+k];}
    const double& operator()( unsigned i, unsigned k) const
    { 
        return hv[i*n+k];
    }
    container& data(){ return hv;}
    const container& data() const {return hv;}
    private:
    container hv;
};

}//namespace dg

#endif //_DG_VECTOR_
