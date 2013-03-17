#ifndef _DG_VECTOR_
#define _DG_VECTOR_

namespace dg
{

//TODO some safety measurments
template< typename T, size_t n, class container = thrust::host_vector<T> >
class ArrVec1d_View
{
    public:
    typedef container Vector;
    ArrVec1d_View( container& v ):hv(v){ }
    double& operator()( unsigned i, unsigned k) {return hv[ i*n+k];}
    const double& operator()( unsigned i, unsigned k) const
    { 
        return hv[i*n+k];
    }
    container& data(){ return hv;}
    const container& data() const {return hv;}

    friend std::ostream& operator<<( std::ostream& os, const ArrVec1d_View& v)
    {
        unsigned N = v.hv.size()/n;
        for( unsigned i=0; i<N; i++)
        {
            for( unsigned j=0; j<n; j++)
                os << v(i,j) << " ";
            os << "\n";
        }
        return os;
    }
    private:
    container& hv;
};

//an Array is a View but owns the data it views
template< typename T, size_t n, class container = thrust::host_vector<T> >
class ArrVec1d : public ArrVec1d_View<T, n, container>
{
    public:
    typedef ArrVec1d_View<T, n, container> View;
    ArrVec1d():View(hv){}
    ArrVec1d( const container& c): View(hv), hv(c) {}
    ArrVec1d( unsigned size, double value=0):View(hv), hv( n*size, value){}
    private:
    container hv;
};

}//namespace dg

#endif //_DG_VECTOR_
