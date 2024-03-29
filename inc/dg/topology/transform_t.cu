#include <iostream>


#include "dg/backend/typedefs.h"
#include "base_geometry.h"
#include "transform.h"
#include "tensor.h"



struct TestGrid : public dg::aRealGeometry2d<double>
{
    using real_type = double;

    TestGrid( ): dg::aRealGeometry2d<real_type>({0.,1.,1,2,dg::PER}, {0.,1.,1,2,dg::PER}){}
    virtual TestGrid* clone()const override final{
        return new TestGrid(*this);
    }
    private:
    virtual void do_set(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) override final{
        aRealTopology2d<real_type>::do_set(nx,Nx,ny,Ny);
    }
    virtual dg::SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const override final {
        dg::SparseTensor<thrust::host_vector<real_type> > metric(*this);
        metric.values().resize(5);
        metric.values()[2] = dg::evaluate( dg::CONSTANT(2), *this);
        metric.values()[3] = dg::evaluate( dg::CONSTANT(3), *this);
        metric.values()[4] = dg::evaluate( dg::CONSTANT(4), *this);
        metric.idx( 0,0) = 2;
        metric.idx( 1,0) = metric.idx(0,1) = 3;
        metric.idx( 1,1) = 4;
        return metric;
        //( 2, 3 )
        //( 3, 4 )
    }
    virtual dg::SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const  override final{
        dg::SparseTensor<thrust::host_vector<real_type> > jac(*this);
        jac.values().resize(6);
        jac.values()[2] = dg::evaluate( dg::CONSTANT(0.1), *this);
        jac.values()[3] = dg::evaluate( dg::CONSTANT(0.2), *this);
        jac.values()[4] = dg::evaluate( dg::CONSTANT(0.3), *this);
        jac.values()[5] = dg::evaluate( dg::CONSTANT(0.4), *this);
        jac.idx(0,0) = 2; jac.idx( 0,1) = 3;
        jac.idx(1,0) = 4; jac.idx( 1,1) = 5;
        return jac;
        //( 0.1, 0.2 )
        //( 0.3, 0.4 )
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const override final{
        std::vector<thrust::host_vector<real_type> > map(2);
        map[0] = std::vector<real_type>{{0,1,1,0}};
        map[1] = std::vector<real_type>{{0,0,1,1}};
        return map;
    }
};

double test_function( double x, double y){
    return ( x*x + y*y);
}

int main()
{
    TestGrid  test;
    dg::HVec points = dg::pullback( test_function, test);
    std::cout << "Test transform functionality!\n";
    std::cout << "Test pullback\n";
    std::cout << points[0]<< " (0) "<<points[1]<<" (1) "<<points[2]<<" (2) "<<points[3]<< " (1)\n";
    if( points[0] != 0 || points[1] != 1 || points[2] != 2 || points[3] != 1)
        std::cerr << "Pullback FAILED\n";
    else
        std::cerr << "Pullback PASSED\n";
    std::cout << "Test pushForwardPerp\n";
    dg::HVec vx(points), vy(points), vz(points);
    dg::pushForwardPerp( dg::CONSTANT(4), dg::CONSTANT(5), vx, vy, test);
    std::cout << vx[0]<< " (1.4) "<<vy[0]<< " (3.2)\n";
    if( vx[0] - 1.4 > 1e-15 || vy[0] - 3.2 > 1e-15 )
        std::cerr << "PushForwardPerp FAILED\n";
    else
        std::cerr << "PushForwardPerp PASSED\n";
    std::cout << "Test pushForward\n";
    dg::pushForward( dg::CONSTANT(4), dg::CONSTANT(5), dg::CONSTANT(6), vx, vy, vz, test);
    std::cout << vx[0]<< " (1.4) "<<vy[0]<<" (3.2) "<<vz[0]<< " (6)\n";
    if( vx[0] - 1.4 > 1e-15 || vy[0] - 3.2 > 1e-15 || vz[0] - 6 > 1e-15 )
        std::cerr << "PushForward FAILED\n";
    else
        std::cerr << "PushForward PASSED\n";
    std::cout << "Test pushForwardPerp Tensor\n";
    dg::SparseTensor<dg::HVec> tensor;
    dg::pushForwardPerp( dg::CONSTANT(2), dg::CONSTANT(3), dg::CONSTANT(4), tensor, test);
    std::cout << tensor.value(0,0)[0]<<" (0.30) "<< tensor.value(0,1)[0]<<" (0.68) "<<std::endl;
    std::cout << tensor.value(1,0)[0]<<" (0.68) "<< tensor.value(1,1)[0]<<" (1.54)"<<std::endl;
    if( tensor.value(0,0)[0] - 0.30 > 1e-15 || tensor.value(0,1)[0] - 0.68 > 1e-15 ||
        tensor.value(1,0)[0] - 0.68 > 1e-15 || tensor.value(1,1)[0] - 1.54 > 1e-15)
        std::cerr << "PushForwardPerp Tensor FAILED\n";
    else
        std::cerr << "PushForwardPerp Tensor PASSED\n";
    return 0;
}
