#pragma once

#include "backend/fast_interpolation.h"
#include "backend/interpolation.cuh"
#include "backend/exceptions.h"
#include "backend/memory.h"
#include "blas.h"
#include "cg.h"

namespace dg
{


template< class Geometry, class Matrix, class container> 
struct MultigridCG2d
{
    MultigridCG2d( const Geometry& grid, unsigned stages, int extrapolation_type = 2 ) {
        stages_=stages;
        if( stages < 2 ) throw Error( Message(_ping_)<<" There must be minimum 2 stages in a multigrid solver! You gave " << stages);
        grids_.resize( stages);
        grids_[0].reset( grid);
        grids_[0].get().display();
        for( unsigned u=1; u<stages; u++)
        {
            grids_[u] = grids_[u-1]; //deep copy
            grids_[u].get().multiplyCellNumbers(0.5, 0.5);
            grids_[u].get().display();
        }
        inter_.resize(stages-1);
        project_.resize( stages-1);
        for( unsigned u=0; u<stages-1; u++)
        {
            //Projecting from one grid to the next is the same as 
            //projecting from the original grid to the coarse grids
            project_[u] = dg::create::fast_projection( grids_[u].get(), 2,2);
            inter_[u] = dg::create::fast_interpolation(grids_[u+1].get(),2,2);
        }

        dg::blas1::transfer( dg::evaluate( dg::zero, grid), x0_);
        x1_=x0_, x2_=x0_;
        x_ =  project( x0_); 
        r_ = x_, b_ = x_;
        set_extrapolationType(extrapolation_type);
        cg_.resize( stages);
        for( unsigned u=0; u<stages; u++)
            cg_[u].construct( x_[u], x_[u].size());
    }

    template<class SymmetricOp>
    std::vector<unsigned> direct_solve( std::vector<SymmetricOp>& op, container&  x, const container& b, double eps)
    {
        dg::blas1::axpbygz( alpha[0], x0_, alpha[1], x1_, alpha[2], x2_); 
        x_[0].swap(x2_);
        project( x_[0], x_);
        project( b, b_);
        std::vector<unsigned> number(stages_);
        for( unsigned u=stages_-1; u>0; u--)
        {
            dg::blas1::pointwiseDivide( b_[u], op[u].inv_weights(), b_[u]);
            number[u] = cg_[u]( op[u], x_[u], b_[u], op[u].precond(), op[u].inv_weights(), eps, 1.);
            dg::blas2::symv( inter_[u-1], x_[u], x_[u-1]);
        }
        dg::blas1::pointwiseDivide( b_[0], op[0].inv_weights(), b_[0]);
        number[0] = cg_[0]( op[0], x_[0], b_[0], op[0].precond(), op[0].inv_weights(), eps);
        x_[0].swap( x);
        x1_.swap( x2_);
        x0_.swap( x1_);
        
        blas1::copy( x, x0_);
        return number;
    }

    ///src may alias first element of out
    void project( const container& src, std::vector<container>& out)
    {
        dg::blas1::copy( src, out[0]);
        for( unsigned u=0; u<grids_.size()-1; u++)
            dg::blas2::gemv( project_[u], out[u], out[u+1]);
    }

    std::vector<container> project( const container& src)
    {
        std::vector<container> out( grids_.size());
        for( unsigned u=0; u<grids_.size(); u++)
            dg::blas1::transfer( dg::evaluate( dg::zero, grids_[u].get()), out[u]);
        project( src, out);
        return out;

    }
    unsigned stages()const{return stages_;}
    /**
     * @brief Set the extrapolation Type for following inversions
     *
     * @param extrapolationType number of last values to use for next extrapolation of initial guess
     */
    void set_extrapolationType( int extrapolationType)
    {
        assert( extrapolationType <= 3 && extrapolationType >= 0);
        switch(extrapolationType)
        {
            case(0): alpha[0] = 0, alpha[1] = 0, alpha[2] = 0;
                     break;
            case(1): alpha[0] = 1, alpha[1] = 0, alpha[2] = 0;
                     break;
            case(2): alpha[0] = 2, alpha[1] = -1, alpha[2] = 0;
                     break;
            case(3): alpha[0] = 3, alpha[1] = -3, alpha[2] = 1;
                     break;
            default: alpha[0] = 2, alpha[1] = -1, alpha[2] = 0;
        }
    }

    const std::vector<dg::Handle< Geometry > > grids()const{return grids_;}

    private:
    unsigned stages_;
    std::vector< dg::Handle< Geometry> > grids_;
    std::vector< MultiMatrix<Matrix, container>  >  inter_;
    std::vector< MultiMatrix<Matrix, container>  >  project_;
    std::vector< CG<container> > cg_;
    std::vector< container> x_, r_, b_; 
    container x0_, x1_, x2_;
    double alpha[3];
};

}//namespace dg
