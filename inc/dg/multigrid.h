#pragma once

#include "backend/fast_interpolation.h"
#include "backend/exceptions.h"
#include "backend/memory.h"
#include "blas.h"

namespace dg
{


template< class Geometry, class Matrix, class container> 
struct Multigrid2d
{
    Multigrid2d( const Geometry& grid, unsigned stages ) {
        if( stages < 2 ) throw Error( Message(_ping_)<<" There must be minimum 2 stages in a multigrid solver! You gave " << stages);
        grids_.resize( stages );
        grids_[0].reset( grid);
        for( unsigned u=1; u<stages; u++)
        {
            grids_[u] = grids_[u-1]; //deep copy
            grids_[u].get().multiplyCellNumbers(0.5,0.5);
        }
        inter_.resize(stages-1)
        project_.resize( stages-1);
        for( unsigned u=0; u<stages-1; u++)
        {
            project_[u] = dg::create::fast_projection( grids_[u], 2,2);
            inter_[u] = dg::create::fast_interpolation(grids_[u+1],2,2);
        }

        cg_.resize( stages);
    }



    private:
    std::vector< Handle< Geometry> > grids_;
    std::vector< MultiMatrix<Matrix, container>  >  inter_;
    std::vector< MultiMatrix<Matrix, container>  >  project_;
    std::vector< CG<container> > cg_;
};

}//namespace dg
