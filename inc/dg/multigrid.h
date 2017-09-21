#pragma once

#include "backend/fast_interpolation.h"
#include "backend/interpolation.cuh"
#include "backend/exceptions.h"
#include "backend/memory.h"
#include "blas.h"
#include "cg.h"
#include "backend/timer.cuh"

namespace dg
{

/**
* @brief Class for the solution of symmetric matrix equation discretizeable on multiple grids
*
* We use conjugate gradien (CG) at each stage and refine the grids in the first two dimensions (2d / x and y) 
* @copydoc hide_geometry_matrix_container
* @ingroup numerical1
*/
template< class Geometry, class Matrix, class container> 
struct MultigridCG2d
{
    /**
    * @brief Construct the grids and the interpolation/projection operators
    *
    * @param grid the original grid (Nx() and Ny() must be evenly divisable by pow(2, stages-1)
    * @param stages number of grids in total (The second grid contains half the points of the original grids,  
    *   The third grid contains half of the second grid ...). Must be > 1
    * @param scheme_type scheme type in the solve function
    */
    MultigridCG2d( const Geometry& grid, const unsigned stages, const int scheme_type = 0 )
    {
        stages_= stages;
        if(stages < 2 ) throw Error( Message(_ping_)<<" There must be minimum 2 stages in a multigrid solver! You gave " << stages);
        
		grids_.resize(stages);
        cg_.resize(stages);

        grids_[0].reset( grid);
        //grids_[0].get().display();
        
		for(unsigned u=1; u<stages; u++)
        {
            grids_[u] = grids_[u-1]; // deep copy
            grids_[u].get().multiplyCellNumbers(0.5, 0.5);
            //grids_[u].get().display();
        }
        
		inter_.resize(stages-1);
		interT_.resize(stages-1);
        project_.resize( stages-1);
        
		for(unsigned u=0; u<stages-1; u++)
        {
            // Projecting from one grid to the next is the same as 
            // projecting from the original grid to the coarse grids
            project_[u] = dg::create::fast_projection(grids_[u].get(), 2, 2, dg::normed);
            inter_[u] = dg::create::fast_interpolation(grids_[u+1].get(), 2, 2);
            interT_[u] = dg::create::fast_projection(grids_[u].get(), 2, 2, dg::not_normed);
        }

        container x0;
        dg::blas1::transfer( dg::evaluate( dg::zero, grid), x0);
        x_ = project(x0); 
        m_r = x_,
		b_ = x_;        
        set_scheme(scheme_type);        
    }

	template<class SymmetricOp>
	std::vector<unsigned> solve(/*const*/ std::vector<SymmetricOp>& op, container& x, const container& b, const double eps)
	{
        //project initial guess down to coarse grid
        project(x, x_);
        dg::blas2::symv(op[0].weights(), b, b_[0]);
        // project b down to coarse grid
        for( unsigned u=0; u<stages_-1; u++)
            dg::blas2::gemv( interT_[u], b_[u], b_[u+1]);

        
        unsigned int numStageSteps = m_schemeLayout.size();
		std::vector<unsigned> number(numStageSteps);

        unsigned u = m_startStage;
                        
        for (unsigned i = 0; i < numStageSteps; i++)
        {
            unsigned w = u + m_schemeLayout[i].m_step;
            //
            // iterate the solver on the system A x = b, with x = 0 as inital guess
            cg_[u].set_max(m_schemeLayout[i].m_niter);
            number[i] = cg_[u](op[u], x_[u], b_[u], op[u].precond(), op[u].inv_weights(), eps);

            //
            // debug print
            std::cout << "pass: " << i << ", stage: " << u << ", max iter: " << m_schemeLayout[i].m_niter << ", iter: " << number[i] << std::endl;

            if (m_schemeLayout[i].m_step > 0)
            {
                //
                // compute residual r = Wb - A x
                dg::blas2::symv(op[u], x_[u], m_r[u]);
                dg::blas1::axpby(-1.0, m_r[u], 1.0, b_[u], m_r[u]);
                //
                // transfer residual to the rhs of the coarser grid
                dg::blas2::symv(interT_[u], m_r[u], b_[w]);
                //dg::blas2::symv(project_[u], x_[u], x_[w]);
                std::cout << "zeroed " << w << ", ";
                dg::blas1::scal(x_[w], 0.0);
            }
            else if (m_schemeLayout[i].m_step < 0)
            {
                //
                // correct the solution vector of the finer grid
                // x[w] = x[w] + P^{-1} x[u]
                dg::blas2::symv(1., inter_[w], x_[u], 1., x_[w]);
                //dg::blas2::symv(inter_[w], x_[u], x_[w]);
            }
            
            u = w;
		}

		x_[0].swap(x);
		return number;
	}

    /**
    * @brief Nested iterations
    *
    * - Compute residual with given initial guess. 
    * - Project residual down to the coarsest grid. 
    * - Solve equation on the coarse grid 
    * - interpolate solution up to next finer grid and repeat until the original grid is reached. 
    * @copydoc hide_symmetric_op
    * @param op Index 0 is the matrix on the original grid, 1 on the half grid, 2 on the quarter grid, ...
    * @param x (read/write) contains initial guess on input and the solution on output
    * @param b The right hand side (will be multiplied by weights)
    * @param eps the accuracy: iteration stops if \f$ ||b - Ax|| < \epsilon( ||b|| + 1) \f$ 
    * @return the number of iterations in each of the stages
    */
    template<class SymmetricOp>
    std::vector<unsigned> direct_solve( std::vector<SymmetricOp>& op, container&  x, const container& b, double eps)
    {
        dg::blas2::symv(op[0].weights(), b, b_[0]);
        // compute residual r = Wb - A x
        dg::blas2::symv(op[0], x, m_r[0]);
        dg::blas1::axpby(-1.0, m_r[0], 1.0, b_[0], m_r[0]);
        // project residual down to coarse grid
        for( unsigned u=0; u<stages_-1; u++)
            dg::blas2::gemv( interT_[u], m_r[u], m_r[u+1]);
        std::vector<unsigned> number(stages_);
        Timer t;
        
        dg::blas1::scal( x_[stages_-1], 0.0);
        //now solve residual equations
		for( unsigned u=stages_-1; u>0; u--)
        {
            t.tic();
            cg_[u].set_max(grids_[u].get().size());
            number[u] = cg_[u]( op[u], x_[u], m_r[u], op[u].precond(), op[u].inv_weights(), eps/2, 1.);
            dg::blas2::symv( inter_[u-1], x_[u], x_[u-1]);
            t.toc();
            std::cout << "stage: " << u << ", iter: " << number[u] << ", took "<<t.diff()<<"s\n";

        }
        t.tic();

        //update initial guess
        dg::blas1::axpby( 1., x_[0], 1., x);
        cg_[0].set_max(grids_[0].get().size());
        number[0] = cg_[0]( op[0], x, b_[0], op[0].precond(), op[0].inv_weights(), eps);
        t.toc();
        std::cout << "stage: " << 0 << ", iter: " << number[0] << ", took "<<t.diff()<<"s\n";
        
        return number;
    }

    /**
    * @brief Project vector to all involved grids
    * @param src the input vector (may alias first element of out)
    * @param out the input vector projected to all grids ( index 0 contains a copy of src, 1 is the projetion to the first coarse grid, 2 is the next coarser grid, ...)
    */
    void project( const container& src, std::vector<container>& out)
    {
        dg::blas1::copy( src, out[0]);
        for( unsigned u=0; u<grids_.size()-1; u++)
            dg::blas2::gemv( project_[u], out[u], out[u+1]);
    }

    /**
    * @brief Project vector to all involved grids (allocate memory version)
    * @param src the input vector 
    * @return the input vector projected to all grids ( index 0 contains a copy of src, 1 is the projetion to the first coarse grid, 2 is the next coarser grid, ...)
    */
    std::vector<container> project( const container& src)
    {
        std::vector<container> out( grids_.size());
        for( unsigned u=0; u<grids_.size(); u++)
            dg::blas1::transfer( dg::evaluate( dg::zero, grids_[u].get()), out[u]);
        project( src, out);
        return out;

    }
    ///@return number of stages 
    unsigned stages()const{return stages_;}

    const std::vector<dg::Handle< Geometry > > grids()const { return grids_; }

private:

	void set_scheme(const int scheme_type)
	{
        assert(scheme_type <= 1 && scheme_type >= 0);
        
        // initialize one conjugate-gradient for each grid size
        for (unsigned u = 0; u < stages_; u++)
            cg_[u].construct(x_[u], 1);

        switch (scheme_type)
        {
            // from coarse to fine
        case(0):
            
            //m_mode = nestediteration;
            m_startStage = stages_ - 1;
            for (int u = stages_ - 1; u >= 0; u--)
                m_schemeLayout.push_back(stepinfo(-1, x_[u].size()));
            break;

        case(1):

            //m_mode = correctionscheme;
            m_startStage = 0;
            for (unsigned u = 0; u < stages_-1; u++)
                m_schemeLayout.push_back(stepinfo(1, 5));
            
            m_schemeLayout.push_back(stepinfo(-1, 1000));

            for (int u = stages_ - 2; u >= 0; u--)
                m_schemeLayout.push_back(stepinfo(-1, 1000));

            break;

        default:
            break;
        }

        // there is no step at the last stage so the step must be zero
        m_schemeLayout.back().m_niter = x_[0].size();
        m_schemeLayout.back().m_step = 0;

        PrintScheme();

        // checks:
        // (0) the last entry should be the stage before the original grid
        // (1) there can not be more than n-1 interpolations in succession
        
        unsigned u = m_startStage;
        assert( u <= stages_ - 1);
        for (unsigned i = 0; i < m_schemeLayout.size(); i++)
        {
            u += m_schemeLayout[i].m_step;
            assert( u <= stages_ - 1);
        }   
        assert(u == 0);
	}

    ///print scheme information to std::cout
    void PrintScheme(void)
    {
        std::cout << "Scheme: " << std::endl;
        unsigned u = m_startStage;
        for (unsigned i = 0; i < m_schemeLayout.size(); i++)
        {
            std::cout << "num " << i << ", stage: " << u << ", iterations on current stage: " << m_schemeLayout[i].m_niter << ", step direction " << m_schemeLayout[i].m_step << std::endl;
            u += m_schemeLayout[i].m_step;
        }
    }

private:
    unsigned stages_;
    std::vector< dg::Handle< Geometry> > grids_;
    std::vector< MultiMatrix<Matrix, container> >  inter_;
    std::vector< MultiMatrix<Matrix, container> >  interT_;
    std::vector< MultiMatrix<Matrix, container> >  project_;
    std::vector< CG<container> > cg_;
    std::vector< container> x_, m_r, b_; 

    struct stepinfo
    {
        stepinfo(int step, const unsigned niter) : m_step(step), m_niter(niter)
        {
        };

        int m_step; // {+1,-1}
        unsigned int m_niter;        
    };

    unsigned m_startStage;
    std::vector<stepinfo> m_schemeLayout;
};

}//namespace dg
