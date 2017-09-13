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


template< class Geometry, class Matrix, class container> 
struct MultigridCG2d
{
    MultigridCG2d( const Geometry& grid, const unsigned stages, const int scheme_type = 0, const int extrapolation_type = 2 )
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
        project_.resize( stages-1);
        
		for(unsigned u=0; u<stages-1; u++)
        {
            // Projecting from one grid to the next is the same as 
            // projecting from the original grid to the coarse grids
            project_[u] = dg::create::fast_projection(grids_[u].get(), 2, 2);
            inter_[u] = dg::create::fast_interpolation(grids_[u+1].get(), 2, 2);
        }

        dg::blas1::transfer(dg::evaluate(dg::zero, grid), x0_);
        x1_ = x0_, x2_ = x0_;
        
        x_ = project(x0_); 
        m_r = x_,
		b_ = x_;        
		set_extrapolationType(extrapolation_type);
        set_scheme(scheme_type);        
    }

	template<class SymmetricOp>
	std::vector<unsigned> solve(/*const*/ std::vector<SymmetricOp>& op, container& x, const container& b, const double eps)
	{
		dg::blas1::axpbygz(alpha[0], x0_, alpha[1], x1_, alpha[2], x2_);
		x_[0].swap(x2_);

        project(x_[0], x_);
		project(b, b_);

        //
        // define new rhs
        for (unsigned u = 0; u < stages_; u++)
            dg::blas1::pointwiseDivide(b_[u], op[u].inv_weights(), b_[u]);
        
        unsigned int numStageSteps = m_schemeLayout.size();
		std::vector<unsigned> number(numStageSteps);

        unsigned u = m_startStage;
                        
        for (unsigned i = 0; i < numStageSteps; i++)
        {
            unsigned w = u + m_schemeLayout[i].m_step;

            //
            // zero initial guess
            if (m_schemeLayout[i].m_step > 0)
            {
                std::cout << "zeroed " << w << ", ";
                dg::blas1::scal(x_[w], 0.0);
            }

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
                // compute residual r = b - A x
                dg::blas2::symv(op[u], x_[u], m_r[u]);
                dg::blas1::axpby(-1.0, m_r[u], 1.0, b_[u], m_r[u]);
                dg::blas1::pointwiseDot( m_r[u], op[u].inv_weights(), m_r[u]);

                //
                // transfer residual to the rhs of the coarser grid
                dg::blas2::symv(project_[u], m_r[u], b_[w]);
                dg::blas1::pointwiseDivide( b_[w], op[w].inv_weights(), b_[w]);

                //dg::blas2::symv(project_[u], x_[u], x_[w]);
            }
            else if (m_schemeLayout[i].m_step < 0)
            {
                //
                // correct the solution vector of the finer grid
                // x[w] = x[w] + P^{-1} x[u]
                dg::blas2::symv(inter_[w], x_[u], m_r[w]);
                dg::blas1::axpby(1.0, x_[w], 1.0, m_r[w], x_[w]);

                //dg::blas2::symv(inter_[w], x_[u], x_[w]);
            }
            
            u = w;
		}

		x_[0].swap(x);
		x1_.swap(x2_);
		x0_.swap(x1_);

		blas1::copy(x, x0_);
		return number;
	}

    template<class SymmetricOp>
    std::vector<unsigned> direct_solve( std::vector<SymmetricOp>& op, container&  x, const container& b, double eps)
    {
        Timer t;
        t.tic();
        //compute initial guess
        dg::blas1::axpbygz( alpha[0], x0_, alpha[1], x1_, alpha[2], x2_); 
        
		x_[0].swap(x2_);
        dg::blas1::copy( x_[0], x);//save initial guess

        //dg::blas1::scal( x_[0], 0.0);
		//project( x_[0], x_);
        project( b, b_);
        //project( b, b_);
        // compute residual r = b - A x
        dg::blas2::symv(op[0], x_[0], m_r[0]);
        dg::blas1::pointwiseDot( m_r[0], op[0].inv_weights(), m_r[0]);
        dg::blas1::axpby(-1.0, m_r[0], 1.0, b_[0], b_[0]);
        project( b_[0], b_);
        dg::blas1::scal( x_[0], 0.0);
		project( x_[0], x_);
        std::vector<unsigned> number(stages_);
        
        //now solve residual equations
		for( unsigned u=stages_-1; u>0; u--)
        {
            cg_[u].set_max(grids_[u].get().size());
            dg::blas1::pointwiseDivide( b_[u], op[u].inv_weights(), b_[u]);
            number[u] = cg_[u]( op[u], x_[u], b_[u], op[u].precond(), op[u].inv_weights(), eps/2, 1.);
            dg::blas2::symv( inter_[u-1], x_[u], x_[u-1]);
            std::cout << "stage: " << u << ", max iter: " << cg_[u].get_max() << ", iter: " << number[u] << std::endl;
        }

		dg::blas1::pointwiseDivide( b_[0], op[0].inv_weights(), b_[0]);
        cg_[0].set_max(grids_[0].get().size());
        number[0] = cg_[0]( op[0], x_[0], b_[0], op[0].precond(), op[0].inv_weights(), eps);
        std::cout << "stage: " << 0 << ", max iter: " << cg_[0].get_max() << ", iter: " << number[0] << std::endl;
        //update initial guess
        dg::blas1::axpby( 1., x_[0], 1., x);

        
		//x_[0].swap( x);
        x1_.swap( x2_);
        x0_.swap( x1_);
        
        blas1::copy( x, x0_);
        t.toc();
        std::cout<< "Took "<<t.diff()<<"s\n";
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
                m_schemeLayout.push_back(stepinfo(1, 50));
            
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
        assert(u >= 0 && u <= stages_ - 1);
        for (unsigned i = 0; i < m_schemeLayout.size(); i++)
        {
            u += m_schemeLayout[i].m_step;
            assert(u >= 0 && u <= stages_ - 1);
        }   
        assert(u == 0);
	}

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
    std::vector< MultiMatrix<Matrix, container> >  project_;
    std::vector< CG<container> > cg_;
    std::vector< container> x_, m_r, b_; 
    container x0_, x1_, x2_;
    double alpha[3];

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
