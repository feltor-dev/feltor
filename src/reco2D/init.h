#pragma once


namespace asela{
///The purpose of this file is to provide an interface for custom initial
///conditions and /source profiles.  Just add your own to the relevant map
///below.
std::map<std::string, std::function< std::array<std::array<dg::x::DVec,2>,2>(
    Asela<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& f,
    const dg::x::CartesianGrid2d& grid, const asela::Parameters& p,
    const Json::Value& js, enum dg::file::error mode)
> > initial_conditions =
{
    { "zero",
        []( Asela<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& f,
            const dg::x::CartesianGrid2d& grid, const asela::Parameters& p,
            const Json::Value& js, enum dg::file::error mode)
        {
            std::array<std::array<dg::x::DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] =
                dg::construct<dg::x::DVec>(dg::evaluate( dg::zero, grid));
            return y0;
        }
    },
    { "harris",
        []( Asela<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& f,
            const dg::x::CartesianGrid2d& grid, const asela::Parameters& p,
            const Json::Value& js, enum dg::file::error mode)
        {
            double amp0 = dg::file::get( mode, js, "init", "amplitude0", 1e-1).asDouble();
            double amp1 = dg::file::get( mode, js, "init", "amplitude1", 1e-5).asDouble();
            double mY  = dg::file::get( mode, js, "init", "my", 1).asDouble();
            std::array<std::array<dg::x::DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] =
                dg::construct<dg::x::DVec>(dg::evaluate( dg::zero, grid));
            double kx = 2.*M_PI/p.lxhalf, ky = mY*M_PI/p.lyhalf;
            double kxp = M_PI/p.lxhalf/2.;
            dg::x::DVec apar = dg::evaluate( [=](double x, double y){ return
                cos(kxp*x)*( amp0/cosh(kx*x)/cosh(kx*x) - amp1*cos( ky*y));
                }, grid);
            //cos 
            f.compute_lapM( -1./p.beta, apar, 0., y0[1][0]);
            dg::blas1::axpby(1., y0[1][0], 1./p.mu[0], apar, y0[1][0]);
            // if we include Gamma A again we must take care here
            dg::blas1::axpby(1./p.mu[1], apar, 0.0, y0[1][1]);
            return y0;
        }
    },
    { "island",
        []( Asela<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& f,
            const dg::x::CartesianGrid2d& grid, const asela::Parameters& p,
            const Json::Value& js, enum dg::file::error mode)
        {
            double amp1 = dg::file::get( mode, js, "init", "amplitude1", 1e-5).asDouble();
            double mY  = dg::file::get( mode, js, "init", "my", 1).asDouble();
            std::array<std::array<dg::x::DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] =
                dg::construct<dg::x::DVec>(dg::evaluate( dg::zero, grid));
            double kx = 2.*M_PI/p.lxhalf, ky = mY*M_PI/p.lyhalf;
            double kxp = M_PI/p.lxhalf/2.;
            dg::x::DVec apar = dg::evaluate( [=](double x, double y){ return
                cos(kxp*x)*( 1./kx*log( cosh(kx*x) + 0.2*cos( kx*y) ) - amp1*cos( ky*y));
                }, grid);
            f.compute_lapM( -1./p.beta, apar, 0., y0[1][0]);
            dg::blas1::axpby(1., y0[1][0], 1./p.mu[0], apar, y0[1][0]);
            // if we include Gamma A again we must take care here
            dg::blas1::axpby(1./p.mu[1], apar, 0.0, y0[1][1]);
            return y0;
        }
    }
};
}//namespace asela
