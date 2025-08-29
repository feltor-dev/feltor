#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"

namespace feltor
{
namespace routines{


// result = Sum_i v_i w_i
template<class Container>
void dot( const std::array<Container, 3>& v,
          const std::array<Container, 3>& w,
          Container& result)
{
    dg::blas1::evaluate( result, dg::equals(), dg::PairSum(),
        v[0], w[0], v[1], w[1], v[2], w[2]);
}

// result = alpha * Sum_i v_i w_i + beta*result
template<class Container>
void dot( double alpha, const std::array<Container, 3>& v,
          const std::array<Container, 3>& w, double beta,
          Container& result)
{
    dg::blas1::evaluate( result, dg::Axpby(alpha,beta), dg::PairSum(),
        v[0], w[0], v[1], w[1], v[2], w[2]);
}


struct Dot{
    DG_DEVICE void operator()(
            double lambda,
        double d0P, double d1P, double d2P,
        double& c_0, double& c_1, double& c_2)
    {
        c_0 = lambda*(d0P);
        c_1 = lambda*(d1P);
        c_2 = lambda*(d2P);
    }
};
// c_i = lambda*a_i
template<class Container>
void scal( const Container& lambda,
          const std::array<Container, 3>& a,
          std::array<Container, 3>& c)
{
    dg::blas1::subroutine( Dot(), lambda,
        a[0], a[1], a[2], c[0], c[1], c[2]);
}

struct Times{
    DG_DEVICE void operator()(
        double d0P, double d1P, double d2P, //any three vectors
        double d0S, double d1S, double d2S,
        double& c_0, double& c_1, double& c_2)
    {
        c_0 = (d1P*d2S-d2P*d1S);
        c_1 = (d2P*d0S-d0P*d2S);
        c_2 = (d0P*d1S-d1P*d0S);
    }
};

// Vec c = ( Vec a x Vec b)
template<class Container>
void times(
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          std::array<Container, 3>& c)
{
    dg::blas1::subroutine( Times(),
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}

struct Jacobian{
    DG_DEVICE double operator()(
        double d0P, double d1P, double d2P, //any three vectors
        double d0S, double d1S, double d2S,
        double b_0, double b_1, double b_2)
    {
        return      b_0*( d1P*d2S-d2P*d1S)+
                    b_1*( d2P*d0S-d0P*d2S)+
                    b_2*( d0P*d1S-d1P*d0S);
    }
};

// result = Vec c Cdot ( Vec a x Vec b)
template<class Container>
void jacobian(
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          const std::array<Container, 3>& c,
          Container& result)
{
    dg::blas1::evaluate( result, dg::equals(), Jacobian(),
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
// result = alpha * Vec c Cdot ( Vec a x Vec b) + beta * result
template<class Container>
void jacobian(
        double alpha,
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          const std::array<Container, 3>& c,
          double beta,
          Container& result)
{
    dg::blas1::evaluate( result, dg::Axpby(alpha,beta), Jacobian(),
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
}//namespace routines


// generate a Curvilinear flux aligned grid
// config contains configuration parameters from input file
// mag is the magnetic field
// psipO is psi value at O-point (write only)
// psipmax is maximum psi value based on psipO and fx_0 from config file (write only)
// f0 is linear grid constant (write only)
dg::geo::CurvilinearGrid2d generate_XGrid( dg::file::WrappedJsonValue config,
    const dg::geo::TokamakMagneticField& mag, double& psipO, double& psipmax, double& f0)
{
    //we use so many Neta so that we get close to the X-point
    unsigned npsi = config.get("n",3).asUInt();
    unsigned Npsi = config.get("Npsi", 64).asUInt();
    unsigned Neta = config.get("Neta", 640).asUInt();
    std::cout << "Using X-point grid resolution (n("<<npsi<<"), Npsi("<<Npsi<<"), Neta("<<Neta<<"))\n";
    double RO = mag.R0(), ZO = 0;
    int point = dg::geo::findOpoint( mag.get_psip(), RO, ZO);
    psipO = mag.psip()(RO, ZO);
    std::cout << "O-point found at "<<RO<<" "<<ZO
              <<" with Psip "<<psipO<<std::endl;
    if( point == 1 )
        std::cout << " (minimum)"<<std::endl;
    if( point == 2 )
        std::cout << " (maximum)"<<std::endl;
    double fx_0 = config.get( "fx_0", 1./8.).asDouble(); //must evenly divide Npsi
    psipmax = -fx_0/(1.-fx_0)*psipO;
    std::cout << "psi outer in g1d_out is "<<psipmax<<"\n";
    std::cout << "Generate orthogonal flux-aligned grid ... \n";
    std::unique_ptr<dg::geo::aGenerator2d> generator;
    if( !(mag.params().getDescription() == dg::geo::description::standardX))
        generator = std::make_unique<dg::geo::SimpleOrthogonal>(
            mag.get_psip(),
            psipO<psipmax ? psipO : psipmax,
            psipO<psipmax ? psipmax : psipO,
            mag.R0() + 0.1*mag.params().a(), 0., 0.1*psipO, 1);
    else
    {

        double RX = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
        double ZX = -1.1*mag.params().elongation()*mag.params().a();
        dg::geo::findXpoint( mag.get_psip(), RX, ZX);
        const double psipX = mag.psip()( RX, ZX);
        std::cout << "X-point set at "<<RX<<" "<<ZX<<" with Psi_p = "<<psipX<<"\n";
        dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( mag.get_psip(), RX, ZX) ;
        generator = std::make_unique<dg::geo::SeparatrixOrthogonalAdaptor>(
            mag.get_psip(), monitor_chi,
            //psipO<psipmax ? psipO : psipmax,
            psipO, // must be the closed flux surface (is the sign of f0 still valid?)
            RX, ZX, mag.R0(), 0, 1, false, fx_0);
    }
    dg::geo::CurvilinearGrid2d gridX2d(*generator,
            npsi, Npsi, Neta, dg::DIR, dg::PER);
    std::cout << "DONE!\n";
    //f0 makes a - sign if psipmax < psipO
    f0 = ( gridX2d.x1() - gridX2d.x0() ) / ( psipmax - psipO );
    return gridX2d;
}


/// generate list of 2d flux grid file outputs
std::vector<std::tuple<std::string, dg::HVec, std::string> >
    compute_twoflux_labels( const dg::geo::CurvilinearGrid2d& gridX2d)
{
    std::vector<std::tuple<std::string, dg::HVec, std::string> > map2d;
    std::vector<dg::HVec > coordsX = gridX2d.map();
    map2d.emplace_back( "xc", coordsX[0],
        "x-coordinate in Cartesian coordinate system of FSA-grid");
    map2d.emplace_back( "yc", coordsX[1],
        "y-coordinate in Cartesian coordinate system of FSA-grid");
    map2d.emplace_back( "vol", dg::create::volume( gridX2d),
        "2d Volume form (including dG weights) of FSA-grid. Multiply by 2*Pi*xc to get 3d volume form");
    return map2d;
}

/// generate list of 1d flux grid file outputs
/// ------------------- Compute 1d flux labels ---------------------//
std::vector<std::tuple<std::string, dg::HVec, std::string> >
    compute_oneflux_labels(
    dg::Average<dg::IHMatrix, dg::HVec>& poloidal_average,
    const dg::geo::CurvilinearGrid2d& gridX2d,
    const dg::geo::TokamakMagneticField& mod_mag,
    double psipO, double psipmax, double f0,
    dg::HVec& dvdpsip, dg::HVec& volX2d,
    dg::Grid1d& g1d_out, // Psip grid
    dg::Grid1d& g1d_out_eta // Eta grid
    )
{
    unsigned npsi = gridX2d.n();
    unsigned Npsi = gridX2d.Nx();
    unsigned Neta = gridX2d.Ny();
    std::vector<dg::HVec > coordsX = gridX2d.map();
    std::vector<std::tuple<std::string, dg::HVec, std::string> > map1d;
    /// Compute flux volume label
    dg::SparseTensor<dg::HVec> metricX = gridX2d.metric();
    volX2d = dg::tensor::volume2d( metricX);
    poloidal_average( volX2d, dvdpsip, false);
    //O-point fsa value is always 0 (hence the DIR boundary condition)
    g1d_out = dg::Grid1d(psipO<psipmax ? psipO : psipmax,
                       psipO<psipmax ? psipmax : psipO,
                       npsi, Npsi, psipO < psipmax ? dg::DIR_NEU : dg::NEU_DIR);
    g1d_out_eta = dg::Grid1d(gridX2d.y0(), gridX2d.y1(), npsi, Neta, dg::DIR_NEU);
    /// 1D grid for the eta (poloidal) directions instead of psi for the radial cut
    dg::blas1::scal( dvdpsip, 2.*M_PI*f0);
    map1d.emplace_back( "dv2ddpsi", dvdpsip,
        "Derivative of 2d flux volume (=area) with respect to flux label psi");
    dg::direction integration_dir = psipO < psipmax ? dg::forward : dg::backward;
    dg::HVec X_psi_vol = dg::integrate( dvdpsip, g1d_out, integration_dir);
    map1d.emplace_back( "psi_vol2d", X_psi_vol,
        "2d Flux volume (area) evaluated with X-point grid");
    dg::blas1::pointwiseDot( coordsX[0], volX2d, volX2d); //R\sqrt{g}
    poloidal_average( volX2d, dvdpsip, false);
    dg::blas1::scal( dvdpsip, 4.*M_PI*M_PI*f0);
    map1d.emplace_back( "dvdpsi", dvdpsip,
        "Derivative of flux volume with respect to flux label psi");
    X_psi_vol = dg::integrate( dvdpsip, g1d_out, integration_dir);
    map1d.emplace_back( "psi_vol", X_psi_vol,
        "Flux volume evaluated with X-point grid");

    /// Compute flux area label
    dg::HVec gradZetaX = metricX.value(0,0), X_psi_area;
    dg::blas1::transform( gradZetaX, gradZetaX, dg::SQRT<double>());
    dg::blas1::pointwiseDot( volX2d, gradZetaX, gradZetaX); //R\sqrt{g}|\nabla\zeta|
    poloidal_average( gradZetaX, X_psi_area, false);
    dg::blas1::scal( X_psi_area, 4.*M_PI*M_PI);
    map1d.emplace_back( "psi_area", X_psi_area,
        "Flux area evaluated with X-point grid");
    dg::blas1::pointwiseDivide( gradZetaX, coordsX[0], gradZetaX); //R\sqrt{g}|\nabla\zeta|
    poloidal_average( gradZetaX, X_psi_area, false);
    dg::blas1::scal( X_psi_area, 2.*M_PI);
    map1d.emplace_back( "psi_arc", X_psi_area,
        "Psip arc length evaluated with X-point grid");

    dg::HVec rho = dg::evaluate( dg::cooX1d, g1d_out);
    dg::blas1::axpby( -1./psipO, rho, +1., 1., rho); //transform psi to rho
    map1d.emplace_back("rho", rho,
        "Alternative flux label rho = 1-psi/psimin");
    dg::blas1::transform( rho, rho, dg::SQRT<double>());
    map1d.emplace_back("rho_p", rho,
        "Alternative flux label rho_p = sqrt(1-psi/psimin)");
    dg::geo::SafetyFactor qprof( mod_mag);
    dg::HVec psi_vals = dg::evaluate( dg::cooX1d, g1d_out);
    // we need to avoid calling SafetyFactor outside closed fieldlines
    dg::blas1::subroutine( [psipO]( double& psi){
           if( (psipO < 0 && psi > 0) || (psipO>0 && psi <0))
               psi = psipO/2.; // just use a random value
        }, psi_vals);
    dg::HVec qprofile( psi_vals);
    dg::blas1::evaluate( qprofile, dg::equals(), qprof, psi_vals);
    map1d.emplace_back("q-profile", qprofile,
        "q-profile (Safety factor) using direct integration");
    map1d.emplace_back("psi_psi",    dg::evaluate( dg::cooX1d, g1d_out),
        "Poloidal flux label psi (same as coordinate)");
    dg::HVec psit = dg::integrate( qprofile, g1d_out, integration_dir);
    //std::cout << "q-pfo "<<qprofile[10]<<"\n";
    //std::cout << "Psi_t "<<psit[10]<<"\n";
    map1d.emplace_back("psit1d", psit,
        "Toroidal flux label psi_t integrated using q-profile");
    //we need to avoid integrating >=0 for total psi_t
    dg::Grid1d g1d_fine(psipO<0. ? psipO : 0., psipO<0. ? 0. : psipO, npsi
            ,Npsi,dg::DIR_NEU);
    qprofile = dg::evaluate( qprof, g1d_fine);
    dg::HVec w1d = dg::create::weights( g1d_fine);
    double psit_tot = dg::blas1::dot( w1d, qprofile);
    if( integration_dir == dg::backward)
        psit_tot *= -1;
    //std::cout << "q-pfo "<<qprofile[10]<<"\n";
    //std::cout << "Psi_t "<<psit[10]<<"\n";
    //std::cout << "total "<<psit_tot<<"\n";
    dg::blas1::scal ( psit, 1./psit_tot);
    dg::blas1::transform( psit, psit, dg::SQRT<double>());
    map1d.emplace_back("rho_t", psit,
        "Toroidal flux label rho_t = sqrt( psit/psit_tot)");
    return map1d;
}

}//namespace feltor
namespace common
{
#ifdef WITH_MPI
// A signal handler is called if the os sends a signal to the process!
// I am currently not sure what advantage we have from using the following or if it will
// always work
// ATTENTION: in slurm should be used with --signal=SIGINT@30 (<signal>@<time in seconds>)
void sigterm_handler(int signal)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    std::cout << " pid "<<rank<<" sigterm_handler, got signal " << signal << std::endl;
    MPI_Finalize();
    exit(signal);
}
#endif //WITH_MPI

template<class Parameters>
void parse_input_file( int argc, char* argv[], dg::file::WrappedJsonValue& js)
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
    if( argc != 2 && argc != 3 && argc != 4)
    {
        DG_RANK0 std::cerr << "ERROR: Wrong number of arguments!\nUsage: "
                << argv[0]<<" [input.json] \n OR \n"
                << argv[0]<<" [input.json] [output.nc]\n OR \n"
                << argv[0]<<" [input.json] [output.nc] [initial.nc] "<<std::endl;
        dg::abort_program();
    }
    try{
        js = dg::file::file2Json( argv[1],
                dg::file::comments::are_discarded, dg::file::error::is_throw);
        Parameters p( js);
    } catch( std::exception& e) {
        DG_RANK0 std::cerr << "ERROR in input file "<<argv[1]<<std::endl;
        DG_RANK0 std::cerr << e.what()<<std::endl;
        dg::abort_program();
    }
}

void parse_geometry_file( std::string argv1, dg::file::WrappedJsonValue& js)
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
    std::string geometry_params = js["magnetic_field"]["input"].asString();
    if( geometry_params == "file")
    {
        std::string path = js["magnetic_field"]["file"].asString();
        double scale = 1;
        try{
            scale = js["magnetic_field"]["scale"].asDouble();
        } catch( std::exception& e) {
            DG_RANK0 std::cerr << "scale needs to be present in input file "<<argv1<<" if magnetic field from file\n";
            DG_RANK0 std::cerr << e.what()<<std::endl;
            dg::abort_program();
        }
        try{
            js.asJson()["magnetic_field"]["params"] = dg::file::file2Json( path,
                    dg::file::comments::are_discarded, dg::file::error::is_throw);
            // convert unit to rhos
            double R0 = js["magnetic_field"]["params"]["R_0"].asDouble();
            js.asJson()["magnetic_field"]["params"]["R_0"] = R0*scale;
        }catch(std::runtime_error& e)
        {
            DG_RANK0 std::cerr << "ERROR in geometry file "<<path<<std::endl;
            DG_RANK0 std::cerr << e.what()<<std::endl;
            dg::abort_program();
        }
    }
    else if( geometry_params != "params")
    {
        DG_RANK0 std::cerr << "Error: Unknown magnetic field input '"
                           << geometry_params<<"'. Exit now!\n";
        dg::abort_program();
    }
}

// this may even go into dg::geo
std::map<std::string,double> box( const dg::file::WrappedJsonValue& js)
{
    double boxscaleRm, boxscaleRp;
    boxscaleRm  = js["grid"][ "scaleR"].get( 0u, 1.05).asDouble();
    boxscaleRp  = js["grid"][ "scaleR"].get( 1u, 1.05).asDouble();
    // easiest way to get correct a for all field types:
    dg::geo::TokamakMagneticField mag =
        dg::geo::createMagneticField(js["magnetic_field"]["params"]);
    const double Rmin=mag.R0()-boxscaleRm*mag.params().a();
    const double Rmax=mag.R0()+boxscaleRp*mag.params().a();
    double boxscaleZm, boxscaleZp;
    boxscaleZm  = js["grid"][ "scaleZ"].get( 0u, 1.05).asDouble();
    boxscaleZp  = js["grid"][ "scaleZ"].get( 1u, 1.05).asDouble();
    const double Zmin=-boxscaleZm*mag.params().a();
    const double Zmax= boxscaleZp*mag.params().a();

    return std::map<std::string, double>{
        {"Rmin", Rmin},{"Rmax",Rmax}, {"Zmin", Zmin},{"Zmax",Zmax}
    };
}

void create_mag_wall(
        const std::string argv1,
        const dg::file::WrappedJsonValue& js,
        dg::geo::TokamakMagneticField& mag,
        dg::geo::TokamakMagneticField& mod_mag,
        dg::geo::TokamakMagneticField& unmod_mag,
        dg::geo::CylindricalFunctor& wall,
        dg::geo::CylindricalFunctor& transition
        )
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
    bool periodify = false, modify_B = false;
    std::map<std::string,double> box;
    try{
        mag = unmod_mag = dg::geo::createMagneticField(js["magnetic_field"]["params"]);
        mod_mag = dg::geo::createModifiedField(js["magnetic_field"]["params"],
                js["boundary"]["wall"], wall, transition);
        periodify   = js["FCI"].get( "periodify", true).asBool();
        modify_B = js["boundary"]["wall"].get( "modify-B", false).asBool();
        box = common::box( js);
    }catch(std::runtime_error& e)
    {
        DG_RANK0 std::cerr << "ERROR in input file "<<argv1<<std::endl;
        DG_RANK0 std::cerr <<e.what()<<std::endl;
        dg::abort_program();
    }
    if( periodify)
    {
        unmod_mag = dg::geo::periodify( unmod_mag, box.at("Rmin"),
                box.at("Rmax"), box.at("Zmin"), box.at("Zmax"), dg::NEU,
                dg::NEU);
        mod_mag = dg::geo::periodify( mod_mag, box.at("Rmin"),
                box.at("Rmax"), box.at("Zmin"), box.at("Zmax"), dg::NEU,
                dg::NEU);
    }
    if( modify_B)
        mag = mod_mag;
    else
        mag = unmod_mag;
}

#ifdef WITH_MPI
void check_Nz( unsigned Nz, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    if( dims[2] >= (int)Nz)
    {
        DG_RANK0 std::cerr << "ERROR: Number of processes in z "<<dims[2]
                    <<" may not be larger or equal Nz "<<Nz<<std::endl;
        dg::abort_program();
    }
}
#endif //WITH_MPI

template<class Geometry, class Equations>
void create_and_set_sheath(
        const std::string argv1,
        const dg::file::WrappedJsonValue& js,
        const dg::geo::TokamakMagneticField& mag, // after modification...
        const dg::geo::CylindricalFunctor& wall,
        dg::geo::CylindricalFunctor& sheath,
        dg::geo::CylindricalFunctor& sheath_coordinate,
        const Geometry& grid,
        Equations& feltor
        )
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
    dg::Timer t;
    t.tic();
    DG_RANK0 std::cout << "# Compute Sheath coordinates \n";
    double sheath_rate = 0.;
    try{
        auto box = common::box( js);
        dg::Grid2d sheath_walls( box.at("Rmin"), box.at("Rmax"),
                box.at("Zmin"), box.at("Zmax"), 1, 1, 1);
        dg::geo::createSheathRegion( js["boundary"]["sheath"],
            dg::geo::createMagneticField(js["magnetic_field"]["params"]),
            wall, sheath_walls, sheath);
        double sheath_max_angle = js["boundary"]["sheath"].get( "max_angle",
                4).asDouble()*2.*M_PI;
        std::string sheath_coord = js["boundary"]["sheath"].get( "coordinate",
                "s").asString();
        // sheath is created on feltor magnetic field
        sheath_coordinate = dg::geo::WallFieldlineCoordinate(
                dg::geo::createBHat( mag), sheath_walls,
                sheath_max_angle, 1e-6, sheath_coord, dg::geo::mod::SOLRegion( mag, wall));
        sheath_rate = js ["boundary"]["sheath"].get( "penalization",
                0.).asDouble();
    }catch(std::runtime_error& e)
    {
        DG_RANK0 std::cerr << "ERROR in input file "<<argv1<<std::endl;
        DG_RANK0 std::cerr <<e.what()<<std::endl;
        dg::abort_program();
    }
    std::unique_ptr<dg::x::aGeometry2d> perp_grid_ptr( grid.perp_grid());
    dg::x::HVec coord2d = dg::pullback( sheath_coordinate, *perp_grid_ptr);
    dg::x::DVec coord3d;
    dg::assign3dfrom2d( coord2d, coord3d, grid);
    feltor.set_sheath(
            sheath_rate,
            dg::construct<dg::x::DVec>(dg::pullback( sheath, grid)),
            coord3d);
    t.toc();
    DG_RANK0 std::cout << "# ... took  "<<t.diff()<<"s\n";
}
template<class Container>
void multiply_rhs_penalization(
        Container& yp, bool penalize_wall, const Container& m_wall,
        bool penalize_sheath, const Container& m_sheath)
{
    //mask right hand side in penalization region
    if( penalize_wall && penalize_sheath)
    {
        dg::blas1::subroutine( []DG_DEVICE(
            double& rhs, double wall, double sheath){
                rhs *= (1.0-wall-sheath);
            }, yp, m_wall, m_sheath);
    }
    else if( penalize_wall)
    {
        dg::blas1::subroutine( []DG_DEVICE( double& rhs, double wall){
                rhs *= (1.0-wall); }, yp, m_wall);
    }
    else if( penalize_sheath)
    {
        dg::blas1::subroutine( []DG_DEVICE( double& rhs, double sheath){
                rhs *= (1.0-sheath); }, yp, m_sheath);
    }
}


template<class Vector, class Explicit>
std::unique_ptr<dg::aTimeloop<Vector>> init_timestepper(
    const dg::file::WrappedJsonValue& js, Explicit& feltor, double time, const Vector& y0,
    bool& adaptive, unsigned& nfailed)
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
    DG_RANK0 std::cout << "# Initialize Timestepper" << std::endl;
    auto odeint = std::unique_ptr<dg::aTimeloop<Vector>>();
    std::string tableau     = js["timestepper"].get("tableau", "TVB-3-3").asString();
    std::string timestepper = js["timestepper"].get("type", "multistep").asString();
    if( timestepper == "multistep")
    {
        double dt = js[ "timestepper"]["dt"].asDouble( 0.01);
        odeint = std::make_unique<dg::MultistepTimeloop<Vector>>(
            dg::ExplicitMultistep<Vector>(tableau,y0), feltor, time, y0, dt);
    }
    else if (timestepper == "adaptive")
    {
        //adapt.stepper().ignore_fsal();
        double rtol = js[ "timestepper"][ "rtol"].asDouble( 1e-7);
        double atol = js[ "timestepper"][ "atol"].asDouble( 1e-10);
        double reject_limit = js["timestepper"].get("reject-limit", 2).asDouble();
        auto step = [=, &feltor, &nfailed, adapt = dg::Adaptive<dg::ERKStep<Vector>>(tableau, y0) ](
        auto t0, auto y0, auto& t, auto& y, auto& dt) mutable
        {
            adapt.step( feltor, t0, y0, t, y, dt, dg::pid_control, dg::l2norm,
                    rtol, atol, reject_limit);
            // do more things here ...
            if ( adapt.failed() )
                nfailed ++;
        };
        odeint = std::make_unique<dg::AdaptiveTimeloop<Vector>>(step);
        adaptive = true;
    }
    else
    {
        DG_RANK0 std::cerr << "Error: Unrecognized timestepper: '"
                           << timestepper << "'! Exit now!\n";
        dg::abort_program();
    }
    DG_RANK0 std::cout << "# Done!\n";
    return odeint;

}


}//namespace common
