#error Documentation only
/*! @namespace dg 
 * @brief This is the namespace for all functions and 
 * classes defined and used by the discontinuous Galerkin solvers.
 */
/*! 
 * @defgroup backend Level 1: Vectors, Matrices and basic operations
 * @{
 *     @defgroup blas Basic Linear Algebra Subroutines
 *
 *         These routines form the heart of our container free numerical algorithms. 
 *         They are called by all our numerical algorithms like conjugate gradient or 
 *         time integrators.
 *     @{
 *         @defgroup blas1 BLAS level 1 routines
 *             This group contains Vector-Vector operations.
 *             Successive calls to blas routines are executed sequentially.
 *             A manual synchronization of threads or devices is never needed in an application 
 *             using these functions. All functions returning a value block until the value is ready.
 *         @defgroup blas2 BLAS level 2 routines
 *             This group contains Matrix-Vector operations.
 *             Successive calls to blas routines are executed sequentially.
 *             A manual synchronization of threads or devices is never needed in an application 
 *             using these functions. All functions returning a value block until the value is ready.
 *     @}
 *     @defgroup sparsematrix Sparse matrix formats
 *     @defgroup mpi_structures MPI backend functionality
 *     @defgroup typedefs Typedefs
       Useful type definitions for easy programming
 * @}
 * @defgroup numerical0 Level 2: Basic numerical algorithms
 * These algorithms make use only of blas level 1 and 2 functions
 * @{
 *     @defgroup time Time integrators
 *     @defgroup invert Matrix inversion
 *     @defgroup root Root finding
 * @}
 * @defgroup geo Level 3: Topology and Geometry
 * @{
 *     @defgroup grid Topological grids and operations
 *
 *     Objects that store topological information (which point is neighbour of which other point) 
 *     about the grid. 
 *     @{
 *         @defgroup evaluation Function discretization
 *             
 *             The function discretisation routines compute the DG discretisation
 *             of analytic functions on a given grid. In 1D the discretisation
 *             simply consists of n function values per grid cell ( where n is the number
 *             of Legendre coefficients used; currently 1, 2, 3, 4 or 5) evaluated at
 *             the Gaussian abscissas in the respective cell. In 2D and 3D we simply 
 *             use the product space. We choose x to be the contiguous direction.
 *             The first elements of the resulting vector lie in the cell at (x0,y0) and the last
 *             in (x1, y1).
 *         @defgroup lowlevel Lowlevel helper functions and classes
 *             Low level helper routines.
 *         @defgroup highlevel Weight functions
 *         @defgroup creation Discrete derivatives 
 *
 *             High level matrix creation functions
           @defgroup interpolation Interpolation and projection
 *         @defgroup scatter Scatter
 *     @}
 *     @defgroup geometry Geometric grids and operations
 *
          These routines form the heart of our geometry free numerical algorithms. 
          They are called by our geometric operators like the Poisson bracket. 
      @{
          @defgroup basicgrids Basic grids
 *        @defgroup utilities Fieldalignment and Averaging
 *            The classes to perform field line integration for DS and averaging classes
      @}
 * @}
 * @defgroup numerical1 Level 4: Advanced numerical schemes
 *
 * These routines make use of both the basic operations as well as the interfaces defined in the Geometry section.
 * @{
 *     @defgroup arakawa Discretization of Poisson bracket
 *     @defgroup matrixoperators Elliptic and Helmholtz operators
 *     @defgroup fieldaligned Fieldaligned derivatives
 * @}
 * @defgroup templates Level 99: Template models
   Documentation for template models
 * @defgroup misc Level 00: Miscellaneous additions
   @{
 *     @defgroup functions Functions and Functors
 * 
 *         The functions are useful mainly in the constructor of Operator objects. 
 *         The functors are useful for either vector transformations or
 *         as init functions in the evaluate routines.
       @defgroup timer Timer class
   @}
 * 
 */
/*! @mainpage
 * Welcome to the DG library. 
 *
 * @par Design principles
 *
 * The DG library is built on top of the <a href="https://thrust.github.io/">thrust</a> and <a href="http://cusplibrary.github.io/index.html">cusp</a> libraries. 
 * Its intention is to provide easy to use
 * functions and objects needed for the integration of 2D and 3D partial differential equations discretized with a
 * discontinuous galerkin method.  
 * Since it is built on top of <a href="https://thrust.github.io/">thrust</a> and <a href="http://cusplibrary.github.io/index.html">cusp</a>, code can run on a CPU as well as a GPU by simply 
 * switching between thrust's host_vector and device_vector. 
 * The DG library uses a design pattern also employed in the cusp library and other modern C++ codes. 
 * It might be referred to as <a href="http://dx.doi.org/10.1063/1.168674">container-free numerical algorithms</a>. 
 *
 *
 *
 */

/**
 * @brief Struct that performs collective scatter and gather operations across processes
 * on distributed vectors using mpi
 *
 * @ingroup templates
 @attention this is not a real class it's there for documentation only
 @attention parameter names can be different
 *
 * @code
 int i = myrank;
 double values[10] = {i,i,i,i, 9,9,9,9};
 thrust::host_vector<double> hvalues( values, values+10);
 int pids[10] =      {0,1,2,3, 0,1,2,3};
 thrust::host_vector<int> hpids( pids, pids+10);
 BijectiveComm coll( hpids, MPI_COMM_WORLD);
 thrust::host_vector<double> hrecv = coll.scatter( hvalues);
 //hrecv is now {0,9,1,9,2,9,3,9} e.g. for process 0 
 thrust::host_vector<double> hrecv2( coll.send_size());
 coll.gather( hrecv, hrecv2);
 //hrecv2 now equals hvalues independent of process rank
 @endcode
 */
struct aCommunicator
{

    /**
     * @brief Scatters data according to a specific scheme given in the Constructor
     *
     * The order of the received elements is according to their original array index (i.e. a[0] appears before a[1]) and their process rank of origin ( i.e. values from rank 0 appear before values from rank 1)
     * @param values data to send (must have the size given 
     * by the map in the constructor, s.a. send_size())
     * @tparam LocalContainer a container on a shared memory system
     *
     * @return received data from other processes of size recv_size()
     * @note a scatter followed by a gather of the received values restores the original array
     * @note this function is only needed in the RowDistMat matrix format
     */
    template< class LocalContainer>
    LocalContainer collect( const LocalContainer& values)const;

    /**
     * @brief Gather data according to the map given in the constructor 
     *
     * This method is the inverse of scatter 
     * @tparam LocalContainer a container on a shared memory system
     * @param gatherFrom other processes collect data from this vector (has to be of size given by recv_size())
     * @param values contains values from other processes sent back to the origin (must have the size of the map given in the constructor, or send_size())
     * @note a scatter followed by a gather of the received values restores the original array
     * @note this format is only needed in the ColDistMat matrix format
     */
    template< class LocalContainer>
    void send_and_reduce( const LocalContainer& gatherFrom, LocalContainer& values) const;

    /**
     * @brief compute total # of elements the calling process receives in the scatter process (or sends in the gather process)
     *
     * (which might not equal the send size in each process)
     *
     * @return # of elements to receive
     */
    unsigned recv_size() const;
    /**
     * @brief return # of elements the calling process has to send in a scatter process (or receive in the gather process)
     *
     * equals the size of the map given in the constructor
     * @return # of elements to send
     */
    unsigned send_size() const; 
    /**
    * @brief The size of the collected vector = recv_size()
    *
    * may return 0
    * @return 
    */
    unsigned size() const;
    /**
    * @brief The internal mpi communicator used 
    *
    * used to assert that communicators of matrix and vector are the same
    * @return MPI Communicator
    */
    MPI_Comm communicator() const {return p_.communicator();}
};
