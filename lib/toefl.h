#error Documentation only
/*! @namespace toefl
 * @brief This is the namespace for all functions and 
 * classes defined and used by the toefl solvers.
 *
 *
 * @mainpage
 * This is the toefl library.
 * @par A word on indexing
 *
 * Assuming we have a physical box, that spans the coordinate range (0,0) to (lx, ly) 
 * we make the following correspondance between matrix elements and grid points:
 *
 * The first element of our matrices corresponds to the lower left corner of the box (i.e. the origin)
 * Subsequent elements progress left to right through the remaining grid points in the lowest row of the grid
 * and then in successively higher rows of the grid. The final element corresponds to the upper right corner 
 * of the grid. 
 * Note that in most image container formats (e.g. PGM) and in windowing systems (e.g. GLFW) 
 * the first element of a vector (double* field; field[0]) corresponds to the upper left corner of an image. 
 * Subsequent elements progress left-to-right in the highest row and then in successively lower rows of the image. 
 * The final element corresponds to the lower right corner of the image. (cf. http://de.wikipedia.org/wiki/Portable_Graymap)
 * This is the way you read text. 
 * In C-Convention the 2nd index of a 2-dimensional field is the continuous one.
 * That means that the X-INDEX IS THE SECOND ONE.
 * For cases that defer from this we use the suffix _T (for transposed) in variable names, 
 * i.e. the vector double* field_T would store the y-direction continuously and the x-index would be the first one.
 *
 * @par A word on CUDA coding convention
 *
 * When programming on the graphics card it is important to note that 
 * the memory spaces of CPU and GPU are seperate and no one can (directly) access elements
 * that live in the other memory space. 
 * We introduce the convention that a variable with the prefix d_ lives in 
 * device memory (i.e. the GPU). Furthermore a pointer to a variable that lives
 * in device memory is indicated with the suffix _d.
 *
 * double* d_ptr_d; is thus a pointer in device memory pointing to a double in device memory 
 * double* ptr_d; is a pointer in cpu memory that points to a double in device memory.
 *
 * 
 *
 */
