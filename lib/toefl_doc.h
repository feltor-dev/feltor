#error Documentation only
/*! @namespace toefl
 * @brief This is the namespace for all functions and 
 * classes defined and used by the toefl solvers.
 *
 *
 * @mainpage
 * Welcome to the TOEFL library and the INNTO code.
 *
 * @par Structure of the library
 *  The library provides a matrix class designed to cope well with inplace fourier
 *  transformations as well as finite difference schemes like the arakawa scheme.
 *  This data structure is the basis for three kind of solver classes. The Fourier
 *  transform classes, the Arakawa class and the Karniadakis class which takes
 *  care of timestepping issues. The library further provides a read function
 *  for ASCII parameter input and a texture interface for generating
 *  textures to be used by OpenGL visualisation. 
 *
 * @note Every .h file comes along with a _t.cpp and a _b.cpp file. These 
 * are for testing and benchmarking purposes. Type make *_t/b to compile
 * and run a testing/benchmark program. (e.g. make arakawa_t)
 *
 * @par Error Checking
 *  For performance reasons errors (like out of bound indexing) are only 
 *  checked when the Macro TL_DEBUG is defined. (Use e.g. the gcc compiler option
 *  -DTL_DEBUG to define it). This might decreases performance by a factor of 10, 
 *  but it's essential when you build a new solver. When it is defined, instances of 
 *  the Message class will be thrown at you and if you don't catch them will 
 *  terminate your program. So you probably want to catch them and see what 
 *  the message is after all ;-)
 *
 * @par A word on indexing
 * Assuming we have a physical box, that spans the coordinate range (0,0) to (lx, ly) 
 * we make the following correspondance between matrix elements and grid points:
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
 * @par Compile and link process
 *  The TOEFL library is header only i.e. you don't have to precompile anything. 
 *  The TOEFL library makes use of many features of the new C++11 standard so 
 *  be sure your Compiler supports that. 
 *  On the other hand you will need to add -lfftw and -lm to your compile command
 *  in order to link the fftw and math library. Some headers like
 *  texture.h will also require you to have OpenGL installed.
 *
 * @par Memory leaks
 *  Most of the headers have been explicitely tested for memory leaks with valgrind 
 *  and it is fair to assume that this library doesn't generate any. 
 *  However the fftw plans keep some persistent memory even after destroying
 *  them and fftw_cleanup() should be called before exit of a program to 
 *  also release this memory. 
 *  Also in connection with direct visualisation using OpenGL (small) memory leaks
 *  may occur because some older nvidia drivers generate memory leaks.
 *
 * @par The INNTO code
 *  INNTO stands for Innsbruck TOEFL. The innto code is included 
 *  in and build on top of the TOEFL library. 
 *  Its documentation comes in a seperate 
 *  tex-file in the main directory which can be compiled using make.
 *
 *
 * Have fun using the TOEFL library.
 * 
 *
 */
/*
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
 */
