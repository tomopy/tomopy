// Platform Compatability Layer (PAL)
//
// Uses the preprocessor to swap implementations of key functions for different
// platforms without complicating the implementaiton of gridrec

#pragma once

// Complex types and functions ----------------------------------------------//
#ifdef __cplusplus
// For CXX use the standard library complex types
#    include <complex>
typedef std::complex<float> PAL_COMPLEX;
#    define conjf std::conj<float>
#    define crealf std::real<float>
#    define cimagf std::imag<float>
using namespace std::literals::complex_literals;
#    define I 1if
#else
// For C use the complex header which is only present with some compilers
#    include <complex.h>
typedef float _Complex PAL_COMPLEX;
#    include <stdlib.h>
#endif

// Memory allocation and alignment ------------------------------------------//

// Use X/Open-7, where posix_memalign is introduced
#define _XOPEN_SOURCE 700

#define __LIKELY(x) __builtin_expect(!!(x), 1)
#if defined(_MSC_VER)
#    if defined(__LIKELY)
#        undef __LIKELY
#    endif
#    define __LIKELY(EXPR) EXPR
#endif

#ifdef __INTEL_COMPILER
#    define __PRAGMA_SIMD _Pragma("simd assert")
#    define __PRAGMA_SIMD_VECREMAINDER _Pragma("simd assert, vecremainder")
#    define __PRAGMA_SIMD_VECREMAINDER_VECLEN8                                           \
        _Pragma("simd assert, vecremainder, vectorlength(8)")
#    define __PRAGMA_OMP_SIMD_COLLAPSE _Pragma("omp simd collapse(2)")
#    define __PRAGMA_IVDEP _Pragma("ivdep")
#    define __ASSSUME_64BYTES_ALIGNED(x) __assume_aligned((x), 64)
#else
#    define __PRAGMA_SIMD
#    define __PRAGMA_SIMD_VECREMAINDER
#    define __PRAGMA_SIMD_VECREMAINDER_VECLEN8
#    define __PRAGMA_OMP_SIMD_COLLAPSE
#    define __PRAGMA_IVDEP
#    define __ASSSUME_64BYTES_ALIGNED(x)
#endif

//===========================================================================//

inline float*
malloc_vector_f(size_t n)
{
#ifdef __cplusplus
    return new float[n];
#else
    return (float*) malloc(n * sizeof(float));
#endif
}

inline void
free_vector_f(float* v)
{
#ifdef __cplusplus
    delete[] v;
    v = nullptr;
#else
    free(v);
#endif
}

//===========================================================================//

inline PAL_COMPLEX*
malloc_vector_c(size_t n)
{
#ifdef __cplusplus
    return new PAL_COMPLEX[n];
#else
    return (PAL_COMPLEX*) malloc(n * sizeof(PAL_COMPLEX));
#endif
}

#ifdef __cplusplus
inline void
free_vector_c(PAL_COMPLEX*& v)
{
    delete[] v;
    v = nullptr;
}
#else
inline void
free_vector_c(PAL_COMPLEX* v)
{
    free(v);
}
#endif

//===========================================================================//

static inline void*
malloc_64bytes_aligned(size_t sz)
{
#if defined(__MINGW32__)
    return __mingw_aligned_malloc(sz, 64);
#elif defined(_MSC_VER)
    void* r = _aligned_malloc(sz, 64);
    return r;
#else
    void* r   = NULL;
    int   err = posix_memalign(&r, 64, sz);
    return (err) ? NULL : r;
#endif
}

//===========================================================================//

PAL_COMPLEX**
malloc_matrix_c(size_t nr, size_t nc)
{
#ifdef __cplusplus
    PAL_COMPLEX** m = nullptr;
#else
    PAL_COMPLEX** m = NULL;
#endif
    size_t i;

    // Allocate pointers to rows,
    m = (PAL_COMPLEX**) malloc_64bytes_aligned(nr * sizeof(PAL_COMPLEX*));

    /* Allocate rows and set the pointers to them */
    m[0] = malloc_vector_c(nr * nc);

    for(i = 1; i < nr; i++)
    {
        m[i] = m[i - 1] + nc;
    }
    return m;
}

//===========================================================================//

#ifdef __cplusplus
void
free_matrix_c(PAL_COMPLEX**& m)
{
    free_vector_c(m[0]);
#    if defined(__MINGW32__)
    __mingw_aligned_free(m);
#    elif defined(_MSC_VER)
    _aligned_free(m);
#    else
    free(m);
#    endif
    m = nullptr;
}
#else
inline void
free_matrix_c(PAL_COMPLEX** m)
{
    free_vector_c(m[0]);
#    ifdef __MINGW32__
    __mingw_aligned_free(m);
#    else
    free(m);
#    endif
}
#endif
