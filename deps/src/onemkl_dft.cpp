#include "onemkl_dft.h"
#include "sycl.hpp"  // internal struct definitions

#include <oneapi/mkl/dft.hpp>
#include <vector>
#include <complex>
#include <new>
#include <exception>
#include <cstring>

using namespace oneapi::mkl::dft;

struct onemklDftDescriptor_st {
    precision prec;
    domain dom;
    void *ptr; // pointer to concrete descriptor<prec, dom>
};

static inline precision to_prec(onemklDftPrecision p) {
    return (p == ONEMKL_DFT_PRECISION_DOUBLE) ? precision::DOUBLE : precision::SINGLE;
}

static inline domain to_dom(onemklDftDomain d) {
    return (d == ONEMKL_DFT_DOMAIN_COMPLEX) ? domain::COMPLEX : domain::REAL;
}

// Helper to allocate descriptor depending on precision/domain
static int allocate_descriptor(onemklDftDescriptor_t *out, precision p, domain d, const std::vector<int64_t> &lengths) {
    try {
        auto *desc = new onemklDftDescriptor_st();
        desc->prec = p;
        desc->dom = d;
        if (p == precision::SINGLE && d == domain::REAL) {
            desc->ptr = new descriptor<precision::SINGLE, domain::REAL>(lengths);
        } else if (p == precision::SINGLE && d == domain::COMPLEX) {
            desc->ptr = new descriptor<precision::SINGLE, domain::COMPLEX>(lengths);
        } else if (p == precision::DOUBLE && d == domain::REAL) {
            desc->ptr = new descriptor<precision::DOUBLE, domain::REAL>(lengths);
        } else { // DOUBLE COMPLEX
            desc->ptr = new descriptor<precision::DOUBLE, domain::COMPLEX>(lengths);
        }
        *out = desc;
        return 0;
    } catch (...) {
        return -1;
    }
}

int onemklDftCreate1D(onemklDftDescriptor_t *desc,
                      onemklDftPrecision precision,
                      onemklDftDomain domain,
                      int64_t length) {
    std::vector<int64_t> dims{length};
    return allocate_descriptor(desc, to_prec(precision), to_dom(domain), dims);
}

int onemklDftCreateND(onemklDftDescriptor_t *desc,
                      onemklDftPrecision precision,
                      onemklDftDomain domain,
                      int64_t dim,
                      const int64_t *lengths) {
    if (dim <= 0 || lengths == nullptr) return -2;
    std::vector<int64_t> dims(lengths, lengths + dim);
    return allocate_descriptor(desc, to_prec(precision), to_dom(domain), dims);
}

int onemklDftDestroy(onemklDftDescriptor_t desc) {
    if (!desc) return 0;
    try {
        if (desc->prec == precision::SINGLE && desc->dom == domain::REAL) {
            delete static_cast< descriptor<precision::SINGLE, domain::REAL>* >(desc->ptr);
        } else if (desc->prec == precision::SINGLE && desc->dom == domain::COMPLEX) {
            delete static_cast< descriptor<precision::SINGLE, domain::COMPLEX>* >(desc->ptr);
        } else if (desc->prec == precision::DOUBLE && desc->dom == domain::REAL) {
            delete static_cast< descriptor<precision::DOUBLE, domain::REAL>* >(desc->ptr);
        } else {
            delete static_cast< descriptor<precision::DOUBLE, domain::COMPLEX>* >(desc->ptr);
        }
        delete desc;
        return 0;
    } catch (...) {
        return -1;
    }
}

int onemklDftCommit(onemklDftDescriptor_t desc, syclQueue_t queue) {
    if (!desc || !queue) return -2;
    try {
        if (desc->prec == precision::SINGLE && desc->dom == domain::REAL) {
            static_cast< descriptor<precision::SINGLE, domain::REAL>* >(desc->ptr)->commit(queue->val);
        } else if (desc->prec == precision::SINGLE && desc->dom == domain::COMPLEX) {
            static_cast< descriptor<precision::SINGLE, domain::COMPLEX>* >(desc->ptr)->commit(queue->val);
        } else if (desc->prec == precision::DOUBLE && desc->dom == domain::REAL) {
            static_cast< descriptor<precision::DOUBLE, domain::REAL>* >(desc->ptr)->commit(queue->val);
        } else {
            static_cast< descriptor<precision::DOUBLE, domain::COMPLEX>* >(desc->ptr)->commit(queue->val);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

// Internal mapping helpers for config params/values; rely on enum ordering matching header.
static inline config_param to_param(onemklDftConfigParam p) { return static_cast<config_param>(p); }
static inline config_value to_cvalue(onemklDftConfigValue v) { return static_cast<config_value>(v); }

// Dispatch macro re-used for configuration
#define ONEMKL_DFT_DISPATCH_CFG(desc_expr, CALL) \
    do { \
        if (desc->prec == precision::SINGLE && desc->dom == domain::REAL) { \
            auto *d = static_cast< descriptor<precision::SINGLE, domain::REAL>* >(desc_expr); \
            CALL; \
        } else if (desc->prec == precision::SINGLE && desc->dom == domain::COMPLEX) { \
            auto *d = static_cast< descriptor<precision::SINGLE, domain::COMPLEX>* >(desc_expr); \
            CALL; \
        } else if (desc->prec == precision::DOUBLE && desc->dom == domain::REAL) { \
            auto *d = static_cast< descriptor<precision::DOUBLE, domain::REAL>* >(desc_expr); \
            CALL; \
        } else { \
            auto *d = static_cast< descriptor<precision::DOUBLE, domain::COMPLEX>* >(desc_expr); \
            CALL; \
        } \
    } while (0)

int onemklDftSetValueInt64(onemklDftDescriptor_t desc, onemklDftConfigParam param, int64_t value) {
    if (!desc) return -2;
    try { ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->set_value(to_param(param), value)); return 0; } catch (...) { return -1; }
}

int onemklDftSetValueDouble(onemklDftDescriptor_t desc, onemklDftConfigParam param, double value) {
    if (!desc) return -2;
    try { ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->set_value(to_param(param), value)); return 0; } catch (...) { return -1; }
}

int onemklDftSetValueInt64Array(onemklDftDescriptor_t desc, onemklDftConfigParam param, const int64_t *values, int64_t n) {
    if (!desc || !values || n < 0) return -2;
    try { std::vector<int64_t> v(values, values + n); ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->set_value(to_param(param), v)); return 0; } catch (...) { return -1; }
}

int onemklDftSetValueConfigValue(onemklDftDescriptor_t desc, onemklDftConfigParam param, onemklDftConfigValue value) {
    if (!desc) return -2;
    try { ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->set_value(to_param(param), to_cvalue(value))); return 0; } catch (...) { return -1; }
}

int onemklDftGetValueInt64(onemklDftDescriptor_t desc, onemklDftConfigParam param, int64_t *value) {
    if (!desc || !value) return -2;
    try { ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->get_value(to_param(param), value)); return 0; } catch (...) { return -1; }
}

int onemklDftGetValueDouble(onemklDftDescriptor_t desc, onemklDftConfigParam param, double *value) {
    if (!desc || !value) return -2;
    try { ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->get_value(to_param(param), value)); return 0; } catch (...) { return -1; }
}

int onemklDftGetValueInt64Array(onemklDftDescriptor_t desc, onemklDftConfigParam param, int64_t *values, int64_t *n) {
    if (!desc || !values || !n || *n <= 0) return -2;
    try {
        std::vector<int64_t> v; ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->get_value(to_param(param), &v));
        int64_t to_copy = (*n < (int64_t)v.size()) ? *n : (int64_t)v.size();
        std::memcpy(values, v.data(), sizeof(int64_t)*to_copy);
        *n = to_copy; return 0;
    } catch (...) { return -1; }
}

int onemklDftGetValueConfigValue(onemklDftDescriptor_t desc, onemklDftConfigParam param, onemklDftConfigValue *value) {
    if (!desc || !value) return -2;
    try { config_value cv; ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->get_value(to_param(param), &cv)); *value = static_cast<onemklDftConfigValue>(cv); return 0; } catch (...) { return -1; }
}

// Helper macro to dispatch compute operations
#define ONEMKL_DFT_DISPATCH(desc_expr, CALL) \
    do { \
        if (desc->prec == precision::SINGLE && desc->dom == domain::REAL) { \
            auto *d = static_cast< descriptor<precision::SINGLE, domain::REAL>* >(desc_expr); \
            CALL; \
        } else if (desc->prec == precision::SINGLE && desc->dom == domain::COMPLEX) { \
            auto *d = static_cast< descriptor<precision::SINGLE, domain::COMPLEX>* >(desc_expr); \
            CALL; \
        } else if (desc->prec == precision::DOUBLE && desc->dom == domain::REAL) { \
            auto *d = static_cast< descriptor<precision::DOUBLE, domain::REAL>* >(desc_expr); \
            CALL; \
        } else { \
            auto *d = static_cast< descriptor<precision::DOUBLE, domain::COMPLEX>* >(desc_expr); \
            CALL; \
        } \
    } while (0)

int onemklDftComputeForward(onemklDftDescriptor_t desc, void *inout) {
    if (!desc || !inout) return -2;
    try {
        ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, inout));
        return 0;
    } catch (...) {
        return -1;
    }
}

int onemklDftComputeForwardOutOfPlace(onemklDftDescriptor_t desc, void *in, void *out) {
    if (!desc || !in || !out) return -2;
    try {
        ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, in, out));
        return 0;
    } catch (...) {
        return -1;
    }
}

int onemklDftComputeBackward(onemklDftDescriptor_t desc, void *inout) {
    if (!desc || !inout) return -2;
    try {
        ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, inout));
        return 0;
    } catch (...) {
        return -1;
    }
}

int onemklDftComputeBackwardOutOfPlace(onemklDftDescriptor_t desc, void *in, void *out) {
    if (!desc || !in || !out) return -2;
    try {
        ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, in, out));
        return 0;
    } catch (...) {
        return -1;
    }
}

// Keep dispatch macros defined for buffer variants below; undef at end of file.

// Buffer API helpers: create temporary buffers referencing host memory.
// NOTE: This assumes the memory is accessible and sized appropriately.
template <typename T>
static inline sycl::buffer<T,1> make_buffer(T *ptr, int64_t n) {
    return sycl::buffer<T,1>(ptr, sycl::range<1>(static_cast<size_t>(n)));
}

// Query total element count from LENGTHS config (product of lengths).
static int64_t get_element_count(onemklDftDescriptor_t desc) {
    int64_t n = 0; int64_t dims = 0; if (onemklDftGetValueInt64(desc, ONEMKL_DFT_PARAM_DIMENSION, &dims) != 0) return -1; if (dims <= 0 || dims > 8) return -1; int64_t lens[16]; int64_t want = dims; if (onemklDftGetValueInt64Array(desc, ONEMKL_DFT_PARAM_LENGTHS, lens, &want) != 0) return -1; if (want != dims) return -1; int64_t total = 1; for (int i=0;i<dims;i++){ if (lens[i]<=0) return -1; total *= lens[i]; } return total; }

// Select real/complex element size variant for pointers.
int onemklDftComputeForwardBuffer(onemklDftDescriptor_t desc, void *inout) {
    if (!desc || !inout) return -2; int64_t n = get_element_count(desc); if (n <= 0) return -3; try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) { auto buf = make_buffer((float*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, buf)); }
            else { auto buf = make_buffer((double*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, buf)); }
        } else { // COMPLEX
            if (desc->prec == precision::SINGLE) { auto buf = make_buffer((std::complex<float>*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, buf)); }
            else { auto buf = make_buffer((std::complex<double>*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, buf)); }
        }
        return 0; } catch (...) { return -1; }
}

int onemklDftComputeForwardOutOfPlaceBuffer(onemklDftDescriptor_t desc, void *in, void *out) {
    if (!desc || !in || !out) return -2; int64_t n = get_element_count(desc); if (n <= 0) return -3; try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) { auto bufi = make_buffer((float*)in, n); auto bufo = make_buffer((float*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, bufi, bufo)); }
            else { auto bufi = make_buffer((double*)in, n); auto bufo = make_buffer((double*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, bufi, bufo)); }
        } else {
            if (desc->prec == precision::SINGLE) { auto bufi = make_buffer((std::complex<float>*)in, n); auto bufo = make_buffer((std::complex<float>*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, bufi, bufo)); }
            else { auto bufi = make_buffer((std::complex<double>*)in, n); auto bufo = make_buffer((std::complex<double>*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, bufi, bufo)); }
        }
        return 0; } catch (...) { return -1; }
}

int onemklDftComputeBackwardBuffer(onemklDftDescriptor_t desc, void *inout) {
    if (!desc || !inout) return -2; int64_t n = get_element_count(desc); if (n <= 0) return -3; try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) { auto buf = make_buffer((float*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, buf)); }
            else { auto buf = make_buffer((double*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, buf)); }
        } else {
            if (desc->prec == precision::SINGLE) { auto buf = make_buffer((std::complex<float>*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, buf)); }
            else { auto buf = make_buffer((std::complex<double>*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, buf)); }
        }
        return 0; } catch (...) { return -1; }
}

int onemklDftComputeBackwardOutOfPlaceBuffer(onemklDftDescriptor_t desc, void *in, void *out) {
    if (!desc || !in || !out) return -2; int64_t n = get_element_count(desc); if (n <= 0) return -3; try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) { auto bufi = make_buffer((float*)in, n); auto bufo = make_buffer((float*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, bufi, bufo)); }
            else { auto bufi = make_buffer((double*)in, n); auto bufo = make_buffer((double*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, bufi, bufo)); }
        } else {
            if (desc->prec == precision::SINGLE) { auto bufi = make_buffer((std::complex<float>*)in, n); auto bufo = make_buffer((std::complex<float>*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, bufi, bufo)); }
            else { auto bufi = make_buffer((std::complex<double>*)in, n); auto bufo = make_buffer((std::complex<double>*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, bufi, bufo)); }
        }
        return 0; } catch (...) { return -1; }
}

#undef ONEMKL_DFT_DISPATCH
#undef ONEMKL_DFT_DISPATCH_CFG
