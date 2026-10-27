/*
 * A stand-in for the handful of BNNS / vImage entry points that
 * tools/native16_diagnostics.py probes, written from the SDK header
 * documentation. It exists only so the ctypes plumbing (struct layout,
 * argument passing, alloc_memory callbacks, client-pointer detection) can
 * be exercised on a machine without Accelerate. It says nothing about what
 * the real Accelerate does -- that is what the device run is for.
 *
 * Behaviour chosen to cover both outcomes the probe distinguishes:
 *   - FC with BNNSFlagsUseClientPtr reads the caller's weights at apply
 *     time; without it, it copies them into memory from alloc_memory.
 *   - FC refuses BF16 *input* (so a "refused" path is exercised).
 *
 * Built by tests/test_native16_diagnostics.py with the system C compiler.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define F16 0x10010u
#define F32 0x10020u
#define BF16 0x18010u

typedef struct {
  uint32_t flags, layout;
  size_t size[8], stride[8];
  void *data;
  uint32_t data_type;
  void *table_data;
  uint32_t table_data_type;
  float data_scale, data_bias;
} D;
typedef struct {
  uint32_t function; float alpha, beta; int32_t is, io, ish;
  const void *a, *b, *c;
} Act;
typedef struct { D i_desc, w_desc, o_desc, bias; Act activation; } FCP;
typedef int (*AllocF)(void **, size_t, size_t);
typedef void (*FreeF)(void *);
typedef struct { uint32_t flags; size_t n_threads; AllocF al; FreeF fr; } FP;
typedef struct { void *data; unsigned long h, w; size_t rb; } VB;

static size_t esize(uint32_t t) { return t == F32 ? 4 : 2; }

static float h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
  uint32_t u;
  if (e == 0) {
    if (m == 0) u = s;
    else { e = 113; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023;
           u = s | (e << 23) | (m << 13); }
  } else if (e == 31) u = s | 0x7f800000u | (m << 13);
  else u = s | ((e + 112) << 23) | (m << 13);
  float f; memcpy(&f, &u, 4); return f;
}
static uint16_t f2h(float f) {   /* round to nearest even, normal range */
  uint32_t u; memcpy(&u, &f, 4);
  uint32_t s = (u >> 16) & 0x8000; int e = (int)((u >> 23) & 255) - 112;
  uint32_t m = u & 0x7fffff;
  if (e <= 0) return (uint16_t)s;
  if (e >= 31) return (uint16_t)(s | 0x7c00);
  uint32_t r = (uint32_t)(e << 10) | (m >> 13), rem = m & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
  return (uint16_t)(s | r);
}
static float get(const void *p, uint32_t t, size_t i) {
  if (t == F32) return ((const float *)p)[i];
  uint16_t h = ((const uint16_t *)p)[i];
  if (t == F16) return h2f(h);
  uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static void put(void *p, uint32_t t, size_t i, float v) {
  if (t == F32) { ((float *)p)[i] = v; return; }
  if (t == F16) { ((uint16_t *)p)[i] = f2h(v); return; }
  uint32_t u; memcpy(&u, &v, 4);
  u += 0x7fff + ((u >> 16) & 1);
  ((uint16_t *)p)[i] = (uint16_t)(u >> 16);
}
static int okt(uint32_t t) { return t == F32 || t == F16 || t == BF16; }

size_t BNNSNDArrayGetDataSize(const D *d) {
  size_t n = 1, r = d->layout >> 16;
  for (size_t k = 0; k < r; k++) n *= d->size[k];
  return n * esize(d->data_type);
}

int BNNSCopy(D *dst, const D *src, const FP *p) {
  (void)p;
  if (!okt(dst->data_type) || !okt(src->data_type)) return -1;
  for (size_t i = 0; i < src->size[0]; i++)
    put(dst->data, dst->data_type, i, get(src->data, src->data_type, i));
  return 0;
}

long vImageConvert_Planar16FtoPlanarF(const VB *s, const VB *d, uint32_t f) {
  (void)f;
  for (size_t i = 0; i < s->w; i++)
    ((float *)d->data)[i] = h2f(((const uint16_t *)s->data)[i]);
  return 0;
}

typedef struct {
  size_t n_in, n_out; uint32_t wt, it, ot; void *w; int owned; FreeF fr;
} Filter;

void *BNNSFilterCreateLayerFullyConnected(const FCP *l, const FP *p) {
  if (!okt(l->w_desc.data_type) || !okt(l->o_desc.data_type)) return NULL;
  if (l->i_desc.data_type != F32 && l->i_desc.data_type != F16) return NULL;
  if (l->w_desc.layout != 0x20000u || l->w_desc.size[0] != l->i_desc.size[0]
      || l->w_desc.size[1] != l->o_desc.size[0]) return NULL;
  Filter *f = NULL;
  if (p && p->al) { if (p->al((void **)&f, 16, sizeof *f)) return NULL; }
  else f = malloc(sizeof *f);
  f->n_in = l->i_desc.size[0]; f->n_out = l->o_desc.size[0];
  f->wt = l->w_desc.data_type; f->it = l->i_desc.data_type;
  f->ot = l->o_desc.data_type; f->fr = p ? p->fr : NULL;
  size_t nb = f->n_in * f->n_out * esize(f->wt);
  if (p && (p->flags & 1)) { f->w = l->w_desc.data; f->owned = 0; }
  else {
    if (p && p->al) p->al(&f->w, 16, nb); else f->w = malloc(nb);
    memcpy(f->w, l->w_desc.data, nb); f->owned = 1;
  }
  return f;
}

int BNNSFilterApply(void *fv, const void *in, void *out) {
  Filter *f = fv;
  for (size_t o = 0; o < f->n_out; o++) {
    float acc = 0;
    for (size_t i = 0; i < f->n_in; i++)
      acc += get(f->w, f->wt, o * f->n_in + i) * get(in, f->it, i);
    put(out, f->ot, o, acc);
  }
  return 0;
}

void BNNSFilterDestroy(void *fv) {
  Filter *f = fv;
  if (!f) return;
  FreeF fr = f->fr ? f->fr : free;
  if (f->owned) fr(f->w);
  fr(f);
}

static int mm_ok(const D *a, const D *b, const D *c) {
  return okt(a->data_type) && okt(b->data_type) && okt(c->data_type)
         && a->layout == 0x28001u && b->layout == 0x28001u
         && c->layout == 0x28001u && a->size[1] == b->size[0]
         && c->size[0] == a->size[0] && c->size[1] == b->size[1];
}

long BNNSMatMulWorkspaceSize(bool ta, bool tb, float alpha, const D *a,
                             const D *b, const D *c, const FP *p) {
  (void)alpha; (void)p;
  if (ta || tb || !mm_ok(a, b, c)) return -1;
  return 64;
}

int BNNSMatMul(bool ta, bool tb, float alpha, const D *a, const D *b,
               const D *c, void *ws, const FP *p) {
  (void)ws; (void)p;
  if (ta || tb || !mm_ok(a, b, c)) return -1;
  /* 2DFirstMajor: (i, j) at j*stride[0] + i*stride[1] */
  for (size_t i = 0; i < c->size[0]; i++)
    for (size_t j = 0; j < c->size[1]; j++) {
      float acc = 0;
      for (size_t k = 0; k < a->size[1]; k++)
        acc += get(a->data, a->data_type, k * a->stride[0] + i * a->stride[1])
             * get(b->data, b->data_type, j * b->stride[0] + k * b->stride[1]);
      put(c->data, c->data_type, j * c->stride[0] + i * c->stride[1],
          alpha * acc);
    }
  return 0;
}
