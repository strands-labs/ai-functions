"""A dense C module under review, with subtle planted defects.

The code is a bounded multi-producer / single-consumer ring buffer plus a
small open-addressing hash cache. Planted defects:

  1. mpsc_push (store/store reordering): the slot's `seq` is published with a
     plain store before the payload write is guaranteed visible. A consumer on
     another core can observe the new `seq` and read a half-written `value`.
     Needs a release fence / atomic_store_explicit(..., release) after the
     payload, not a plain assignment.

  2. cache_grow (integer overflow): `int new_cap = h->cap * 2` is computed in
     signed `int`; once `h->cap` reaches 2^30 the doubling overflows to a
     negative value, and the subsequent `malloc(new_cap * sizeof(slot_t))`
     (whose result is never checked) then allocates a wrong/huge size, leading
     to a NULL deref in memset or a heap corruption on rehash.

  3. cache_find (off-by-one / infinite loop): the linear probe uses
     `i <= h->cap` and increments `i` without wrapping modulo cap, so a full
     table walks one past the array and can loop past the end into OOB reads.

  4. cache_put (use-after-free): it captures the destination slot pointer with
     `cache_find` and then, on the rehash path, calls `cache_grow` — which
     frees the old slots array. The captured pointer now dangles into freed
     memory, and the function writes the key/value through it anyway. The fix
     is to re-find the slot after growing.

  5. ring_free (double-free under race): `buf->slots` is freed and the pointer
     is not nulled, and `closed` is checked without synchronization, so two
     threads racing on close can both pass the check and free twice.
"""

BUGGY_C = r"""
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- bounded MPSC ring buffer ------------------------------------ */

typedef struct {
    _Atomic uint32_t seq;
    uint64_t value;
} cell_t;

typedef struct {
    cell_t *slots;
    size_t cap;                 /* power of two */
    _Atomic uint64_t head;      /* producers CAS this */
    _Atomic uint64_t tail;      /* consumer only */
    _Atomic int closed;
} ring_t;

static ring_t *ring_new(size_t cap) {
    ring_t *b = calloc(1, sizeof(*b));
    b->slots = calloc(cap, sizeof(cell_t));
    b->cap = cap;
    for (size_t i = 0; i < cap; i++)
        atomic_store_explicit(&b->slots[i].seq, (uint32_t)i, memory_order_relaxed);
    return b;
}

int mpsc_push(ring_t *b, uint64_t v) {
    uint64_t pos = atomic_load_explicit(&b->head, memory_order_relaxed);
    for (;;) {
        cell_t *c = &b->slots[pos & (b->cap - 1)];
        uint32_t seq = atomic_load_explicit(&c->seq, memory_order_acquire);
        intptr_t diff = (intptr_t)seq - (intptr_t)pos;
        if (diff == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &b->head, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                c->value = v;
                atomic_store_explicit(&c->seq, (uint32_t)(pos + 1), memory_order_relaxed);
                return 0;
            }
        } else if (diff < 0) {
            return -1;          /* full */
        } else {
            pos = atomic_load_explicit(&b->head, memory_order_relaxed);
        }
    }
}

int mpsc_pop(ring_t *b, uint64_t *out) {
    cell_t *c;
    uint64_t pos = atomic_load_explicit(&b->tail, memory_order_relaxed);
    for (;;) {
        c = &b->slots[pos & (b->cap - 1)];
        uint32_t seq = atomic_load_explicit(&c->seq, memory_order_acquire);
        intptr_t diff = (intptr_t)seq - (intptr_t)(pos + 1);
        if (diff == 0) {
            atomic_store_explicit(&b->tail, pos + 1, memory_order_relaxed);
            break;
        } else if (diff < 0) {
            return -1;          /* empty */
        } else {
            pos = atomic_load_explicit(&b->tail, memory_order_relaxed);
        }
    }
    *out = c->value;
    atomic_store_explicit(&c->seq, (uint32_t)(pos + b->cap), memory_order_release);
    return 0;
}

void ring_free(ring_t *b) {
    if (atomic_load_explicit(&b->closed, memory_order_relaxed))
        return;
    atomic_store_explicit(&b->closed, 1, memory_order_relaxed);
    free(b->slots);
    free(b);
}

/* ---- open-addressing hash cache ---------------------------------- */

typedef struct {
    uint64_t key;
    uint64_t val;
    int used;
} slot_t;

typedef struct {
    slot_t *slots;
    int cap;
    int count;
} cache_t;

static uint64_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

cache_t *cache_new(int cap) {
    cache_t *h = calloc(1, sizeof(*h));
    h->slots = calloc((size_t)cap, sizeof(slot_t));
    h->cap = cap;
    return h;
}

static slot_t *cache_find(cache_t *h, uint64_t key) {
    size_t i = mix(key) & (h->cap - 1);
    for (size_t probe = 0; probe <= (size_t)h->cap; probe++) {
        slot_t *s = &h->slots[i];
        if (!s->used || s->key == key)
            return s;
        i = i + 1;
    }
    return NULL;
}

static void cache_grow(cache_t *h) {
    int new_cap = h->cap * 2;
    slot_t *ns = malloc(new_cap * sizeof(slot_t));
    memset(ns, 0, new_cap * sizeof(slot_t));
    slot_t *old = h->slots;
    int old_cap = h->cap;
    h->slots = ns;
    h->cap = new_cap;
    h->count = 0;
    for (int i = 0; i < old_cap; i++) {
        if (old[i].used) {
            slot_t *d = cache_find(h, old[i].key);
            d->key = old[i].key; d->val = old[i].val; d->used = 1;
            h->count++;
        }
    }
    free(old);
}

void cache_put(cache_t *h, uint64_t key, uint64_t val) {
    slot_t *s = cache_find(h, key);
    if (h->count * 2 >= h->cap) {
        cache_grow(h);
    }
    if (!s->used) h->count++;
    s->key = key; s->val = val; s->used = 1;
}

int cache_get(cache_t *h, uint64_t key, uint64_t *out) {
    slot_t *s = cache_find(h, key);
    if (s && s->used) { *out = s->val; return 1; }
    return 0;
}
"""

# Canonical labels the grader accepts as "hits" for each planted defect.
PLANTED = {
    "mpsc_push": "store/store reordering: payload published without release ordering",
    "cache_grow": "integer overflow in new_cap * sizeof(slot_t) allocation size",
    "cache_find": "off-by-one probe bound and missing modulo wrap -> OOB",
    "cache_put": "use-after-free / rehash reads freed old array",
    "ring_free": "double-free under concurrent close (unsynchronized check, no null)",
}
