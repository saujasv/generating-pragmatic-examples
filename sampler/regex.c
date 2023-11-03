// after https://swtch.com/~rsc/regexp/regexp1.html
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STATE_MATCH 0 // matching state
#define STATE_SPLIT 1 // go down both branches
struct nfa {
  uint32_t n; // number of nodes
  uint32_t size; // array size allocated

  // node properties
  // each node has a recognized char and up to two successors
  uint8_t *c; // should be a-zA-Z0-9
  uint32_t *out1; // successor indices
  uint32_t *out2;

  uint32_t start;

  struct nfa *next;
};

struct nfa *nfa_freelist = NULL;

struct nfa *nfa() {
  struct nfa *nfa;
  if (nfa_freelist == NULL) {
    nfa = malloc(sizeof(struct nfa));
    nfa->size = 0;
    nfa->c = NULL;
    nfa->out1 = NULL;
    nfa->out2 = NULL;
  } else {
    nfa = nfa_freelist;
    nfa_freelist = nfa_freelist->next;
  }

  nfa->n = 0;
  nfa->start = UINT32_MAX;

  return nfa;
}

void nfa_free(struct nfa *nfa) {
  nfa->next = nfa_freelist;
  nfa_freelist = nfa;
}

void nfa_cleanup() {
  while (nfa_freelist != NULL) {
    struct nfa *nfa = nfa_freelist;
    nfa_freelist = nfa_freelist->next;
    free(nfa->c);
    free(nfa->out1);
    free(nfa->out2);
    free(nfa);
  }
}

// add a state to an NFA
uint32_t state(struct nfa *nfa, uint8_t c, uint32_t out1, uint32_t out2) {
  // reallocate node arrays if necessary
  uint32_t n = nfa->n;
  assert(n < UINT32_MAX);
  if (n >= nfa->size) {
    nfa->size = nfa->size ? nfa->size * 2 : 16;
    nfa->c = realloc(nfa->c, nfa->size * sizeof(uint8_t));
    nfa->out1 = realloc(nfa->out1, nfa->size * sizeof(uint32_t));
    nfa->out2 = realloc(nfa->out2, nfa->size * sizeof(uint32_t));
  }

  nfa->c[n] = c;
  nfa->out1[n] = out1;
  nfa->out2[n] = out2;
  nfa->n++;

  return n;
}

// vectors of states
struct vec {
  uint32_t n;
  uint32_t size; // vec capacity is (1 << size)
  struct vec *next;
  uint32_t s[];
};

static struct vec *vec_freelist[32] = {NULL};

struct vec *vec_get(uint32_t size) {
  //printf("vec_get %u\n", size);
  struct vec *v;
  assert(size < 32);
  if (vec_freelist[size] != NULL) {
    v = vec_freelist[size];
    vec_freelist[size] = vec_freelist[size]->next;
    //printf("vec_get found %u\n", size);
  } else {
    v = malloc(sizeof(struct vec) + (1 << size) * sizeof(uint32_t));
    v->size = size;
    //printf("vec_get alloc %u\n", size);
  }
  //for (int i = 0; i < (1 << size); i++) {
  //  v->s[i] = -1;
  //}
  v->next = NULL;
  return v;
}

void vec_put(struct vec *v) {
  //printf("vec_put %u\n", v->size);
  v->next = vec_freelist[v->size];
  vec_freelist[v->size] = v;
}

void vec_cleanup() {
  for (int i = 0; i < 32; i++) {
    while (vec_freelist[i] != NULL) {
      struct vec *v = vec_freelist[i];
      vec_freelist[i] = vec_freelist[i]->next;
      free(v);
    }
  }
}

// create a vec containing 1 element
struct vec *vec(uint32_t s) {
  struct vec *v = vec_get(0);
  v->n = 1;
  v->s[0] = s;
  return v;
}

struct vec *add(struct vec *v, uint32_t s) {
  if (v->n >= (1 << v->size)) {
    struct vec *new = vec_get(v->size + 1);
    new->n = v->n;
    memcpy(new->s, v->s, sizeof(uint32_t) * v->n);
    vec_put(v);
    v = new;
  }
  v->s[v->n++] = s;
  return v;
}

// append v2 to v1 and free v2
struct vec *append(struct vec *v1, struct vec *v2) {
  uint32_t tot = v1->n + v2->n;
  if (tot > (1 << v1->size)) {
    uint32_t newsize = (v1->size > v2->size ? v1->size : v2->size) + 1;
    struct vec *new = vec_get(newsize);
    new->n = v1->n;
    memcpy(new->s, v1->s, v1->n * sizeof(uint32_t));
    vec_put(v1);
    v1 = new;
  }
  memcpy(&(v1->s[v1->n]), v2->s, v2->n * sizeof(uint32_t));
  v1->n = tot;
  vec_put(v2);
  return v1;
}

struct vec *copy(struct vec *v) {
  struct vec *new = vec_get(v->size);
  new->n = v->n;
  memcpy(new->s, v->s, v->n * sizeof(uint32_t));
  return new;
}

// attach all dangling arrows in vec v to state s
void patch(struct nfa *nfa, struct vec *v, uint32_t s) {
  for (int i = 0; i < v->n; i++) {
    uint32_t t = v->s[i];
    assert(nfa->out1[t] == UINT32_MAX || nfa->out2[t] == UINT32_MAX);
    if (nfa->out1[t] == UINT32_MAX) {
      nfa->out1[t] = s;
    }
    if (nfa->out2[t] == UINT32_MAX) {
      nfa->out2[t] = s;
    }
  }
  vec_put(v);
}

// NFA fragments
// start: index of start state
// min, max: each fragment encompasses block of states [min, max)
// invariant: states in a frag only point to states in the same frag
struct frag {
  uint32_t start, min, max;
  struct vec *out; // list of states with dangling arrows
  struct frag *next;
};

static struct frag *frag_freelist = NULL;

struct frag *frag(uint32_t start, uint32_t min, uint32_t max, struct vec *out) {
  struct frag *f;
  if (frag_freelist == NULL) {
    f = malloc(sizeof(struct frag));
  } else {
    f = frag_freelist;
    frag_freelist = frag_freelist->next;
  }

  f->start = start;
  f->min = min;
  f->max = max;
  f->out = out;
  f->next = NULL;
  return f;
}

void frag_free(struct frag *f) {
  f->next = frag_freelist;
  frag_freelist = f;
}

void frag_cleanup() {
  while (frag_freelist != NULL) {
    struct frag *f = frag_freelist;
    frag_freelist = frag_freelist->next;
    free(f);
  }
}

// copy an NFA fragment, using all new states
struct frag *dup(struct nfa *nfa, struct frag *f) {
  uint32_t base = nfa->n;
  uint32_t shift = base - f->min;

  // add copied states
  for (int i = f->min; i < f->max; i++) {
    uint32_t new1, new2;
    if (nfa->out1[i] != UINT32_MAX) {
      assert(f->min <= nfa->out1[i] && nfa->out1[i] < f->max);
      new1 = nfa->out1[i] + shift;
    } else {
      new1 = UINT32_MAX;
    }
    if (nfa->out2[i] != UINT32_MAX) {
      assert(f->min <= nfa->out2[i] && nfa->out2[i] < f->max);
      new2 = nfa->out2[i] + shift;
    } else {
      new2 = UINT32_MAX;
    }
    state(nfa, nfa->c[i], new1, new2);
  }

  // copy and shift dangling pointer list
  struct vec *new_out = copy(f->out);
  for (int i = 0; i < new_out->n; i++) {
    new_out->s[i] += shift;
  }

  return frag(f->start + shift, f->min + shift, f->max + shift, new_out);
}

// maintains a stack of NFA fragments
// use build_* methods to operate on stack
// invariant: frags on stack cover sequential, disjoint, adjacent blocks
#define BUILDER_MAX 1024
struct builder {
  struct nfa *nfa;
  struct frag *stack[BUILDER_MAX];
  uint16_t stack_i;
};

struct builder *builder() {
  struct builder *b = malloc(sizeof(struct builder));
  b->nfa = nfa();
  b->stack_i = 0;
  return b;
}

void push(struct builder *b, struct frag *f) {
  assert(b->stack_i >= 0);
  assert(b->stack_i < BUILDER_MAX);
  b->stack[b->stack_i] = f;
  b->stack_i++;
}

struct frag *pop(struct builder *b) {
  assert(b->stack_i > 0);
  b->stack_i--;
  return b->stack[b->stack_i];
}

struct nfa *build_done(struct builder *b) {
  assert(b->stack_i == 1);
  struct frag *f = pop(b);
  uint32_t s = state(b->nfa, STATE_MATCH, UINT32_MAX, UINT32_MAX);
  patch(b->nfa, f->out, s);
  struct nfa *nfa = b->nfa;
  nfa->start = f->start;
  frag_free(f);
  free(b);
  return nfa;
}

// push fragment that recognizes a character
void build_char(struct builder *b, uint8_t c) {
  uint32_t s = state(b->nfa, c, UINT32_MAX, UINT32_MAX);
  push(b, frag(s, s, s + 1, vec(s)));
}

// push concat of top two fragments
void build_concat(struct builder *b) {
  assert(b->stack_i >= 2);
  struct frag *f2 = pop(b);
  struct frag *f1 = pop(b);
  patch(b->nfa, f1->out, f2->start);
  push(b, frag(f1->start, f1->min, f2->max, f2->out));
  frag_free(f1);
  frag_free(f2);
}

// concat top n fragments into one frag
void build_concatn(struct builder *b, int n) {
  assert(n >= 1);
  assert(b->stack_i >= n);
  struct frag *base = b->stack[b->stack_i - n]; // first frag in concat

  // concat each subsequent frag onto base
  for (int i = b->stack_i - n + 1; i < b->stack_i; i++) {
    struct frag *f = b->stack[i];
    patch(b->nfa, base->out, f->start);
    base->max = f->max;
    base->out = f->out;
    frag_free(f);
  }

  b->stack_i -= n - 1;
}

// push union of top two fragments
void build_union(struct builder *b) {
  assert(b->stack_i >= 2);
  struct frag *f2 = pop(b);
  struct frag *f1 = pop(b);
  //printf("%p %p\n", f1, f2);
  uint32_t s = state(b->nfa, STATE_SPLIT, f1->start, f2->start);
  push(b, frag(s, f1->min, s + 1, append(f1->out, f2->out)));
  frag_free(f1);
  frag_free(f2);
}

// top fragment 0/1 times
void build_optional(struct builder *b) {
  assert(b->stack_i >= 1);
  struct frag *f = pop(b);
  uint32_t s = state(b->nfa, STATE_SPLIT, f->start, UINT32_MAX);
  push(b, frag(s, f->min, s + 1, add(f->out, s)));
  frag_free(f);
}

// top fragment 0+ times
void build_star(struct builder *b) {
  assert(b->stack_i >= 1);
  struct frag *f = pop(b);
  uint32_t s = state(b->nfa, STATE_SPLIT, f->start, UINT32_MAX);
  patch(b->nfa, f->out, s);
  push(b, frag(s, f->min, s + 1, vec(s)));
  frag_free(f);
}

// top fragment 1+ times
void build_plus(struct builder *b) {
  assert(b->stack_i >= 1);
  struct frag *f = pop(b);
  uint32_t s = state(b->nfa, STATE_SPLIT, f->start, UINT32_MAX);
  patch(b->nfa, f->out, s);
  push(b, frag(f->start, f->min, s + 1, vec(s)));
  frag_free(f);
}

// repeat top fragment exactly n times
void build_repeat(struct builder *b, int n) {
  assert(b->stack_i >= 1);
  assert(n >= 1);
  struct frag *f = pop(b);
  push(b, f);
  for (int i = 0; i < n - 1; i++) {
    push(b, dup(b->nfa, f));
  }
  build_concatn(b, n);
}

// repeat top fragment at least n times
void build_repatleast(struct builder *b, int n) {
  assert(b->stack_i >= 1);
  assert(n >= 1);
  struct frag *f = pop(b);
  push(b, f);
  for (int i = 0; i < n - 1; i++) {
    push(b, dup(b->nfa, f));
  }
  build_plus(b);
  build_concatn(b, n);
}

// repeat top fragment at most n times
void build_repatmost(struct builder *b, int n) {
  assert(b->stack_i >= 1);
  assert(n >= 1);
  struct frag *last = pop(b);
  uint32_t s = state(b->nfa, STATE_SPLIT, last->start, UINT32_MAX);
  struct frag *opt = frag(s, last->min, s + 1, vec(s));
  for (int i = 1; i < n; i++) {
    struct frag *f = dup(b->nfa, last);
    uint32_t s = state(b->nfa, STATE_SPLIT, f->start, UINT32_MAX);
    patch(b->nfa, last->out, s);
    opt->max = s + 1;
    opt->out = add(opt->out, s);
    frag_free(last);
    last = f;
  }
  opt->out = append(last->out, opt->out);
  frag_free(last);
  push(b, opt);
}

// repeat top fragment n to m times
void build_reprange(struct builder *b, int n, int m) {
  assert(b->stack_i >= 1);
  assert(n >= 1);
  assert(m > n);
  struct frag *f = pop(b);

  // first, generate n repeats
  push(b, f);
  for (int i = 0; i < n - 1; i++) {
    push(b, dup(b->nfa, f));
  }

  // then, generate n - m optional repeats (as one frag)
  push(b, dup(b->nfa, f));
  build_repatmost(b, m - n);

  build_concatn(b, n + 1);
}

// push fragment that recognizes a charset
// TODO use nodes that recognize whole charsets
void build_charset(struct builder *b, uint8_t *cs) {
  int built = 0;
  for (int i = 0; i < 256; i++) {
    if (cs[i]) {
      build_char(b, i);
      if (built) {
        build_union(b);
      } else {
        built = 1;
      }
    }
  }
}

struct vec *addstate(struct nfa *nfa, uint32_t s, uint32_t gen, uint32_t *id, struct vec *next) {
  uint32_t addstack[nfa->n * 2];
  uint32_t stack_i = 0;

  addstack[stack_i++] = s;
  while (stack_i != 0) {
    assert(stack_i < nfa->n);
    uint32_t a = addstack[--stack_i];
    //printf("adding (%d) c=%d out1=%d out2=%d\n", a, nfa->c[a], nfa->out1[a], nfa->out2[a]);
    if (id[a] == gen) {
      continue;
    }
    id[a] = gen;
    if (nfa->c[a] == STATE_SPLIT) {
      addstack[stack_i++] = nfa->out1[a];
      addstack[stack_i++] = nfa->out2[a];
      continue;
    }
    next = add(next, a);
  }

  return next;
}

// string recognition
int nfa_match(struct nfa *nfa, char *c) {
  uint32_t *id = calloc(nfa->n, sizeof(uint32_t));
  uint32_t gen = 1;
  struct vec *cur = vec(UINT32_MAX);
  struct vec *next = vec(UINT32_MAX);
  struct vec *t;

  cur->n = 0;
  cur = addstate(nfa, nfa->start, gen, id, cur);

  /*printf("state: ");
  for (int i = 0; i < cur->n; i++) {
    printf("%d ", cur->s[i]);
  }
  printf("\n");*/

  int res = 0;

  while (*c) {
    gen++;
    next->n = 0;
    for (int i = 0; i < cur->n; i++) {
      uint32_t s = cur->s[i];
      if (nfa->c[s] == *c) {
        next = addstate(nfa, nfa->out1[s], gen, id, next);
      }
    }

    /*printf("state: ");
    for (int i = 0; i < cur->n; i++) {
      printf("%d ", cur->s[i]);
    }
    printf("\n");*/

    // early stop criterion: no live states remaining
    if (next->n == 0) {
      goto done;
    }

    t = cur;
    cur = next;
    next = t;
    c++;
  }

  for (int i = 0; i < cur->n; i++) {
    if (nfa->c[cur->s[i]] == STATE_MATCH) {
      res = 1;
      goto done;
    }
  }

done:
  free(id);
  free(cur);
  free(next);

  //printf("match: %d\n", res);
  return res;
}

void cleanup() {
  frag_cleanup();
  vec_cleanup();
  nfa_cleanup();
}
