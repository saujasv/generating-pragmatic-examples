#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "regex.c"

int main(int argc, char **argv) {
  srand(1);

  struct timespec tv;
  for (int n = 1; n < 100; n++) {
    clock_gettime(CLOCK_MONOTONIC, &tv);
    uint64_t start = tv.tv_sec * 1000000000 + tv.tv_nsec;

    struct builder *b = builder();
    for (int i = 0; i < n; i++) {
      build_char(b, 'a');
      build_optional(b);
    }
    for (int i = 0; i < n; i++) {
      build_char(b, 'a');
    }
    build_concatn(b, n * 2);
    struct nfa *nfa = build_done(b);

    clock_gettime(CLOCK_MONOTONIC, &tv);
    uint64_t built = tv.tv_sec * 1000000000 + tv.tv_nsec;

    char c[256];
    for (int i = 0; i < n; i++) {
      c[i] = 'a';
    }
    c[n] = '\0';
    assert(nfa_match(nfa, c));

    for (int i = 0; i < n - 1; i++) {
      c[i] = 'a';
    }
    c[n - 1] = '\0';
    assert(!nfa_match(nfa, c));

    nfa_free(nfa);

    clock_gettime(CLOCK_MONOTONIC, &tv);
    uint64_t done = tv.tv_sec * 1000000000 + tv.tv_nsec;

    printf("times for %d: build=%lu match=%lu total=%lu\n", n,
      (built - start),
      (done - built),
      (done - start));
  }

  // fuzz for errors
  for (int test_n = 0; test_n < 1000; test_n++) {
    //printf("test %d\n", test_n);
    struct builder *b = builder();
    for (int op_n = 0; op_n < 50; op_n++) {
      float choice = (float) rand() / RAND_MAX;

      //printf("%d : ", b->stack_i);

      if (b->stack_i == 0) {
        choice *= 0.3;
      } else if (b->stack_i == 1) {
        choice *= 0.7;
      }

      if (choice < 0.15) {
        int c = rand() % 26 + 97;
        //printf("char %d\n", c);
        build_char(b, c);
      } else if (choice < 0.3) {
        uint8_t cs[256];
        for (int i = 0; i < 256; i++) {
          cs[i] = 0;
        }
        //printf("charset ");
        for (int i = 0; i < 4; i++) {
          int c = rand() % 26 + 97;
          cs[c] = 1;
          //printf("%d ", c);
        }
        //printf("\n");
        build_charset(b, cs);
      } else if (choice < 0.7) {
        int n, m;
        int r;
        struct frag *top = b->stack[b->stack_i - 1];
        if (top->max - top->min < 50) {
          r = rand() % 7;
        } else {
          r = rand() % 3;
        }
        switch (r) {
          case 0:
            //printf("optional\n");
            build_optional(b);
            break;
          case 1:
            //printf("star\n");
            build_star(b);
            break;
          case 2:
            //printf("plus\n");
            build_plus(b);
            break;
          case 3:
            n = rand() % 5 + 1;
            //printf("repeat %d\n", n);
            build_repeat(b, n);
            break;
          case 4:
            n = rand() % 5 + 1;
            //printf("repatleast %d\n", n);
            build_repatleast(b, n);
            break;
          case 5:
            n = rand() % 5 + 1;
            //printf("repatmost %d\n", n);
            build_repatmost(b, n);
            break;
          case 6:
            n = rand() % 5 + 1;
            m = n + rand() % 5 + 1;
            //printf("reprange %d %d\n", n, m);
            build_reprange(b, n, m);
            break;
        }
      } else {
        int n;
        switch (rand() % 3) {
          case 0:
            //printf("concat\n");
            build_concat(b);
            break;
          case 1:
            //printf("union\n");
            build_union(b);
            break;
          case 2:
            if (b->stack_i >= 4) {
              n = rand() % 4 + 1;
            } else {
              n = rand() % b->stack_i + 1;
            }
            //printf("concatn %d\n", n);
            build_concatn(b, n);
            break;
        }
      }
    }
    // close out the state
    while (b->stack_i != 1) {
      int n;
      switch (rand() % 3) {
        case 0:
          //printf("concat\n");
          build_concat(b);
          break;
        case 1:
          //printf("union\n");
          build_union(b);
          break;
        case 2:
          if (b->stack_i >= 4) {
            n = rand() % 4 + 1;
          } else {
            n = rand() % b->stack_i + 1;
          }
          //printf("concatn %d\n", n);
          build_concatn(b, n);
          break;
      }
    }
    struct nfa *nfa = build_done(b);
    assert(nfa->n != 0);
    //printf("final nfa: %d states\n", nfa->n);

    nfa_free(nfa);
    //printf("\n");
  }

  cleanup();
  return 0;
}
