#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "regex.c"

#define BUFSIZE 1024

#define REGEX_CONCAT 0
#define REGEX_CONCATN 1
#define REGEX_UNION 2
#define REGEX_OPTIONAL 3
#define REGEX_STAR 4
#define REGEX_PLUS 5
#define REGEX_REPEAT 6
#define REGEX_REPATLEAST 7
#define REGEX_REPATMOST 8
#define REGEX_REPRANGE 9
#define REGEX_CHAR 10
#define REGEX_CHARSET 11

struct regex_op {
  uint8_t type;
  union {
    uint8_t n;
    struct { uint8_t n; uint8_t m; } pair;
    uint8_t *cs;
  } v;
  struct regex_op *next;
};

static struct regex_op *op_freelist = NULL;

struct regex_op *op_get() {
  struct regex_op *op;
  if (op_freelist != NULL) {
    op = op_freelist;
    op_freelist = op_freelist->next;
  } else {
    op = malloc(sizeof(struct regex_op));
  }
  op->next = NULL;
  return op;
}

void op_put(struct regex_op *op) {
  op->next = op_freelist;
  op_freelist = op;
}

void op_cleanup() {
  while (op_freelist != NULL) {
    struct regex_op *op = op_freelist;
    op_freelist = op_freelist->next;
    free(op);
  }
}

struct regex {
  uint32_t len;
  uint32_t size;
  struct regex *next;
  struct regex_op *ops[];
};

struct regex *regex_freelist = NULL;

struct regex *regex() {
  struct regex *r;
  if (regex_freelist == NULL) {
    r = malloc(sizeof(struct regex) + 2 * sizeof(struct regex_op *));
    r->size = 2;
  } else {
    r = regex_freelist;
    regex_freelist = regex_freelist->next;
  }
  r->len = 0;
  r->next = NULL;
  return r;
}

struct regex *regex_add(struct regex *r, struct regex_op *op) {
  if (r->len >= r->size) {
    r->size *= 2;
    r = realloc(r, sizeof(struct regex) + r->size * sizeof(struct regex_op *));
  }
  r->ops[r->len++] = op;
  return r;
}

void regex_free(struct regex *r) {
  for (int i = 0; i < r->len; i++) {
    struct regex_op *op = r->ops[i];
    if (op->type == REGEX_CHARSET) {
      free(op->v.cs);
    }
    op_put(op);
  }
  r->next = regex_freelist;
  regex_freelist = r;
}

void regex_cleanup() {
  while (regex_freelist != NULL) {
    struct regex *r = regex_freelist;
    regex_freelist = regex_freelist->next;
    free(r);
  }
}

struct nfa *regex_run(struct regex *r) {
  struct builder *b = builder();
  for (int i = 0; i < r->len; i++) {
    struct regex_op *op = r->ops[i];
    switch (op->type) {
      case REGEX_CONCAT:
        build_concat(b);
        break;
      case REGEX_CONCATN:
        build_concatn(b, op->v.n);
        break;
      case REGEX_UNION:
        build_union(b);
        break;
      case REGEX_OPTIONAL:
        build_optional(b);
        break;
      case REGEX_STAR:
        build_star(b);
        break;
      case REGEX_PLUS:
        build_star(b);
        break;
      case REGEX_REPEAT:
        build_repeat(b, op->v.n);
        break;
      case REGEX_REPATLEAST:
        build_repatleast(b, op->v.n);
        break;
      case REGEX_REPATMOST:
        build_repatmost(b, op->v.n);
        break;
      case REGEX_REPRANGE:
        build_reprange(b, op->v.pair.n, op->v.pair.m);
        break;
      case REGEX_CHAR:
        build_char(b, op->v.n);
        break;
      case REGEX_CHARSET:
        build_charset(b, op->v.cs);
        break;
      default:
        exit(1);
        break;
    }
  }
  return build_done(b);
}

uint8_t rand_char() {
  uint8_t c = rand() % 62;
  if (0 <= c && c < 10) {
    return '0' + c;
  } else if (10 <= c && c < 36) {
    return 'A' + c - 10;
  } else {
    return 'a' + c - 36;
  }
}

struct regex *regex_sample_comp_expr(struct regex *r) {
  struct regex_op *op = op_get();

  int l;
  uint8_t *cs;
  switch (rand() % 3) {
    case 0:
      op->type = REGEX_CHAR;
      op->v.n = rand_char();
      break;
    case 1:
      l = rand() % 3 + 2;
      cs = calloc(256, sizeof(uint8_t));
      for (int i = 0; i < l; i++) {
        cs[rand_char()] = 1;
      }
      op->type = REGEX_CHARSET;
      op->v.cs = cs;
      break;
    case 2:
      cs = calloc(256, sizeof(uint8_t));
      switch (rand() % 3) {
        case 0:
          for (int c = '0'; c <= '9'; c++) {
            cs[c] = 1;
          }
          break;
        case 1:
          for (int c = 'A'; c <= 'Z'; c++) {
            cs[c] = 1;
          }
          break;
        case 2:
          for (int c = 'a'; c <= 'z'; c++) {
            cs[c] = 1;
          }
          break;
      }
      op->type = REGEX_CHARSET;
      op->v.cs = cs;
      break;
  }

  r = regex_add(r, op);
  return r;
}

struct regex *regex_sample_basic_comp(struct regex *r) {
  r = regex_sample_comp_expr(r);

  struct regex_op *op;
  switch (rand() % 4) {
    case 0:
      op = op_get();
      op->type = REGEX_REPEAT;
      op->v.n = rand() % 5 + 1;
      r = regex_add(r, op);
      break;
    case 1:
      op = op_get();
      op->type = REGEX_REPATLEAST;
      op->v.n = rand() % 5 + 1;
      r = regex_add(r, op);
      break;
    case 2:
      op = op_get();
      op->type = REGEX_REPRANGE;
      op->v.pair.n = rand() % 2 + 1;
      op->v.pair.m = rand() % 3 + 3;
      r = regex_add(r, op);
      break;
  }

  return r;
}

struct regex *regex_sample_macro_comp(struct regex *r) {
  r = regex_sample_basic_comp(r);
  r = regex_sample_basic_comp(r);

  struct regex_op *op = op_get();
  op->type = REGEX_UNION;
  r = regex_add(r, op);

  return r;
}

struct regex *regex_sample_comp(struct regex *r) {
  int choice = rand() % 4;
  if (rand() % 2) {
    r = regex_sample_basic_comp(r);
  } else {
    r = regex_sample_macro_comp(r);
  }

  if (rand() % 2) {
    struct regex_op *op = op_get();
    op->type = REGEX_OPTIONAL;
    r = regex_add(r, op);
  }

  return r;
}

// for now using simplified distribution
struct regex *regex_sample() {
  struct regex *r = regex();

  int choice = rand() % 10;
  int comps;
  if (choice < 4) {
    comps = 2;
  } else if (choice < 7) {
    comps = 3;
  } else if (choice < 9) {
    comps = 4;
  } else {
    comps = 5;
  }

  for (int i = 0; i < comps; i++) {
    r = regex_sample_comp(r);
  }
  struct regex_op *op = op_get();
  op->type = REGEX_CONCATN;
  op->v.n = comps;
  r = regex_add(r, op);

  return r;
}

void regex_print(struct regex *r) {
  for (int i = 0; i < r->len; i++) {
    struct regex_op *op = r->ops[i];
    switch (op->type) {
      case REGEX_CONCAT:
        printf(".");
        break;
      case REGEX_CONCATN:
        printf("(concat %d)", op->v.n);
        break;
      case REGEX_UNION:
        printf("|");
        break;
      case REGEX_OPTIONAL:
        printf("?");
        break;
      case REGEX_STAR:
        printf("*");
        break;
      case REGEX_PLUS:
        printf("+");
        break;
      case REGEX_REPEAT:
        printf("{%d}", op->v.n);
        break;
      case REGEX_REPATLEAST:
        printf("{%d,}", op->v.n);
        break;
      case REGEX_REPATMOST:
        printf("{,%d}", op->v.n);
        break;
      case REGEX_REPRANGE:
        printf("{%d,%d}", op->v.pair.n, op->v.pair.m);
        break;
      case REGEX_CHAR:
        printf("%c", op->v.n);
        break;
      case REGEX_CHARSET:
        printf("[");
        int dash = 0;
        for (int i = 0; i < 255; i++) {
          if (dash && op->v.cs[i] && !op->v.cs[i + 1]) {
            printf("%c", i);
            dash = 0;
            continue;
          }

          if (!dash && op->v.cs[i] && op->v.cs[i + 1]) {
            printf("%c-", i);
            dash = 1;
            continue;
          } else if (!dash && op->v.cs[i] && !op->v.cs[i + 1]) {
            printf("%c", i);
            continue;
          }
        }
        // we don't want to print char 256 anyway
        printf("]");
        break;
      default:
        exit(1);
        break;
    }
  }
  printf("\n");
}

// convert regex to infix representation
char *regex_infix(struct regex *r) {
  char *stack[BUFSIZE];
  int stack_i = 0;
  int l;
  for (int i = 0; i < r->len; i++) {
    assert(stack_i < BUFSIZE);
    struct regex_op *op = r->ops[i];
    char *s, *s1, *s2;
    switch (op->type) {
      case REGEX_CONCAT:
        s1 = stack[--stack_i];
        s2 = stack[--stack_i];
        s = malloc(BUFSIZE);
        l = snprintf(s, BUFSIZE, "%s%s", s1, s2);
        assert(l < BUFSIZE);
        free(s1);
        free(s2);
        stack[stack_i++] = s;
        break;
      case REGEX_CONCATN:
        s = malloc(BUFSIZE);
        l = 0;
        for (int i = 0; i < op->v.n; i++) {
          s1 = stack[--stack_i];
          l += snprintf(s + l, BUFSIZE - l, "%s", s1);
          assert(l < BUFSIZE);
          free(s1);
        }
        stack[stack_i++] = s;
        break;
      case REGEX_UNION:
        s1 = stack[--stack_i];
        s2 = stack[--stack_i];
        s = malloc(BUFSIZE);
        l = snprintf(s, BUFSIZE, "(%s|%s)", s1, s2);
        assert(l < BUFSIZE);
        free(s1);
        free(s2);
        stack[stack_i++] = s;
        break;
      case REGEX_OPTIONAL:
        s1 = stack[--stack_i];
        s = malloc(BUFSIZE);
        l = snprintf(s, BUFSIZE, "(%s)?", s1);
        assert(l < BUFSIZE);
        free(s1);
        stack[stack_i++] = s;
        break;
      case REGEX_STAR:
        s1 = stack[--stack_i];
        s = malloc(BUFSIZE);
        l = snprintf(s, BUFSIZE, "(%s)*", s1);
        assert(l < BUFSIZE);
        free(s1);
        stack[stack_i++] = s;
        break;
      case REGEX_PLUS:
        s1 = stack[--stack_i];
        s = malloc(BUFSIZE);
        l = snprintf(s, BUFSIZE, "(%s)+", s1);
        assert(l < BUFSIZE);
        free(s1);
        stack[stack_i++] = s;
        break;
      case REGEX_REPEAT:
        s1 = stack[--stack_i];
        s = malloc(BUFSIZE);
        l = snprintf(s, BUFSIZE, "(%s){%d}", s1, op->v.n);
        assert(l < BUFSIZE);
        free(s1);
        stack[stack_i++] = s;
        break;
      case REGEX_REPATLEAST:
        s1 = stack[--stack_i];
        s = malloc(BUFSIZE);
        l = snprintf(s, BUFSIZE, "(%s){%d,}", s1, op->v.n);
        assert(l < BUFSIZE);
        free(s1);
        stack[stack_i++] = s;
        break;
      case REGEX_REPATMOST:
        s1 = stack[--stack_i];
        s = malloc(BUFSIZE);
        l = snprintf(s, BUFSIZE, "(%s){,%d}", s1, op->v.n);
        assert(l < BUFSIZE);
        free(s1);
        stack[stack_i++] = s;
        break;
      case REGEX_REPRANGE:
        s1 = stack[--stack_i];
        s = malloc(BUFSIZE);
        l = snprintf(s, BUFSIZE, "(%s){%d,%d}", s1, op->v.pair.n, op->v.pair.m);
        assert(l < BUFSIZE);
        free(s1);
        stack[stack_i++] = s;
        break;
      case REGEX_CHAR:
        s = malloc(BUFSIZE);
        snprintf(s, BUFSIZE, "%c", op->v.n);
        stack[stack_i++] = s;
        break;
      case REGEX_CHARSET:
        s = malloc(BUFSIZE);
        l = 0;
        l += snprintf(s, BUFSIZE - l, "[");
        int dash = 0;
        for (int i = 0; i < 255; i++) {
          if (dash && op->v.cs[i] && !op->v.cs[i + 1]) {
            l += snprintf(s + l, BUFSIZE - l, "%c", i);
            assert(l < BUFSIZE);
            dash = 0;
            continue;
          }

          if (!dash && op->v.cs[i] && op->v.cs[i + 1]) {
            l += snprintf(s + l, BUFSIZE - l, "%c-", i);
            assert(l < BUFSIZE);
            dash = 1;
            continue;
          } else if (!dash && op->v.cs[i] && !op->v.cs[i + 1]) {
            l += snprintf(s + l, BUFSIZE - l, "%c", i);
            assert(l < BUFSIZE);
            continue;
          }
        }
        // we don't want to print char 256 anyway
        l += snprintf(s + l, BUFSIZE - l, "]");
        assert(l < BUFSIZE);

        stack[stack_i++] = s;
        break;
      default:
        exit(1);
        break;
    }
  }
  assert(stack_i == 1);
  return stack[0];
}

int main(int argc, char **argv) {
  srand(1);

  // parse flags
  int samples = -1;
  int time_limit = -1;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-s") == 0) {
      // -s flag: how many regex samples to try
      if (i + 1 != argc) {
        samples = atoi(argv[i + 1]);
        i++;
      } else {
        printf("missing number after -s\n");
        exit(1);
      }
    } else if (strcmp(argv[i], "-t") == 0) {
      // -t flag: timeout in ms
      if (i + 1 != argc) {
        time_limit = atoi(argv[i + 1]);
        i++;
      } else {
        printf("missing number after -t\n");
        exit(1);
      }
    } else {
      printf("unrecognized flag %s\n", argv[i]);
      exit(1);
    }
  }

  fprintf(stderr, "time_limit %d\n", time_limit);
  fflush(stderr);

  while (1) {
    // read constraints from input
    // constraints are of the form
    // [+|-] <string>
    char buf[BUFSIZE];
    if (!fgets(buf, BUFSIZE, stdin)) {
      exit(1);
    }
    if (strcmp(buf, "exit\n") == 0) {
      goto exit;
    }
    int n = atoi(buf); // number of constraints
    if (n >= BUFSIZE) {
      exit(1);
    }

    char *pos[BUFSIZE]; // positive constraints
    int n_pos = 0;
    char *neg[BUFSIZE]; // negative constraints
    int n_neg = 0;
    for (int i = 0; i < n; i++) {
      if (!fgets(buf, BUFSIZE, stdin)) {
        fprintf(stderr, "failed to read line, error %s (%d)",
          strerror(errno), errno);
        exit(1);
      }
      if (buf[0] != '+' && buf[0] != '-') {
        fprintf(stderr, "char 1 should be '+' or '-', got %c", buf[0]);
        exit(1);
      }
      if (buf[1] != ' ') {
        fprintf(stderr, "char 2 should be ' ', got %c", buf[0]);
        exit(1);
      }

      // copy constraint into buffer
      int len = strlen(&buf[2]);
      char *cons = malloc(sizeof(char) * (len + 1));
      strcpy(cons, &buf[2]);
      cons[len - 1] = '\0';

      if (buf[0] == '+') {
        assert(n_pos < BUFSIZE);
        pos[n_pos++] = cons;
      } else {
        assert(n_neg < BUFSIZE);
        neg[n_neg++] = cons;
      }
    }

    fprintf(stderr, "positive constraints (%d):\n", n_pos);
    for (int i = 0; i < n_pos; i++) {
      fprintf(stderr, "%s\n", pos[i]);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "negative constraints (%d):\n", n_neg);
    for (int i = 0; i < n_neg; i++) {
      fprintf(stderr, "%s\n", neg[i]);
    }
    fflush(stderr);

    // begin enumerating regex
    int trial = 0;
    int num_found = 0;

    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    uint64_t start_time = (uint64_t) tv.tv_sec * 1000000000 + tv.tv_nsec;

    while (1) {
      struct regex *r = regex_sample();
      struct nfa *nfa = regex_run(r);

      // does this regex match all constraints?
      int found = 1;
      for (int i = 0; i < n_pos; i++) {
        if (!nfa_match(nfa, pos[i])) {
          found = 0;
          goto done;
        }
      }

      for (int i = 0; i < n_neg; i++) {
        if (nfa_match(nfa, neg[i])) {
          found = 0;
          goto done;
        }
      }

done:
      nfa_free(nfa);

      if (found) {
        char *s = regex_infix(r);
        printf("%s\n", s);
        fflush(stdout);
        free(s);
        num_found++;
      }

      regex_free(r);

      trial++;
      if (samples != -1 && trial >= samples) {
        break;
      }
      if (time_limit != -1) {
        clock_gettime(CLOCK_MONOTONIC, &tv);
        uint64_t time = (uint64_t) tv.tv_sec * 1000000000 + tv.tv_nsec;
        if ((time - start_time) / 1000000 > time_limit) {
          break;
        }
      }
    }
    printf("\n");
    fflush(stdout);

    for (int i = 0; i < n_pos; i++) {
      free(pos[i]);
    }
    for (int i = 0; i < n_neg; i++) {
      free(neg[i]);
    }
  }

exit:
  cleanup();
  op_cleanup();
  regex_cleanup();
  return 0;
}
