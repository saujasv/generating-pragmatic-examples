CC=gcc
CFLAGS=--std=gnu11 -g -O4 -march=native -mtune=native -no-pie
DEPS=regex.c

%: %.c $(DEPS)
	$(CC) -o $@ $< $(CFLAGS)

all: regex_sampler regex_test

clean:
	rm -f regex_sampler regex_test

.PHONY: all clean
