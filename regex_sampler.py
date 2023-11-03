#!/usr/bin/env python3

import os
import subprocess
import sys

class RegexSampler:
  def __init__(self):
    exe = os.path.dirname(__file__) + "/sampler/regex_sampler"
    self.p = subprocess.Popen([exe, "-t", "2000"],
      stdin=subprocess.PIPE, stdout=subprocess.PIPE)

  def _close(self):
    self._write("exit")
    self.p.wait()

  def _write(self, s):
    s = (s + "\n").encode('UTF-8')
    #print("WRITE", s)
    self.p.stdin.write(s)
    self.p.stdin.flush()

  def _readline(self):
    l = self.p.stdout.readline()
    #print("READ", l)
    return l

  def eval_literal_listener(self, pos_constraints, neg_constraints):
    # write to self.p.stdin
    self._write(str(len(pos_constraints) + len(neg_constraints)))
    for c in pos_constraints:
      self._write("+ " + c)
    for c in neg_constraints:
      self._write("- " + c)

    # read from self.p.stdout
    regexes = []
    while True:
      l = self._readline()
      if l == b"\n":
        break
      regexes.append(l)

    return regexes

  def __del__(self):
    self._close()

if __name__ == '__main__':
  r = RegexSampler()
  l = r.eval_literal_listener(["a", "aa", "aaa"], ["aaaa"])
  print(len(l))
  l = r.eval_literal_listener(["a", "aa", "aaa"], ["aaaa", "aaaaa"])
  print(len(l))
