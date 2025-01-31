import os

file = 'guy.txt'
f = open(file,'r')

table = []
file = []
switch = False
for line in f:
  if "table" in line:
    continue
  if "file:" in line:
    switch = True
  if switch:
    file.append(line)
  else:
    table.append(line)

diffs = []

for guy in table:
  if guy not in file:
    diffs.append(guy)

print(diffs)
