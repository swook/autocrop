import os
import re

def filesWithRe(path, regexp):
    files = ['%s/%s' % (path, f) for f in os.listdir(path)]
    return [f for f in files if os.path.isfile(f) and re.match(regexp, f)]
