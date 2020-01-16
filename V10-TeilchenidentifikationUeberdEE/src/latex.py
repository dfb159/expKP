import numpy as np

def value(val, path, file, show=False):
    with open(path + ("" if path[-1] == "/" else "/") + "%s.txt" % file, 'w+') as f:
        f.write(str(val))
    if show:
        print(str(val))

def SI(val, unit="", path="dat", file="tmp", bonus="", show=False):
    value("\\SI[%s]{%s}{%s}" % (bonus,latexify(val), unit), path, file, show)
    
def latexify(val):
    s = str(val)
    s = s.replace('+/-', ' \\pm ') # plus minus with latex notation
    s = s.replace('.', ',') # german comma, can be ignored in latex setup
    s = s.replace('(', '') # remove parenthesis
    s = s.replace(')', '') # remove parenthesis
    return s
        
def table(data, path, file, header=None, leader=None, units=None, horizontal=False):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    a,b = data.shape
    if not units:
        units = [""]*b
    elif isinstance(units, str):
        units = [units]*b
    tab = "\\begin{tabular}{" + ("c|" if leader else "") + "c"*b + "}\n\\toprule\n"
    if header:
        assert len(header) == (b if leader is None else b+1)
        tab += " & ".join(header) + "\\\\ \\midrule\n"
    if leader:
        assert len(leader) == a
    for i in range(len(data)):
        if leader:
            tab += leader[i]+ " & "
        tab += " & ".join(map(lambda x: "\\SI{%s}{%s}" % (latexify(x[0]), x[1]), [(data[i,j], units[i if horizontal else j]) for j in range(b)])) + "\\\\\n"
#        tab += " & ".join(map(lambda x, u: "\\SI{%s}{%s}" % (latexify(x), u), [(data[i,j], units[j]) for j in range(b)])) + "\\\n"
    tab += "\\bottomrule\n\\end{tabular}"
    value(tab, path, file)