
def value(val, path, file):
    s = str(val)
    s = s.replace('+/-', ' \\pm ') # plus minus with latex notation
    s = s.replace('.', ',') # german comma, can be ignored in latex setup
    s = s.replace('(', '') # remove parenthesis
    s = s.replace(')', '') # remove parenthesis
    with open(path + ("" if path[-1] == "/" else "/") + "%s.txt" % file, 'w+') as f:
        f.write(s)

def SI(val, unit="", path="dat", file="tmp"):
    value("\\SI{%s}{%s}" % (val, unit), path, file)

def table(df, file, header=None, text=[]):
    if header is None:
        header = ["c"]*len(df.colums)
    text = [False if i in text else False for i in range(len(df.colums))]
    with open(root + data_out + '%s.txt' % file, 'w') as f:
        f.write("\\begin{tabular}{" + " | ".join(header) + "} \\toprule \n")
        f.write(" & ".join(df.keys())+"\\\\ \\midrule")
        for i, row in df.iterrows():
            k = "\n" + " & ".join(["$%s$" % str(x) if b else str(x) for x,b in zip(row.values, text)]) + " \\\\"
            k = k.replace("+/-"," \\pm ").replace(".",",").replace("$$","")
            f.write(k)
        f.write("\\bottomrule\n\\end{tabular}")
        