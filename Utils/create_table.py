import sys
import argparse
import numpy as np

def main(args):
    parser = argparse.ArgumentParser(description='create results table for deep learning models')
    parser.add_argument('-file', help='''location of results file from multi_class.py
                                         or one_vs_all.py''')
    parser.add_argument('-outfile', default=None)

    args = vars(parser.parse_args())
    results_file = args['file']
    outfile = args['outfile']

    table_header = """\\begin{table*}[t]
  \centering
  \\renewcommand*{\\arraystretch}{0.9}
  \setlength\\tabcolsep{1.8mm}
  \\begin{tabular}{l|lrrrrrrrrrrrrrrr}
    \\toprule
    \multicolumn{2}{c}{} & \multicolumn{15}{c}{Results for Treshold} \\\\
    \cmidrule(l){3-17}
    \multicolumn{2}{c}{} & \multicolumn{3}{c}{0.0} & \multicolumn{3}{c}{0.33} & \multicolumn{3}{c}{0.5} & \multicolumn{3}{c}{0.66} & \multicolumn{3}{c}{0.99} \\\\
    \cmidrule(r){2-2}\cmidrule(rl){3-5}\cmidrule(rl){6-8}\cmidrule(rl){9-11}\cmidrule(rl){12-14}\cmidrule(rl){15-17}
    \multicolumn{1}{c}{}    & Emotion & P & R & \F & P & R & \F & P & R & \F & P & R & \F & P & R & \F \\\\
                   """

    table_footer = """    \cmidrule(r){2-2}\cmidrule(rl){3-5}\cmidrule(rl){6-8}\cmidrule(rl){9-11}\cmidrule(rl){12-14}\cmidrule(rl){15-17}
    \\bottomrule
  \end{tabular}
  \caption{Results of non-linear models for labels of different
    thresholds. We report the mean precision, recall and f1 for each emotion over 5 runs (standard deviations are included in parenthesis).}
  \label{tab:resultsdeepthresholds}
\end{table*}
                   """

    model_header = """    \cmidrule(r){1-2}\cmidrule(rl){3-5}\cmidrule(rl){6-8}\cmidrule(rl){9-11}\cmidrule(rl){12-14}\cmidrule(rl){15-17}
    \multirow{9}{*}{\\rt{"""
    model_header2 = """}}\n"""

    model_outer = """    \cmidrule(r){2-2}\cmidrule(rl){3-5}\cmidrule(rl){6-8}\cmidrule(rl){9-11}\cmidrule(rl){12-14}\cmidrule(rl){15-17}\n"""
    

    emo_fmt = '    & {0}& {1:.1f} ({2:.1f}) & {3:.1f} ({4:.1f}) & {5:.1f} ({6:.1f}) \\\\'
    emotions = ['Anger', 'Anticipation', 'Disgust',
                'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust',
                'Micro-Avg.']

    file = open(results_file).readlines()

    if outfile:
        with open(outfile, 'a') as out:
            out.write(table_header + '\n')
            for i, l in enumerate(file):
                if l.startswith('+++'):
                    name = l.strip().replace('+','')
                    results = file[i+3:i+12]
                    results = [l.replace('±', '').split()[3:] for l in results]
                    results = [np.array(l, dtype=float) * 100 for l in results]
                    out.write(model_header + name + model_header2)
                    for j, r in enumerate(results):
                        if j != 8:
                            towrite = [emotions[j]] + list(r)
                            out.write(emo_fmt.format(*towrite) + '\n')
                        else:
                            towrite = [emotions[j]] + list(r)
                            out.write(model_outer)
                            out.write(emo_fmt.format(*towrite) + '\n')
                    out.write('\n')
            out.write(table_footer)
    else:
        for i, l in enumerate(file):
            if l.startswith('+++'):
                name = l.strip().replace('+','')
                results = file[i+3:i+12]
                results = [l.replace('±', '').split()[3:] for l in results]
                results = [np.array(l, dtype=float) * 100 for l in results]
                print(name)
                for j, r in enumerate(results):
                    towrite = [emotions[j]] + list(r)
                    print(emo_fmt.format(*towrite))
        print()
        print()

            
if __name__ == '__main__':

    args = sys.argv
    main(args)
        

