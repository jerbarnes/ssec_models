import sys, os
import argparse
import numpy as np

def main(args):
    parser = argparse.ArgumentParser(description='create results table for deep learning models')
    parser.add_argument('-results_dir', help='''location of results file from multi_class.py
                                         or one_vs_all.py''')
    parser.add_argument('-outfile', default=None)

    args = vars(parser.parse_args())
    results_dir = args['results_dir']
    outfile = args['outfile']

    table_header = """\\begin{table*}[t]
  \\resizebox{\\textwidth}{!}{
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
  }
  \caption{Results of non-linear models for labels of different
    thresholds. We report the mean precision, recall and f1 for each emotion over 5 runs (standard deviations are included in parenthesis).}
  \label{tab:resultsdeepthresholds}
\end{table*}
                   """

    model_header = """    \cmidrule(r){1-2}\cmidrule(rl){3-5}\cmidrule(rl){6-8}\cmidrule(rl){9-11}\cmidrule(rl){12-14}\cmidrule(rl){15-17}
    \multirow{9}{*}{\\rt{"""
    model_header2 = """}}\n"""

    model_outer = """    \cmidrule(r){2-2}\cmidrule(rl){3-5}\cmidrule(rl){6-8}\cmidrule(rl){9-11}\cmidrule(rl){12-14}\cmidrule(rl){15-17}\n"""
    

    emo_fmt = '    & {0}& {1:.0f} ({2:.1f}) & {3:.0f} ({4:.1f}) & {5:.0f} ({6:.1f}) & {7:.0f} ({8:.1f}) & {9:.0f} ({10:.1f}) & {11:.0f} ({12:.1f}) & {13:.0f} ({14:.1f}) & {15:.0f} ({16:.1f}) & {17:.0f} ({18:.1f}) & {19:.0f} ({20:.1f}) & {21:.0f} ({22:.1f}) & {23:.0f} ({24:.1f}) & {25:.0f} ({26:.1f}) & {27:.0f} ({28:.1f}) & {29:.0f} ({30:.1f}) \\\\'
    emotions = ['Anger', 'Anticipation', 'Disgust',
                'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust',
                'Micro-Avg.']

    results_txts = ['results_0.0.txt', 'results_0.33.txt', 'results_0.5.txt', 'results_0.66.txt', 'results_0.99.txt']
    results_files = [os.path.join(results_dir, f) for f in results_txts]
    files = [open(f).readlines() for f in results_files]

    if outfile:
        with open(outfile, 'a') as out:
            out.write(table_header + '\n')
            for i, l in enumerate(files[0]):
                if l.startswith('+++'):
                    name = l.strip().replace('+','')
                    results = [file[i+3:i+12] for file in files]
                    results = [[l.replace('±', '').split()[3:] for l in result] for result in results]
                    results = [[np.array(l, dtype=float) * 100 for l in result] for result in results]
                    results = np.concatenate(results, axis=1)
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
        for i, l in enumerate(files[0]):
            if l.startswith('+++'):
                name = l.strip().replace('+','')
                results = [file[i+3:i+12] for file in files]
                results = [[l.replace('±', '').split()[3:] for l in result] for result in results]
                results = [[np.array(l, dtype=float) * 100 for l in result] for result in results]
                results = np.concatenate(results, axis=1)
                print(name)
                for j, r in enumerate(results):
                    towrite = [emotions[j]] + list(r)
                    print(emo_fmt.format(*towrite))
        print()
        print()

            
if __name__ == '__main__':

    args = sys.argv
    main(args)
        

