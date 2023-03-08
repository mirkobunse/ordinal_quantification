import argparse, os
import numpy as np
import pandas as pd

ROWS = [ # specify the order
    'SWD',  #23.048
    'ESL',  # 5.395  -->5 clases: 3,4,5,6,7
    'LEV',  #11.023
    'cement_strength_gago',  # 44.142
    'stock.ord',  # 53.455
    'auto.data.ord_chu',  # 10.475
    'bostonhousing.ord_chu',  # 33.853
    'californiahousing_gago',  # 915.165
    'winequality-red_gago',  # 87.078
    'winequality-white_gago_rev',  #244.255
    'skill_gago',  # 538.764
    'SkillCraft1_rev_7clases',  # 424.998
    'kinematics_gago',  # 84.0114
    'SkillCraft1_rev_8clases',  # 398.771
    'ERA',  # 12.290
    'ailerons_gago',   # 1.916.322
    'abalone.ord_chu',  # 210.760
]
COLUMNS = [
    'dataset',
    'CC',
    'AC_L2',
    'AC_Ord',
    'PCC',
    'PAC_L2',
    'EDX',
    'CvMy_Eu',
    'EDy_Eu',
    'EDy_EMD',
    'HDX',
    'HDy',
    'PDF_L2',
    'PDF_EMD',
]

def main(csv_path):
    df = pd.read_csv(csv_path, index_col=None)
    df = df[df["error"] == "emd_score"] # select the metric to print
    df = df.drop(columns=["decomposer", "error"]) # ignore constant columns
    df = df.set_index("dataset").loc[ROWS].reset_index()[COLUMNS] # order rows & columns
    df = pd.concat(( # add an "average" row
        df,
        pd.concat((pd.Series({"dataset": "average"}), df[COLUMNS[1:]].mean())).to_frame().T
    ))
    print(df)
    tex_lines = [
        "\\begin{tabular}{l" + "c"*(len(df.columns)-1) + "}",
        "\\toprule",
        " & ".join(df.columns) + " \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        values = [ v if isinstance(v, str) else f"{v:.4f}" for v in r ]
        values[1 + np.argmax(r[1:])] = "\\textbf{" + values[1 + np.argmax(r[1:])] + "}"
        tex_lines.append(" & ".join(values) + " \\\\")
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "" # empty line at the end of the file
    ])
    tex_path = os.path.splitext(csv_path)[0] + ".tex"
    print("Writing to", tex_path)
    with open(tex_path, "w") as f: # store the Tikz code in a .tex or .tikz file
        f.write("\n".join(tex_lines))

# command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert a CSV of Castano to a LaTeX table")
    parser.add_argument("csv_path", help="path of an input CSV")
    args = parser.parse_args()
    main(csv_path=args.csv_path)
