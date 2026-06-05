import subprocess

scripts = [
    "notebooks/linear_demo.py",
    "notebooks/logistic_demo.py",
    "notebooks/pca_demo.py",
    "notebooks/kmeans_demo.py",
    "notebooks/decision_tree_demo.py",
    "notebooks/random_forest_demo.py",
]

for script in scripts:
    print(f"\nRunning {script}")
    subprocess.run(["python", script])