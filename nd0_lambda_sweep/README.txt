ND0-only lambda validation sweep (0.1 to 0.9, Conv4 + BiLSTM, clean, 400 epochs)

Copy both .py files to the ROOT of your existing 'new ideal' project, next to run_ideal.py.
Back up your existing run_ideal_text.py before replacing it. No changes to methods/ or data/.

Run from Jupyter in the project root:
%run run_lambda_validation_sweep.py --epochs 400 --bert_dir ./bert_features

The sweep runs 9 experiments sequentially. It skips meta-testing and writes:
lambda_validation_results/validation_runs.csv
lambda_validation_results/validation_comparison.csv
lambda_validation_results/recommended_lambda.txt (only after all 9 finish)

IMPORTANT: If ND0 lambda=0.3 has already completed 400 epochs under the original runner,
399.tar may exist without validation_summary.json. This sweep stops rather than overwriting
that checkpoint. Recover the best validation accuracy and epoch from its original console log,
or back up its checkpoint folder before rerunning it. A prior one-epoch 0.tar is not a blocker.

Use ND0 validation results to select a candidate; do not claim cross-dataset stability.
