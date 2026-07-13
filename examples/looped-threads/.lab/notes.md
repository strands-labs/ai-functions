# House Prices — Looped-Threads notes

The lab notebook. Each round reads it first to avoid repeating tried ideas, then
appends one line per experiment. The metric of record is the validation RMSE of
log(SalePrice) — LOWER is better; a low train_rmse beside a high validation RMSE
means overfitting.

Format: `- val_rmse=.. train_rmse=.. | MODEL_NAME=.. | <what changed / lesson>`

## Log

- val_rmse=0.189113 train_rmse=0.135066 | MODEL_NAME=linear | Baseline LinearRegression
- val_rmse=0.189006 train_rmse=0.135083 | MODEL_NAME=ridge a=10 | Added Ridge regularization, marginal improvement
- val_rmse=0.149873 train_rmse=0.062773 | MODEL_NAME=rf n=100 d=10 | Random Forest with depth constraint, major improvement
- val_rmse=0.131598 train_rmse=0.084046 | MODEL_NAME=gb n=100 d=3 lr=.1 | Gradient Boosting, best so far, good train/val balance
- val_rmse=0.138515 train_rmse=0.097253 | MODEL_NAME=hgb d=3 lr=.1 | HistGradientBoosting, good but not as good as GB
- val_rmse=0.985192 train_rmse=0.270578 | MODEL_NAME=mlp 128-64 | MLPRegressor with early stopping, poor performance
- val_rmse=0.141830 train_rmse=0.045321 | MODEL_NAME=et n=100 d=10 | ExtraTrees, better than RF but some overfitting
- val_rmse=0.129069 train_rmse=0.066175 | MODEL_NAME=gb n=200 d=3 lr=.1 | Doubled estimators to 200, improvement over n=100
- val_rmse=0.129169 train_rmse=0.073842 | MODEL_NAME=gb n=200 d=3 lr=.08 | Lower LR (0.08), marginally worse than 0.1
- val_rmse=0.128339 train_rmse=0.054142 | MODEL_NAME=gb n=300 d=3 lr=.1 | Best model! 300 estimators, great balance
- val_rmse=0.134460 train_rmse=0.029578 | MODEL_NAME=gb n=300 d=4 lr=.1 | Deeper trees (d=4), overfitting, worse
- val_rmse=0.134791 train_rmse=0.048895 | MODEL_NAME=vote gb+rf+et | VotingRegressor ensemble, did not help
- val_rmse=0.130497 train_rmse=0.049917 | MODEL_NAME=gb n=300 d=3 lr=.1 sub=.8 | Added subsample=0.8, slightly worse than default
- val_rmse=0.131912 train_rmse=0.047791 | MODEL_NAME=gb n=300 d=3 lr=.12 | Higher LR (0.12), worse than 0.1
- val_rmse=0.128370 train_rmse=0.044794 | MODEL_NAME=gb n=400 d=3 lr=.1 | Increased to 400 estimators, marginally worse
- val_rmse=0.127470 train_rmse=0.056490 | MODEL_NAME=gb n=300 d=3 lr=.1 ms=5 | NEW BEST! Added min_samples_split=5 for regularization
- val_rmse=0.129590 train_rmse=0.057296 | MODEL_NAME=gb n=300 d=3 lr=.1 ms=10 | Too much regularization with ms=10
- val_rmse=0.127999 train_rmse=0.047154 | MODEL_NAME=gb n=400 d=3 lr=.1 ms=5 | Combined n=400 with ms=5, slightly worse than n=300
- val_rmse=0.140518 train_rmse=0.057143 | MODEL_NAME=gb n=300 d=3 lr=.1 ms=5 ml=2 | Adding min_samples_leaf=2 over-regularized
