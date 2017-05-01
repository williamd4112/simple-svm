MODE=$1
KERNEL=$2
DEGREE=$3
C=$4
NU=$5
python main.py --task train --model $MODE --kernel $KERNEL --deg $DEGREE --nu $NU --c $C --train_X data/X_train.csv --train_T data/T_train.csv --test_X data/X_test.csv --test_T data/T_test.csv
python main.py --task eval --model $MODE --train_X data/X_train.csv --train_T data/T_train.csv --test_X data/X_test.csv --test_T data/T_test.csv
python main.py --task plot --model $MODE --train_X data/X_train.csv --train_T data/T_train.csv --test_X data/X_test.csv --test_T data/T_test.csv
