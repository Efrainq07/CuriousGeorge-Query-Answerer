TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
LR=6e-04                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=3        # Batch size.
NUM_NODES=16					 # Number of machines
ROBERTA_PATH="path/to/roberta.base/model.pt" #Path to the original roberta model
CHECKPOINT_PATH="path/to/checkpoints" #Directory to store the checkpoints
UPDATE_FREQ=`expr 784 / $NUM_NODES` # Increase the batch size

DATA_DIR=../Data

#Path to the preprocessed KE dataset, each item corresponds to a data directory for one epoch
KE_DATA=$DATA_DIR/KEI/KEI1_0:$DATA_DIR/KEI/KEI1_1:$DATA_DIR/KEI/KEI1_2:$DATA_DIR/KEI/KEI1_3:$DATA_DIR/KEI/KEI3_0:$DATA_DIR/KEI/KEI3_1:$DATA_DIR/KEI/KEI3_2:$DATA_DIR/KEI/KEI3_3:$DATA_DIR/KEI/KEI5_0:$DATA_DIR/KEI/KEI5_1:$DATA_DIR/KEI/KEI5_2:$DATA_DIR/KEI/KEI5_3:$DATA_DIR/KEI/KEI7_0:$DATA_DIR/KEI/KEI7_1:$DATA_DIR/KEI/KEI7_2:$DATA_DIR/KEI/KEI7_3:$DATA_DIR/KEI/KEI9_0:$DATA_DIR/KEI/KEI9_1:$DATA_DIR/KEI/KEI9_2:$DATA_DIR/KEI/KEI9_3:

DIST_SIZE=`expr $NUM_NODES \* 4`

fairseq-train $DATA_DIR/MLM \                #Path to the preprocessed MLM datasets
        --user-dir .                           # Load local models, tasks, and criterions
        --KEdata $KE_DATA \                      #Path to the preprocessed KE datasets
        --restore-file $ROBERTA_PATH \
        --save-dir $CHECKPOINT_PATH \
        --max-sentences $MAX_SENTENCES \
        --tokens-per-sample 512 \
        --task MLMetKE \                     
        --sample-break-mode complete \
        --required-batch-size-multiple 1 \
        --arch roberta_base \
        --criterion MLMetKE \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --update-freq $UPDATE_FREQ \
        --negative-sample-size 1 \ # Negative sampling size (one negative head and one negative tail)
        --ke-model TransE \ 
        --init-token 0 \
        --separator-token 2 \
        --gamma 4 \        # Margin of the KE objective
        --nrelation 822 \
        --skip-invalid-size-inputs-valid-test \
        --fp16 --fp16-init-scale 2 --threshold-loss-scale 1 --fp16-scale-window 128 \
        --reset-optimizer --distributed-world-size ${DIST_SIZE} --ddp-backend no_c10d --distributed-port 23456 \
        --log-format simple --log-interval 1 \
        #--relation-desc  #Add this option to encode the relation descriptions as relation embeddings (KEPLER-Rel in the paper)