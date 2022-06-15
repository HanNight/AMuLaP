# Required environment variables:
# TASK: SST-2 / CoLA / MNLI / QNLI / RTE / MRPC / QQP
# SEED: random seed (13 / 21 / 42 / 87 / 100)
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# MODEL: pre-trained model name (roberta-*, bert-*), see Transformers model list
# K: the number of label tokens to use for each class (1 / 2 / 4 / 8 / 16 / etc.)

# Number of training instances per label
SHOTNUM=16

# Training steps
MAX_STEP=1000

# Warmup steps
WARMUP_STEP=0

# Validation steps
EVAL_STEP=100

# Task specific parameters
# The default length is 128 and the default number of samples is 16.
# For some tasks, we use longer length.
# All those parameters are set arbitrarily by observing the data distributions.
TASK_EXTRA=""
case $TASK in
    CoLA)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        ;;
    SST-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        ;;
    MRPC)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        ;;
    QQP)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        ;;
    MNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        TASK_EXTRA="--max_seq_len 256"
        ;;
    QNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        ;;
    RTE)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;

esac

# Gradient accumulation steps
# For medium-sized GPUs (e.g., 2080ti with 10GB memory), they can only take 
# a maximum batch size of 2 when using large-size models. So we use gradient
# accumulation steps to achieve the same effect of larger batch sizes.
REAL_BS=2
GS=$(expr $BS / $REAL_BS)

# Use a random number to distinguish different trails (avoid accidental overwriting)
TRIAL_IDTF=$RANDOM
DATA_DIR=data/k-shot/$TASK/$SHOTNUM-$SEED

python run_prompt.py \
    --model_name_or_path $MODEL \
    --task_name $TASK \
    --data_dir $DATA_DIR \
    --output_dir outputs \
    --shot_num $SHOTNUM \
    --seed $SEED \
    --max_train_steps $MAX_STEP \
    --num_warmup_steps $WARMUP_STEP \
    --eval_steps $EVAL_STEP \
    --learning_rate $LR \
    --per_device_train_batch_size $REAL_BS \
    --gradient_accumulation_steps $GS \
    --per_device_eval_batch_size 8 \
    --top_k $K \
    --dedup \
    --template $TEMPLATE \
    $TASK_EXTRA