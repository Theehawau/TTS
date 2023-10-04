DATASET=CLARTTS

DATA_ROOT=/l/users/speech_lab/_SpeechT5PretrainDataset/Finetune/TTS/hubert_labels/$DATASET
SAVE_DIR=/l/users/speech_lab/_SpeechT5PretrainDataset/Finetune/TTS/_models/$DATASET
TRAIN_SET=train
VALID_SET=valid
LABEL_DIR=/l/users/speech_lab/_SpeechT5PretrainDataset/Finetune/TTS/labels/$DATASET
BPE_TOKENIZER=/l/users/speech_lab/_SpeechT5PretrainDataset/v1/arabic.model
USER_DIR=/home/hawau.toyin/SpeechT5/SpeechT5/speecht5
PT_CHECKPOINT_PATH=/l/users/speech_lab/_SpeechT5PretrainDataset/v1/models/checkpoint_best.pt
WANDB_PROJECT=tts-label-fix
 

# conda activate fseq

fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 1 \
  --distributed-port 0 \
  --ddp-backend pytorch_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1 \
  --fp16 \
  \
  --task speecht5 \
  --t5-task t2s \
  --sample-rate 16000 \
  --num-workers 4 \
  --max-tokens 1000000 \
  --update-freq 4 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --max-tokens-valid 3200000 \
  \
  --criterion speecht5 \
  --use-guided-attn-loss \
  --report-accuracy \
  --sentence-avg \
  \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --dropout 0.15 \
  --activation-dropout 0.15 \
  --attention-dropout 0.15 \
  --encoder-layerdrop 0.0 \
  --decoder-layerdrop 0.0 \
  --weight-decay 0.0 \
  --clip-norm 25.0 \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 \
  --feature-grad-mult 1.0 \
  \
  --max-update 200000 \
  --max-text-positions 600 \
  --min-speech-sample-size 1056 \
  --max-speech-sample-size 320000 \
  --max-speech-positions 1876 \
  --required-batch-size-multiple 1 \
  --skip-invalid-size-inputs-valid-test \
  --keep-last-epochs 3 \
  --validate-after-updates 20000 \
  --validate-interval 50 \
  --log-interval 10 \
  --wandb-project ${WANDB_PROJECT} \
  \
  --arch t5_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --freeze-encoder-updates 20000 \
  --freeze-encoder-updates 60000 \
  \
  --finetune-from-model ${PT_CHECKPOINT_PATH}