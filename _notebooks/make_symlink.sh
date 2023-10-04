cd /l/users/speech_lab/_SpeechT5PretrainDataset/Finetune/TTS/v1.2/data

folders=(QASRTTS)
length=${#folders[@]}


for ((i=0; i<length; i++)); do
    rm -rf ${folders[$i]}
    mkdir ${folders[$i]}
    case ${folders[$i]} in
    # 'ASC')
    #     data_links=(
    #         /l/users/speech_lab/ArabicSpeechCorpus/ArabicSpeechCorpus16K/train
    #         /l/users/speech_lab/ArabicSpeechCorpus/ArabicSpeechCorpus16K/test
    #         /l/users/speech_lab/ArabicSpeechCorpus/ArabicSpeechCorpus16K/speaker_embedding
    #     )
    #     data_name=(
    #         ASC_train
    #         ASC_test
    #         speaker_embedding
    #     )
    #     ;;
    # 'CLARTTS')
    #     data_links=(
    #         /l/users/speech_lab/ClArTTS[ClArD_dataset]/ClArTTS16K/train
    #         /l/users/speech_lab/ClArTTS[ClArD_dataset]/ClArTTS16K/dev
    #         /l/users/speech_lab/ClArTTS[ClArD_dataset]/ClArTTS16K/speaker_embedding
    #     )
    #     data_name=(
    #         CLARTTS_train
    #         CLARTTS_dev
    #         speaker_embedding
    #     )
    #     ;;
    # 'MGB2')
    #     data_links=(
    #         /l/users/speech_lab/MGB/MGB2/_segmented/dev
    #         /l/users/speech_lab/MGB/MGB2/_segmented/test
    #         /l/users/speech_lab/MGB/MGB2/_segmented/train
    #         /l/users/speech_lab/MGB/MGB2/_segmented/speaker_embeddings
    #     )
    #     data_name=(
    #         MGB2_dev
    #         MGB2_test
    #         MGB2_train
    #         speaker_embedding
    #     )
    #     ;;
    'QASRTTS')
        data_links=(
            /l/users/speech_lab/QASR_TTS/Mahmoud_wav/wavs
            /l/users/speech_lab/QASR_TTS/Khadija_wav/wavs
        )
        data_name=(
            QASRTTS_train
            QASRTTS_train
        )
        ;;
    *)
        echo ${folders[$i]}
        echo Unknown Dataset.
        ;;
    esac

    data_length=${#data_links[@]}

    if [[ ${#data_links[@]} == ${#data_name[@]} ]]; then
        for ((j=0; j<data_length; j++)); do
        echo Symlink for ${data_links[$j]} dataset
        ln -s ${data_links[$j]} ${folders[$i]}/${data_name[$j]}
    done
    else
        echo Unexpected behaviour: number of paths and number of names are not equal
    fi
    
done
