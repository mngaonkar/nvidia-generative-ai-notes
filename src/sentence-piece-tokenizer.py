import sentencepiece as spm

# Equivalent to spm_train CLI flags
spm.SentencePieceTrainer.train(
    input='text_for_tokenizer.txt',  # Your corpus (or 'file1.txt,file2.txt')
    model_prefix='spm_32k_wiki',
    vocab_size=32768,
    model_type='bpe',
    character_coverage=0.9999,
    byte_fallback=True,
    split_digits=True,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    num_threads=64  # Adjust to your cores
)
