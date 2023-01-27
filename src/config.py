class TransformerConfig:
    attn_dropout = 0.3
    embed_dropout = 0.3
    ff_dropout = 0.3
    def __init__(
        self, vocab_size, max_len, **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        for key, value in kwargs.items():
            setattr(self, key, value)

class LightConfig(TransformerConfig):
    n_heads = 4
    n_blocks = 4
    n_embd = 16

class TrainConfig():
    def __init__(self, lr, epochs, **kwargs):
        self.lr = lr
        self.epochs = epochs
        for key, value in kwargs.items():
            setattr(self, key, value)
