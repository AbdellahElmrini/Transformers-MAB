class TransformerConfig:
    attn_dropout = 0.1
    embed_dropout = 0.1
    ff_dropout = 0.1
    def __init__(
        self, vocab_size, max_len, **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        for key, value in kwargs.items():
            setattr(self, key, value)

class LightConfig(TransformerConfig):
    n_heads = 2
    n_blocks = 3
    n_embd = 4

class TrainConfig():
    def __init__(self, lr, epochs, **kwargs):
        self.lr = lr
        self.epochs = epochs
        for key, value in kwargs.items():
            setattr(self, key, value)
