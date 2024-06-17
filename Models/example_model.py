import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# C'est le code du modèle que on a implémenté dans le notebook
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Attention weights of the different heads are stored in the same tensors.

        # WARNING: We put all the attention heads in the same Linear Layers:
        self.values = nn.Linear(self.embed_size, heads * self.head_dim, bias=False)
        self.keys = nn.Linear(self.embed_size, heads * self.head_dim, bias=False)
        self.queries = nn.Linear(self.embed_size, heads * self.head_dim, bias=False)

        # fc_out after having concatenated the results of the attention of each head
        self.fc_out = nn.Linear(self.embed_size, embed_size)


    def forward(self, pre_values, pre_keys, pre_queries, mask):
        """
        pre_values: (N, value_len, embed_size)
        pre_keys: (N, keys_len, embed_size)
        queries: (N, queries_len, embed_size)



        mask: None or (N, heads, query_len, key_len)
        if mask == 0, attention matrix  -> float("-1e20") (big negative value)
        Ignore the mask, it's too difficult at the beginning.
        """
        assert pre_values.shape == pre_keys.shape

        # Get number of training examples
        N = pre_queries.shape[0]

        value_len, key_len, query_len = pre_values.shape[1], pre_keys.shape[1], pre_queries.shape[1]

        # Compute the values keys and queries
        values = self.values(pre_values)  # (N, value_len, embed_size)
        keys = self.keys(pre_keys)  # (N, key_len, embed_size)
        queries = self.queries(pre_queries)  # (N, query_len, embed_size)
        # WARNING: values (resp. keys and queries), are then used by different heads.
        # Each head uses a fraction of the values (resp. keys and queries) tensor.

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Compute the similarity between query*keys
        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for better stability
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)


        # We compute the attention (aka the ponderated value)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.


        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return self.fc_out(out)

    def forward2(self, pre_values, pre_keys, pre_queries, mask):
        """
        pre_values: (N, value_len, embed_size)
        pre_keys: (N, keys_len, embed_size)
        queries: (N, queries_len, embed_size)

        mask: None or (N, heads, query_len, key_len)
        if mask == 0, attention matrix  -> float("-1e20") (big negative value)
        Ignore the mask, it's too difficult at the beginning.
        """
        assert pre_values.shape == pre_keys.shape

        # Get number of training examples
        N = pre_queries.shape[0]

        value_len, key_len, query_len = pre_values.shape[1], pre_keys.shape[1], pre_queries.shape[1]

        # Compute the values keys and queries
        values = self.values(pre_values)  # (N, value_len, embed_size)
        keys = self.keys(pre_keys)  # (N, key_len, embed_size)
        queries = self.queries(pre_queries)  # (N, query_len, embed_size)
        # WARNING: values (resp. keys and queries), are then used by different heads.
        # Each head uses a fraction of the values (resp. keys and queries) tensor.


        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, value_len, self.heads, self.head_dim)
        queries = self.queries(N, value_len, self.heads, self.head_dim)


        # Compute the similarity between query*keys
        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum("nqhd,nkhd->nhqk" , [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, quy_len, key_len)


        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))


        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for better stability
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)


        # We compute the attention
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.


        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """Reproduce the above figure.

        Tip: Dropout is always used after LayerNorm
        """
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

    def forward2(self, value, key, query, mask):
        """Reproduce the above figure.

        Tip: Dropout is always used after LayerNorm
        """
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm(attention + query))
        result = self.feed_forward(x)
        out = self.dropout(self.norm2(result + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: Tokenized tensor (N, seq_length) containing tokens_ids
        mask: Used for masking the padding inside the encoder.

        Create the position_embedding/word_embedding
        add the embeddings and forward it the all the layers.

        Tip: In order to create the position_embedding, you will need torch.arange and tensor.expand
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        position_embedding = self.position_embedding(positions)
        word_embedding = self.word_embedding(x)
        out = self.dropout(position_embedding + word_embedding)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

    def forward2(self, x, mask):
        """
        x: Tokenized tensor (N, seq_length) containing tokens_ids
        mask: Used for masking the padding inside the encoder.

        Create the position_embedding/word_embedding
        add the embeddings and forward it the all the layers.

        Tip: In order to create the position_embedding, you will need torch.arange and tensor.expand
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        position_embedding = self.position_embedding(positions)
        word_embedding = self.word_embedding(x)
        out = self.dropout(position_embedding + word_embedding)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        """DecoderBlock = masked multi-head attention + TransformerBlock"""
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

    def forward2(self, x, value, key, src_mask, trg_mask):
        """DecoderBlock = masked multi-head attention + TransformerBlock"""
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        output = self.transformer_block(value, key, query, src_mask)


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        """Same as Encoder"""
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

    def forward2(self, x, enc_out, src_mask, trg_mask):
        """Same as Encoder"""
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """src is a tensor containing sequences of tokens. Some sequences have been padded.

        The purpose of the src_mask is to mask those padded tokens.
        This mask is used both during training and inference time.
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)


    def make_trg_mask(self, trg):
        """trg is a tensor containing sequences of tokens which have been predicted.

        trg mask is used only during training.
        """
        # Bonus: Why do we use a lower triangular matrix?
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

class text_model:
    def __init__(self):
        self.model = Transformer(
            src_vocab_size=32,
            trg_vocab_size=32,
            src_pad_idx=0,
            trg_pad_idx=0,
            embed_size=512,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            max_length=100,
        )
        # Remplacez par votre chemin d'accès complet au fichier .pth sinon cela ne marchera pas...
        self.model.load_state_dict(torch.load("path to you inversion_model.pth"))
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()
        self.unique_words = []

    def extract_unique_words(self, prompt: str):
        all_words = set()
        for word in prompt.split():
            all_words.add(word)
        sorted_unique_words = sorted(all_words)
        self.unique_words = sorted_unique_words
    
    def tokenize(self, prompt: str):
        verse = {word: i for i, word in enumerate(self.unique_words)}
        return [verse[word] if word in verse else self.model.src_pad_idx for word in prompt.split()]

    def detokenize(self, indices):
        inverse = {i: word for i, word in enumerate(self.unique_words)}
        return ' '.join([inverse[index] for index in indices if index in inverse])

    def generate_response(self, prompt: str):
        self.extract_unique_words(prompt)

        src = self.tokenize(prompt)
        trg = [self.model.trg_pad_idx]
        src = torch.tensor(src).unsqueeze(0).to(self.model.device)
        trg = torch.tensor(trg).unsqueeze(0).to(self.model.device)
    
        with torch.no_grad():
            for _ in range(25):
                trg_mask = self.model.make_trg_mask(trg)
                out = self.model(src, trg)
                pred_token = out.argmax(2)[:, -1].item()
                trg = torch.cat([trg, torch.tensor([[pred_token]], device=self.model.device)], dim=1)
                if pred_token == self.model.trg_pad_idx:
                    break
        return self.detokenize(trg.squeeze(0).tolist())
    

app = FastAPI()

origins = ["*"]

Model = text_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "text_model API is running!"}

@app.post("/text_model")
async def text_model(input_data: dict):
    prompt = input_data["prompt"]
    return {"Type": "Text", "Response": Model.generate_response(prompt)}