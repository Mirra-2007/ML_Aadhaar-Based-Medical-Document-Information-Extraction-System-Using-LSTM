"""LSTM model definition for Phase 5 (PyTorch).

Architecture: Embedding -> LSTM -> per-token Linear classifier.

    token ids (B, L)
        | nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    embeddings (B, L, embed_dim)
        | nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
    LSTM hidden states (B, L, hidden_dim)
        | nn.Linear(hidden_dim, num_tags)
    logits (B, L, num_tags)   -> CrossEntropyLoss per token

LSTM gate semantics (exactly the equations nn.LSTM implements)
==============================================================
For each timestep t, with input x_t, previous hidden state h_{t-1} and
previous cell state c_{t-1}:

    i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)   input gate
    f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)   forget (retention) gate
    g_t = tanh   (W_g x_t + U_g h_{t-1} + b_g)   candidate cell content
    o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)   output gate
    c_t = f_t * c_{t-1} + i_t * g_t              cell state update
    h_t = o_t * tanh(c_t)                        exposed hidden state

* Forget gate f_t is an ELEMENTWISE MULTIPLICATIVE RETENTION control:
  each dimension of the previous cell state c_{t-1} is scaled by a value
  in (0, 1) - f_t near 1 carries that dimension of the previous state
  forward into c_t; f_t near 0 leaves the new cell state to be dominated
  by i_t * g_t for that dimension. It scales what is retained; it does
  not locate or "delete" particular pieces of text.
* Input gate i_t multiplicatively scales how much of the candidate
  g_t is written into c_t.
* Output gate o_t multiplicatively scales how much of tanh(c_t) is
  exposed as the hidden state h_t used by later timesteps/output.

`test_lstm.TestCaseGatesMatchManualMath` re-implements these equations
from the raw weight matrices and asserts numerical equality with
nn.LSTM output, proving the gates behave as specified.
"""

import torch
import torch.nn as nn


class LstmExtractor(nn.Module):
    """Token-level classifier over LSTM hidden states."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags,
                 pad_id=0, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.pad_id = pad_id
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_tags)

    def forward(self, token_ids):
        """token_ids: LongTensor (batch, length) -> logits (batch, length, tags).

        Tail padding is masked at the loss (targets use -100), and padded
        positions always come after real tokens, so no packed sequence is
        required: LSTM states over trailing padding never influence the
        masked positions' losses.
        """
        embeddings = self.embedding(token_ids)
        hidden_states, _ = self.lstm(embeddings)
        return self.classifier(hidden_states)

    def config_dict(self):
        """Serializable hyperparameters for model_config.json."""
        return {
            "architecture": "Embedding-LSTM-Linear",
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_tags": self.num_tags,
            "pad_id": self.pad_id,
        }
