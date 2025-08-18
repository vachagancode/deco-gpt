import torch
import torch.nn as nn
import torch.nn.functional as F

from random import randint
from tqdm import tqdm

# hyperparameters 
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 6 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500
n_embd = 128
n_heads = 8
n_layers = 4
dropout = 0.15
# ------------------------------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars) + 1
_pad = len(chars) + 1
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] if i != _pad else "" for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
valid_data = data[n:]

def generate_random_additions():
    a = randint(0, 9)
    b = randint(0, 9)
    sum = a + b

    source = encode(f"{a}+{b}=") 
    target = encode(str(sum)[::-1])

    # Pad values to have equal lengths
    source += [_pad for _ in range(block_size - len(source))]
    target += [_pad for _ in range(block_size - len(target))]
    
    return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def generate_data(max_items=8):
    x = []
    y = []
    for _ in range(max_items):
        source, target = generate_random_additions()
        x.append(source)
        y.append(target)

    x = torch.stack(x)
    y = torch.stack(y)

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = generate_data(batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Block(nn.Module):
    """ Transformer block: communication and computation """

    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.LayerNorm(n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = n_embd // n_heads
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 # d_k = head size (in this case)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        scores = wei @ v

        scores = scores.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(scores)
        out = self.dropout(out)

        return out

class BigramLanguageModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size+1, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.mh_attenion = MultiHeadAttention(n_embd, n_heads)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size+1)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx) # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets, ignore_index=_pad)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for s in range(max_new_tokens):
            idx_cont = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cont)
            # print(f"Logits: {logits} | {logits.shape}" if s == 0 else "")
            # Focus only on the last timestep
            logits = logits[:, -1, :]
            # print(f"Logits last: {logits} | {logits.shape}" if s == 0 else "")
            probs = F.softmax(logits, dim=-1)
            # print(f"Probs last: {probs} | {probs.shape}" if s == 0 else "")
            # Sample from the distribution

            idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next == _pad:
                break
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        print(idx)
            
        return idx

def train(max_tokens=50):
    m = BigramLanguageModel(n_layers).to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for iter in tqdm(range(max_iters)):

        if iter % eval_interval == 0:
            losses = estimate_loss(m)
            print(f"Step: {iter} | Train Loss: {losses['train']:.4f} | Validation Loss: {losses['val']:.4f}")

        xb, yb = generate_data(batch_size)

        logits, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    context = torch.tensor(encode("5+5="), dtype=torch.long).unsqueeze(0)
    decoded_text = decode(m.generate(context, max_new_tokens=max_tokens)[0].tolist())
    print(decoded_text)

    return m, decoded_text

# if __name__ == "__main__":
#     # x, y = generate_data()
#     # print(x[0], decode(x[0].tolist()))
#     # print(chars)

#     # train()
#     train()