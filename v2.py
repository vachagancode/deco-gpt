import torch
import torch.nn as nn
import torch.nn.functional as F

from random import randint
from tqdm import tqdm

# hyperparameters  - main
batch_size = 256 # how many independent sequences will we process in parallel?
block_size = 14 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500
n_embd = 256
n_heads = 8
n_layers = 8
dropout = 0.2
# ------------------------------------
# hyperparameters  - test
# batch_size = 1 # how many independent sequences will we process in parallel?
# block_size = 18 # what is the maximum context length for predictions?
# max_iters = 150
# eval_interval = 50
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 500
# n_embd = 64
# n_heads = 4
# n_layers = 4
# dropout = 0.2
# ------------------------------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

_pad = stoi["_"]
_end = stoi[">"]

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l if i != -1])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))

operations = ['+', '-']

def do_proper_operation(a, b, operation):
    match operation:
        case '+':
            return a + b
        case '-':
            return a - b
        case '*':
            return a * b
        case '/':
            return a / b if b != 0 else 0  # Avoid division by zero
    return 0

def generate_random_calculations(multi_operator=False):

    if multi_operator:
        # Generate a random sequence with multiple operations
        n_nums = randint(2, 4)
        sequence = ""
        chain_of_thoughts = []

        final_result = 0

        for i in range(n_nums):
            operation = operations[randint(0, len(operations) - 1)]

            a = randint(0, 10)
            b = randint(0 if operation != "/" else 1, 10)  # Avoid division by zero on the last number
            sequence += f"{a}{operation}{b}" if i == 0 else f"{operation}{b}"
        
        seq = []

        if "/" in sequence:
            seq += sequence.split("/")
        elif "*" in sequence:
            seq += sequence.split("*")
        elif "-" in sequence:
            if "+" in sequence:
                add_idx = sequence.index("+")
                sub_idx = sequence.index("-")

                if add_idx < sub_idx:
                    seq = sequence.split("-")
                    prev = []
                    for i, p in enumerate(seq):
                        if "+" in p:
                            result, cof = only_addition(p)
                            prev.append(result)
                            chain_of_thoughts += cof
                        else:
                            prev.append(p)
                    
                    prev_sub = prev[0]
                    for r in prev[1:]:
                        chain_of_thoughts.append(f"{prev_sub}-{r}={prev_sub-int(r)} then")
                        prev_sub -= int(r)
                        if r == prev[-1]:
                            chain_of_thoughts.append(f" the result is: {prev_sub}")
                else:
                    seq = sequence.split("+")
                    prev = []
                    for i, p in enumerate(seq):
                        if "-" in p:
                            result, cof = only_subtraction(p)
                            prev.append(result)
                            chain_of_thoughts += cof
                        else:
                            prev.append(p)
                    
                    prev_sub = prev[0]
                    for r in prev[1:]:
                        chain_of_thoughts.append(f"{prev_sub}+{r}={prev_sub+int(r)} then")
                        prev_sub += int(r)
                        if r == prev[-1]:
                            chain_of_thoughts.append(f" the result is: {prev_sub}")
            else: # WORKS FINE
                seq = sequence.split('-')
                result = 0
                prev = seq[0]
                for i in range(1, len(seq)):
                    new_prev = int(prev) - int(seq[i])
                    chain_of_thoughts.append(f"{prev}-{seq[i]}={new_prev} then")
                    prev = new_prev
                result = prev
                chain_of_thoughts.append(f" the result is: {result}")
        elif "+" in sequence:
            if "-" not in sequence:
                result, cof = only_addition(sequence, inter=False)
                chain_of_thoughts += cof


        # elif "+" in sequence

        print(sequence)
        print(chain_of_thoughts)


        # print(sequence)
        # print(chain_of_thoughts)
        # print(final_result)

    else:
        # Get random operations
        operation = operations[randint(0, len(operations) - 1)]

        a = randint(0, 100)
        b = randint(0 if operation != '/' else 1, 100)
        result = do_proper_operation(a, b, operation)

        sequence = encode(f"{a}{operation}{b}={str(result)[::-1] if operation != '/' else str(result)[:4][::-1]}")

        # Pad source and target
        sequence += [_pad for _ in range(block_size - len(sequence))]
        sequence[-1] = _end

        source = sequence[:-1]
        target = sequence[1:]
        eq_idx = target.index(*encode("="))
        
        source, target = torch.tensor(source, dtype=torch.long, device=device), torch.tensor(target, dtype=torch.long, device=device)

        target[:eq_idx] = -1

        return source, target, sequence

def only_subtraction(sequence, inter=True):
    cof = []
    seq = sequence.split('-')
    result = 0
    prev = seq[0]
    for i in range(1, len(seq)):
        new_prev = int(prev) - int(seq[i])
        cof.append(f"{prev}-{seq[i]}={new_prev} then")
        prev = new_prev
    result = prev
    if inter == False:
        cof.append(f" the result is: {result}")

    return result, cof

def only_addition(sequence, inter=True):
    cof = []
    seq = sequence.split('+')
    result = 0
    prev = seq[0]
    for i in range(1, len(seq)):
        new_prev = int(prev) + int(seq[i])
        cof.append(f"{prev}+{seq[i]}={new_prev} then")
        prev = new_prev
    result = prev
    if inter == False:
        cof.append(f" the result is: {result}")

    return result, cof

def generate_data(max_items=8):
    x = []
    y = []
    for _ in range(max_items):
        source, target = generate_random_calculations()
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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.mh_attenion = MultiHeadAttention(n_embd, n_heads)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

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

            loss = F.cross_entropy(logits, targets, ignore_index=-1)

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
            if idx_next.item() == _pad or idx_next.item() == _end:
                break
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx

def calculate(model, device, context="77+55=", max_tokens=50):
    context = torch.tensor(encode(context), dtype=torch.long).unsqueeze(0).to(device)
    out = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    eq_idx = len(out) - out.index(encode("=")[0])
    reversed_answer = out[-eq_idx+1:][::-1]
    out[-len(reversed_answer):] = reversed_answer

    return decode(out)

def train(max_tokens=50):
    m = BigramLanguageModel(n_layers).to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.65, patience=1)

    for iter in tqdm(range(max_iters)):

        if iter % eval_interval == 0:
            losses = estimate_loss(m)
            print(f"Step: {iter} | Train Loss: {losses['train']:.4f} | Validation Loss: {losses['val']:.4f}")

            scheduler.step(losses["train"])

        xb, yb = generate_data(batch_size)

        logits, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    context = torch.tensor(encode("5*7="), dtype=torch.long).unsqueeze(0).to(device)
    decoded_text = decode(m.generate(context, max_new_tokens=max_tokens)[0].tolist())
    print(decoded_text)

    return m, decoded_text

# if __name__ == "__main__":
#     model = BigramLanguageModel(n_layers).to(device)
#     model.load_state_dict(torch.load('./models/calculator.pth', map_location=device))

#     context = "74/49="
#     decoded_text = calculate(model, device, context=context, max_tokens=50)
#     print(decoded_text)

if __name__ == "__main__":
    generate_random_calculations(multi_operator=True)