"""Train a character-level MiniGPT or sample a saved checkpoint."""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


class Attention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, width = x.shape
        head_width = width // self.heads
        q, k, v = [
            part.reshape(batch, time, self.heads, head_width).transpose(1, 2)
            for part in self.qkv(x).chunk(3, dim=-1)
        ]
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        joined = attended.transpose(1, 2).reshape(batch, time, width)
        return self.output_dropout(self.proj(joined))


class Block(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)
        self.attention = Attention(width, heads, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 4 * width), nn.ReLU(),
            nn.Linear(4 * width, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.feedforward(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, width, heads, layers, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(block_size, width)
        self.blocks = nn.Sequential(*[Block(width, heads, dropout) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, indices):
        positions = torch.arange(indices.shape[1], device=indices.device)
        hidden = self.token_embedding(indices) + self.position_embedding(positions)
        return self.output(self.final_norm(self.blocks(hidden)))


def get_batch(tokens, block_size, batch_size, device, generator=None):
    # Each y character is the character immediately following its x character.
    starts = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
    offsets = starts[:, None] + torch.arange(block_size)
    return tokens[offsets].to(device), tokens[offsets + 1].to(device)


def next_character_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())


@torch.no_grad()
def estimate_loss(model, splits, batch_size, device):
    model.eval()
    results = {}
    for name, tokens in splits.items():
        # Reuse the same random windows at every evaluation.
        rng = torch.Generator().manual_seed(42)
        losses = [
            next_character_loss(model, *get_batch(tokens, model.block_size, batch_size, device, rng))
            for _ in range(8)
        ]
        results[name] = torch.stack(losses).mean().item()
    model.train()
    return results


def train(args, device):
    corpus = args.input.read_text(encoding="utf-8").strip()
    # Hold out whole records from the end of the corpus.
    records = [entry.strip() for entry in corpus.split("\n\n") if entry.strip()]
    split = int(0.9 * len(records))
    train_text = "\n\n".join(records[:split]) + "\n\n"
    val_text = "\n\n".join(records[split:]) + "\n\n"
    block_size = 128
    if min(len(train_text), len(val_text)) <= block_size:
        raise SystemExit("Add more blank-line-separated records. Both splits need at least 129 characters.")

    chars = sorted(set(train_text + val_text))
    char_to_id = {char: index for index, char in enumerate(chars)}
    splits = {
        name: torch.tensor([char_to_id[char] for char in part], dtype=torch.long)
        for name, part in (("train", train_text), ("val", val_text))
    }
    config = dict(vocab_size=len(chars), block_size=block_size, dropout=0.2)
    config.update(dict(width=96, heads=4, layers=3) if args.small else dict(width=192, heads=6, layers=6))
    model = MiniGPT(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"Device {device} | {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    print(f"Train {len(splits['train'])} chars | Val {len(splits['val'])} chars | Vocab {len(chars)}", flush=True)

    for step in range(args.steps + 1):
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, splits, args.batch_size, device)
            print(f"Step {step:4d} | train {losses['train']:.4f} | val {losses['val']:.4f}", flush=True)
        if step == args.steps:
            break
        loss = next_character_loss(model, *get_batch(splits["train"], block_size, args.batch_size, device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({"weights": model.state_dict(), "chars": chars, "config": config}, args.checkpoint)
    print(f"Saved {args.checkpoint}", flush=True)


@torch.no_grad()
def ask(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    chars = checkpoint["chars"]
    char_to_id = {char: index for index, char in enumerate(chars)}
    model = MiniGPT(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval()

    prompt = f"Q: {args.question.strip()}\nA:"
    unknown = sorted(set(prompt) - set(char_to_id))
    if unknown:
        raise SystemExit(f"Characters outside the training vocabulary: {', '.join(map(repr, unknown))}")
    tokens = torch.tensor([[char_to_id[char] for char in prompt]], device=device)
    answer = []
    for _ in range(args.new_chars):
        logits = model(tokens[:, -model.block_size:])[0, -1]
        next_id = torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1)
        next_char = chars[next_id.item()]
        if next_char == "\n":
            break
        answer.append(next_char)
        tokens = torch.cat((tokens, next_id.view(1, 1)), dim=1)
    print("".join(answer).strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "mini_gpt.pt")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--input", type=Path, default=HERE / "input.txt")
    train_parser.add_argument("--steps", type=int, default=3000)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--eval-every", type=int, default=300)
    train_parser.add_argument("--small", action="store_true", help="Train a smaller model for CPU experiments.")
    ask_parser = commands.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--new-chars", type=int, default=160)
    ask_parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    if args.command == "train" and min(args.steps, args.batch_size, args.eval_every) < 1:
        parser.error("Training steps, batch size, and evaluation interval must be positive.")
    if args.command == "ask" and (args.new_chars < 1 or not 0 < args.temperature < float("inf")):
        parser.error("Generation length and temperature must be positive, and temperature must be finite.")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        torch.set_num_threads(min(4, torch.get_num_threads()))

    if args.command == "train":
        train(args, device)
    else:
        ask(args, device)


if __name__ == "__main__":
    main()
