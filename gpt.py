import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn import functional as F
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def generate(input,n=20):
    text = []
    prev,past = tokenizer.encode(input),None
    text.extend(prev)
    prev = torch.tensor([prev])
    with torch.no_grad():
        for _ in range(n):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :]
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=5)
            #prev = prev[]
            text.append(prev[0].item())

    print(tokenizer.decode(text))
