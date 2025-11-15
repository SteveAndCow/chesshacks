import os

import torch
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.transformer import ChessTransformer
from stockfishdataset import StockfishDataset

def train_stockfish(model, test_dataloader, train_dataloader, save_path, dtype=torch.float, epochs=64):
    # Training setup
    file_path = f".\\{save_path}\\Config.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    criterion = MSELoss()
    model.to("cuda", dtype)

    torch.autograd.set_detect_anomaly(True)

    # Training loop
    step = 1
    for epoch in range(0, epochs):
        batch_steps = 0
        epoch_total_loss = 0

        batches = len(train_dataloader)

        for batch in tqdm(train_dataloader):
            bitboards, values = batch

            values = values.to("cuda", dtype)
            bitboards = bitboards.to("cuda", dtype)

            predicted_policy, predicted_values = model(bitboards)

            value_loss = criterion(predicted_values, values)

            loss = value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            epoch_total_loss += loss.item()

            if batch_steps % (batches // 10) == 0:
                first = f"[Step {batch_steps} / {batches}] Train: MSE Loss = {epoch_total_loss:.4f}"
                print(first)
                first = False

        evaluation_total_loss = evaluate_contrastive(model, test_dataloader)

        term = f"[Epoch {epoch}] Train: MSE Loss = {epoch_total_loss / batches:.4f}"
        term += f"Test: MSE Loss = {evaluation_total_loss / batches:.4f}"

        term += "\n"

        print(term)

        torch.save(model, f".\\{save_path}\\Epoch-{epoch}.pt")


def evaluate_contrastive(model, dataloader, dtype=torch.float):
    total_loss = 0
    criterion = MSELoss()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            bitboards, values = batch

            values = values.to("cuda", dtype)
            bitboards = bitboards.to("cuda", dtype)

            predicted_policy, predicted_values = model(bitboards)

            value_loss = criterion(predicted_values, values)

            total_loss += value_loss.item()

    return total_loss

if __name__ == "__main__":
    print("Loading...")
    test_bitboards = torch.load(".\\training\\data\\processed\\stockfish\\random_evals\\bitboards.pt", weights_only=False)
    test_evaluations = torch.load(".\\training\\data\\processed\\stockfish\\random_evals\\evaluations.pt", weights_only=False)
    test_set = StockfishDataset(test_bitboards, test_evaluations)

    train_bitboards = torch.load(".\\training\\data\\processed\\stockfish\\tactic_evals\\bitboards.pt", weights_only=False)
    train_evaluations = torch.load(".\\training\\data\\processed\\stockfish\\tactic_evals\\evaluations.pt", weights_only=False)
    train_set = StockfishDataset(train_bitboards, train_evaluations)

    train_dataloader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=64,
        shuffle=True,
    )

    save_path = ".\\data\\"

    model = ChessTransformer()
    model = torch.compile(model)
    print("Training...")
    train_stockfish(model, test_dataloader, train_dataloader, save_path=save_path, dtype=torch.float, epochs=64)