import torch


def get_train_device():
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")
    return device


if __name__ == "__main__":
    a = get_train_device()
    print(a)
