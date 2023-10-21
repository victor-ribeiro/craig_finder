import torch.nn
import torch.nn.functional as F
import tqdm


class Classifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_0 = torch.nn.Sequential(torch.nn.Linear(12, 128), torch.nn.ReLU())
        self.block_1 = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU())
        # self.block_2 = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU())
        self.output = torch.nn.Sequential(torch.nn.Linear(128, 1), torch.nn.Sigmoid())

    def forward(self, x):
        x = self.block_0(x)
        x = self.block_1(x)
        # x = self.block_2(x)
        x = self.output(x)
        return x


class Epoch:
    def __init__(self) -> None:
        self._train_loss = []

    def run(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        loss_fn,
        optimizer,
    ) -> float:
        train_loss = 0.0
        for X_batch, y_batch in dataset:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()  # zera os gradientes para cada epoca
            loss.backward()  # calculo do gradiente
            optimizer.step()  # ajuste de pesos
            train_loss += loss.item()
            self._train_loss.append(train_loss)

        return sum(self._train_loss) / len(self._train_loss)


def train(
    model: torch.nn.Module,
    train_data: torch.utils.data.Dataset,
    val_data: torch.utils.data.Dataset,
    loss_fn,
    optimizer,
    n_epochs,
    debug=False,
):
    train_hist = []
    val_hist = []
    for _ in tqdm.trange(
        n_epochs,
        # desc=f"train_loss: {train_hist[-1] or 0} \t val_loss: {val_hist[-1] or 0}",
    ):
        model.train(True)
        train_loss = []

        # train loop
        for features, labels in train_data:
            features = torch.autograd.Variable(features).float()
            labels = labels.unsqueeze(
                1
            ).float()  # torch.autograd.Variable(labels).long()

            y_pred = model(features).float().round()
            # torch.argmax(y_pred, out=y_pred)
            loss = loss_fn(y_pred, labels)
            # loss.requires_grad = True

            optimizer.zero_grad()  # zera os gradientes para cada epoca
            loss.backward()  # calculo do gradiente
            optimizer.step()  # ajuste de pesos
            train_loss.append(loss.item())

        avg_loss = sum(train_loss) / len(train_loss)
        train_hist.append(avg_loss)
        # activate evaluation mode
        # desabilita o calculo do gradiente. validacao_modelo
        # model.eval()
        # with torch.no_grad():
        #     val_loss = 0.0
        #     for x_val, y_val in val_data:
        #         y_perd = model(x_val)
        #         val_loss += loss_fn(y_perd, y_val).item()
        #     val_hist.append(val_loss)
        # if debug:
        #     print(list(model.parameters()), end="\n\n")
        # else:
        #     f"train_loss: {train_hist[-1] or 0} \t val_loss: {val_hist[-1] or 0}"
    # model.train(False)
    print(y_pred)
    return {"val_loss": val_hist, "train_loss": train_hist}
