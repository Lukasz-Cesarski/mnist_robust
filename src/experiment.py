import torch

MODEL_LEARNING_RATE = 1e-2
ADVER_LEARNING_RATE = 1e-1
TRAINING_PHASE_NAME = 'train'
VALIDATION_PHASE_NAME = 'valid'
PHASE_LOG_PREFIX = {
    TRAINING_PHASE_NAME:   "TRAINING:   ",
    VALIDATION_PHASE_NAME: "VALIDATION: ",
}


class Experiment:
    def __init__(self, model, loss_fn, optimizer_class, device=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

    def train_model(self, train_loader, num_epochs, valid_loader=None, validate_every=1):
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        parameters_to_optim = self.model.parameters()
        optimizer = self.optimizer_class(parameters_to_optim, lr=MODEL_LEARNING_RATE)

        for epoch in range(num_epochs+1):
            train_loss_mean, train_acc_mean = self._make_step(train_loader, optimizer, TRAINING_PHASE_NAME)
            self.log_info(
                phase=TRAINING_PHASE_NAME,
                epoch=epoch,
                loss=train_loss_mean,
                acc=train_acc_mean)
            if valid_loader is not None and epoch % validate_every == 0:
                valid_loss_mean, valid_acc_mean = self._make_step(valid_loader, None, VALIDATION_PHASE_NAME)
                self.log_info(
                    phase=VALIDATION_PHASE_NAME,
                    epoch=epoch,
                    loss=valid_loss_mean,
                    acc=valid_acc_mean)

    def train_adversarial(self, x_shape: tuple, targets, num_epochs, info_every=1):
        # noinspection PyArgumentList
        x = torch.nn.Parameter(
            torch.randn(targets.shape[:1] + x_shape,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=True))
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        parameters_to_optim = [x]
        optimizer = self.optimizer_class(parameters_to_optim, lr=ADVER_LEARNING_RATE)
        adver_loader = [(x, targets)]
        for epoch in range(num_epochs+1):
            adver_loss, adver_acc = self._make_step(
                data_loader=adver_loader,
                optimizer=optimizer,
                phase=TRAINING_PHASE_NAME)
            if epoch % info_every == 0:
                self.log_info(
                    phase=TRAINING_PHASE_NAME,
                    epoch=epoch,
                    loss=adver_loss,
                    acc=adver_acc)

        return x.detach()

    def _make_step(self, data_loader, optimizer, phase):
        is_training = phase == TRAINING_PHASE_NAME

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        loss_list = []
        acc_list = []

        for data, targets in data_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            with torch.set_grad_enabled(is_training):
                logits = self.model(data)
                loss = self.loss_fn(logits, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, predictions = torch.max(logits, 1)
            acc = torch.eq(targets, predictions).type(torch.float).mean()
            loss_list.append(loss)
            acc_list.append(acc)

        loss_mean = (sum(loss_list) / len(loss_list)).item()
        acc_mean = (sum(acc_list) / len(acc_list)).item()
        return loss_mean, acc_mean

    def predict(self, x):
        logits = self.model(x)
        probs = torch.nn.Softmax(dim=1)(logits)
        return probs

    @staticmethod
    def log_info(phase, epoch, loss, acc):
        phase_prefix = PHASE_LOG_PREFIX[phase]
        print(f"{phase_prefix} epoch:{epoch:7d}, loss={loss:7.4f}, acc={acc:7.2f}")
