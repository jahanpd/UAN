import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class train:
    def __init__(
        self,
        model,
        trainloader,
        outcome_names,
        valloaders=None,
        limit_steps=None,
        epochs=100,
        device="cuda",
        metric_fn=None,
        early_stopping=None,
        custom_fn=None,
        logdir='./logs/',
        save_dir='./checkpoint/',
        save_name="model.pt",
    ):
        """
            Custom training loop.
            model: Pytorch model
            trainloader: Pytorch dataloader
            valloaders: dict, pytorch dataloaders for validation datasets eg
                        {"validation":dataloader...}
            outcome_names: list of outcome names
            limit_steps: Int, option to end dataloader loop early at
                        defined step (useful for testing)
            epochs: Int, self explanatory
            device: str, one of "cpu" or "cuda"
            metric_fn: function, ran on validation datasets that takes
                        the model and validation dataloader output and
                        saves the metric to logdir, defined as
                    def metric_fn(model, dl, dl_name, writer, epoch, device):
                        x, y, z = dl_output
                        out = model(x, y)
                        save stuff with writer
                        return float
            early_stopping: dict, has two keys - patience and epoch. Patience
                            is how long to wait before stopping, and epoch is
                            when to start counting for early stopping. Uses the
                            float output of metric_fn if present, otherwise the
                            total loss from the model. eg
                            {"patience":10, "epoch":100}
            custom_fn: function, that takes the current epoch, step, and model
                        as input and allows the user to define changes to the
                        model at a specific point in training, eg
                        def custom_fn(step, epoch, model):
                            if step == 10:
                                model.anneal = False
            logdir: str, directory to save tensorboard training metrics
            save_dir: str, dir to save checkpoints and final model
            save_name: str, final name to save model
        """
        self.model = model.to(device)
        self.sample = model.sample
        self.trainloader = trainloader
        self.valloaders = valloaders
        self.outcome_names = outcome_names
        self.steps = limit_steps
        self.epochs = epochs
        self.device = device
        self.metric_fn = metric_fn
        self.custom_fn = custom_fn
        self.logdir = logdir
        self.save_dir = save_dir
        self.save_name = save_name

        if early_stopping is not None:
            early_stopping["best"] = 0
            early_stopping["recent"] = 0
            early_stopping["count"] = 0
            early_stopping["end"] = False
            self.early_stopping = early_stopping
        else:
            self.early_stopping = None
        self.best_state_dict = None

        if device == "cuda":
            print(torch.cuda.current_device())

    def train(self, mode="skip"):
        """
            mode: one of ["all", "mapping", "attention"]
        """
        logdir = self.logdir + datetime.now().strftime("%Y%m%d-%H%M%S")\
            + mode + "/"

        writer = SummaryWriter(
                log_dir=logdir)

        if mode == "mapping":
            self.model.train()
            self.model.alt = [0, 1]

            self.model.xmap.requires_grad_(True)
            self.model.x_attn.requires_grad = False
            self.model.xattn.requires_grad_(False)
            self.model.ffn.requires_grad_(False)
            self.model.attn.requires_grad_(False)
            self.model.y_attn.requires_grad = True

            try:
                self.model.decoder_uni.requires_grad_(True)
                self.model.decoder_full.requires_grad_(True)
                print("set decoder True")
            except Exception:
                self.model.encoder1.requires_grad_(True)
                self.model.encoder2.requires_grad_(True)
                self.model.flow_alpha_uni.requires_grad_(True)
                self.model.flow_beta_uni.requires_grad_(True)
                self.model.flow_alpha_full.requires_grad_(True)
                self.model.flow_beta_full.requires_grad_(True)
                print("set flows True")

        elif mode == "attention":
            self.model.train()
            self.model.alt = [1, 0]

            self.model.xmap.requires_grad_(False)
            self.model.x_attn.requires_grad = True
            self.model.xattn.requires_grad_(True)
            self.model.ffn.requires_grad_(True)
            self.model.attn.requires_grad_(True)
            self.model.y_attn.requires_grad = False

            try:
                self.model.decoder_uni.requires_grad_(True)
                self.model.decoder_full.requires_grad_(True)
                print("set decoder False")
            except Exception:
                self.model.encoder1.requires_grad_(False)
                self.model.encoder2.requires_grad_(False)
                self.model.flow_alpha_uni.requires_grad_(False)
                self.model.flow_beta_uni.requires_grad_(False)
                self.model.flow_alpha_full.requires_grad_(False)
                self.model.flow_beta_full.requires_grad_(False)
                print("set flows False")

        elif mode == "all":
            self.model.train()
            self.model.alt = [1, 1]

            self.model.xmap.requires_grad_(True)
            self.model.x_attn.requires_grad = True
            self.model.xattn.requires_grad_(True)
            self.model.ffn.requires_grad_(True)
            self.model.attn.requires_grad_(True)
            self.model.y_attn.requires_grad = True

            try:
                self.model.decoder_uni.requires_grad_(True)
                self.model.decoder_full.requires_grad_(True)
                print("set decoder True")
            except Exception:
                self.model.encoder1.requires_grad_(True)
                self.model.encoder2.requires_grad_(True)
                self.model.flow_alpha_uni.requires_grad_(True)
                self.model.flow_beta_uni.requires_grad_(True)
                self.model.flow_alpha_full.requires_grad_(True)
                self.model.flow_beta_full.requires_grad_(True)
                print("set flows True")

        epochbar = tqdm(total=self.epochs)
        pbar = tqdm(total=len(self.trainloader))
        for epoch in range(self.epochs):
            # self.model = self.model.to(self.device)
            # self.model.train()
            epochbar.set_postfix(
                ordered_dict={"Epoch": epoch, "Total": self.epochs})

            step = 0
            loss = self.model.loss()
            for key in loss:
                loss[key] = 0
            for d in self.trainloader:
                if self.custom_fn is not None:
                    self.custom_fn(step, epoch, self.model)
                for key in d:
                    d[key] = d[key].to(self.device)
                d["training"] = True
                _ = self.model(**d)
                self.model.step(step)
                record = self.model.loss()
                for key in record:
                    loss[key] += record[key]
                stats = {
                    's': self.model.sample
                }
                if self.early_stopping is not None:
                    stats['best'] = self.early_stopping['best']
                    stats['recent'] = self.early_stopping['recent']
                    stats['count'] = self.early_stopping['count']
                for key in loss:
                    stats[key] = loss[key] / (step + 1)

                pbar.update(1)
                pbar.set_postfix(ordered_dict=stats)
                step += 1
                if self.steps is not None:
                    if step > self.steps:
                        break

            pbar.reset()
            epochbar.update(1)
            for key in loss:
                writer.add_scalar(
                    str(key) + '/train/', float(loss[key]) / (step + 1), epoch)

            # d_m = self.model.d_m
            # mean_embed = self.model.x_attn[0, :, :d_m]  # (x, d_m)
            # std_embed = self.model.x_attn[0, :, d_m:]  # (x, d_m)
            # writer.add_embedding(mean_embed, global_step=epoch, tag="mean")
            # writer.add_embedding(std_embed, global_step=epoch, tag="std")

            if self.valloaders is not None:
                for key in self.valloaders:
                    metric = self.metric_fn(
                        model=self.model,
                        dl=self.valloaders[key],
                        dl_name=str(key),
                        outcome_names=self.outcome_names,
                        epoch=epoch,
                        writer=writer,
                        device=self.device
                        )
                    if (str(key) == "partial_validation") and (
                            self.early_stopping is not None) and (
                                epoch > self.early_stopping["epoch"]
                            ):
                        if self.early_stopping["best"] <= metric:
                            self.early_stopping["best"] = metric
                            self.early_stopping["count"] = 0
                            self.early_stopping["recent"] = metric
                            torch.save(
                                    self.model.state_dict(),
                                    self.save_dir
                                    + 'checkpoint'
                                    + '.pt')
                            self.best_state_dict = self.model.state_dict()
                        else:
                            self.early_stopping["count"] += 1
                            self.early_stopping["recent"] = metric
                        if (self.early_stopping["count"]
                                >= self.early_stopping["patience"]):
                            self.early_stopping["end"] = True
            elif (self.valloaders is None) and (
                    self.early_stopping is not None):
                l_ = 0
                for key in loss:
                    l_ -= loss[key]
                if self.early_stopping["best"] <= l_:
                    self.early_stopping["best"] = l_
                    self.early_stopping["count"] = 0
                    self.early_stopping["recent"] = l_
                    torch.save(
                                self.model.state_dict(),
                                self.save_dir
                                + 'checkpoint'
                                + '.pt')
                    self.best_state_dict = self.model.state_dict()
                else:
                    self.early_stopping["count"] += 1
                    self.early_stopping["recent"] = l_
                if (epoch > self.early_stopping["epoch"]) and\
                        (self.early_stopping["count"]
                            >= self.early_stopping["patience"]):
                    self.early_stopping["end"] = True

            if self.early_stopping is None:
                self.best_state_dict = self.model.state_dict()
                torch.save(
                                self.model.state_dict(),
                                self.save_dir
                                + 'checkpoint'
                                + '.pt')

            if self.early_stopping is not None:
                if self.early_stopping["end"]:
                    break

            if self.model.scheduler1 is not None:
                self.model.scheduler1.step()

        torch.save(
            self.best_state_dict,
            self.save_dir
            + self.save_name
            + '.pt')
