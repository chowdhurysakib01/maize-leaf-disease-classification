
"""
Maize Leaf Classification
Centralized Learning Rate Scheduler Factory

Supported:
1. Cosine Annealing
2. StepLR
3. ReduceLROnPlateau
4. OneCycleLR
5. No Scheduler
6. Linear Warmup

Stepping:
- OneCycleLR: per batch
- Other schedulers: per epoch

Compatible with TrainingConfig.
"""

import warnings

import torch

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LinearLR,
    SequentialLR,
)


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_SCHEDULERS = (
    "none",
    "cosine",
    "step",
    "plateau",
    "onecycle",
)

WARMUP_START_FACTOR = 0.1


# ============================================================
# SCHEDULER CONTROLLER
# ============================================================

class SchedulerController:
    """
    Standardized interface for batch-based and epoch-based
    learning rate scheduling.

    Training loop usage:

        After optimizer.step():
            controller.step_batch()

        After validation:
            controller.step_epoch(metric=validation_score)

    """

    def __init__(
        self,
        optimizer,
        scheduler=None,
        interval="epoch",
        warmup_scheduler=None,
        warmup_epochs=0,
        requires_metric=False,
        scheduler_name="none",
        total_steps=None,
    ):

        if interval not in ("epoch", "batch"):

            raise ValueError(
                "Interval must be epoch or batch."
            )

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.interval = interval

        self.warmup_scheduler = warmup_scheduler

        self.warmup_epochs = int(warmup_epochs)

        self.requires_metric = requires_metric

        self.scheduler_name = scheduler_name

        self.total_steps = total_steps

        self.completed_epochs = 0

        self.completed_batches = 0

    # --------------------------------------------------------
    # Batch stepping
    # --------------------------------------------------------

    def step_batch(self):

        if self.interval != "batch":

            return

        if self.scheduler is None:

            return

        self.scheduler.step()

        self.completed_batches += 1

    # --------------------------------------------------------
    # Epoch stepping
    # --------------------------------------------------------

    def step_epoch(self, metric=None):

        if self.interval != "epoch":

            return

        # Linear warmup phase.

        if (
            self.warmup_scheduler is not None
            and self.completed_epochs < self.warmup_epochs
        ):

            self.warmup_scheduler.step()

            self.completed_epochs += 1

            return

        # Main scheduler phase.

        if self.scheduler is not None:

            if self.requires_metric:

                if metric is None:

                    raise ValueError(
                        f"{self.scheduler_name} requires "
                        "a validation metric."
                    )

                if not torch.isfinite(
                    torch.tensor(float(metric))
                ):

                    raise ValueError(
                        "Scheduler metric must be finite."
                    )

                self.scheduler.step(float(metric))

            else:

                self.scheduler.step()

        self.completed_epochs += 1

    # --------------------------------------------------------
    # Current learning rates
    # --------------------------------------------------------

    def get_last_lr(self):

        return [

            float(group["lr"])

            for group in self.optimizer.param_groups

        ]

    # --------------------------------------------------------
    # Scheduler state
    # --------------------------------------------------------

    def state_dict(self):

        return {

            "scheduler_name": self.scheduler_name,

            "interval": self.interval,

            "completed_epochs": self.completed_epochs,

            "completed_batches": self.completed_batches,

            "warmup_epochs": self.warmup_epochs,

            "scheduler_state": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),

            "warmup_state": (
                self.warmup_scheduler.state_dict()
                if self.warmup_scheduler is not None
                else None
            ),

        }

    def load_state_dict(self, state):

        if state["scheduler_name"] != self.scheduler_name:

            raise ValueError(
                "Checkpoint scheduler does not match current scheduler."
            )

        if self.scheduler is not None:

            if state["scheduler_state"] is not None:

                self.scheduler.load_state_dict(
                    state["scheduler_state"]
                )

        if self.warmup_scheduler is not None:

            if state["warmup_state"] is not None:

                self.warmup_scheduler.load_state_dict(
                    state["warmup_state"]
                )

        self.completed_epochs = state["completed_epochs"]

        self.completed_batches = state["completed_batches"]


# ============================================================
# MONITOR MODE
# ============================================================

def get_monitor_mode(config):

    if config.monitor_mode != "auto":

        return config.monitor_mode

    if config.monitor == "val_loss":

        return "min"

    return "max"


# ============================================================
# WARMUP FACTORY
# ============================================================

def build_warmup_scheduler(
    optimizer,
    warmup_epochs,
    start_factor=WARMUP_START_FACTOR
):
    """
    Gradually increase learning rate from 10% to 100%
    of the configured initial learning rate.
    """

    if warmup_epochs <= 0:

        return None

    return LinearLR(

        optimizer,

        start_factor=start_factor,

        end_factor=1.0,

        total_iters=warmup_epochs,

    )


# ============================================================
# MAIN SCHEDULER FACTORY
# ============================================================

def build_scheduler(
    optimizer,
    config,
    steps_per_epoch=None
):
    """
    Build a scheduler using TrainingConfig.

    Args:
        optimizer:
            PyTorch optimizer.

        config:
            TrainingConfig instance.

        steps_per_epoch:
            Number of training batches per epoch.
            Required for OneCycleLR.

    Returns:
        SchedulerController
    """

    scheduler_name = config.scheduler.lower()

    if scheduler_name not in SUPPORTED_SCHEDULERS:

        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}"
        )

    if config.epochs <= 0:

        raise ValueError(
            "Total epochs must be positive."
        )

    if config.learning_rate <= 0:

        raise ValueError(
            "Learning rate must be positive."
        )

    if config.min_lr < 0:

        raise ValueError(
            "Minimum learning rate cannot be negative."
        )

    if config.min_lr > config.learning_rate:

        raise ValueError(
            "Minimum learning rate cannot exceed initial LR."
        )

    warmup_epochs = config.warmup_epochs

    if warmup_epochs < 0 or warmup_epochs >= config.epochs:

        raise ValueError(
            "Warmup epochs must be between 0 and epochs - 1."
        )

    if scheduler_name == "onecycle":

        if steps_per_epoch is None or steps_per_epoch <= 0:

            raise ValueError(
                "OneCycleLR requires a positive steps_per_epoch."
            )

        if warmup_epochs > 0:

            warnings.warn(
                "OneCycleLR uses its own internal warmup. "
                "The separate warmup_epochs setting is ignored.",
                RuntimeWarning
            )

    # --------------------------------------------------------
    # No scheduler
    # --------------------------------------------------------

    if scheduler_name == "none":

        warmup = build_warmup_scheduler(
            optimizer,
            warmup_epochs
        )

        return SchedulerController(

            optimizer=optimizer,

            scheduler=None,

            interval="epoch",

            warmup_scheduler=warmup,

            warmup_epochs=warmup_epochs,

            scheduler_name="none",

        )

    # --------------------------------------------------------
    # OneCycleLR
    # --------------------------------------------------------

    if scheduler_name == "onecycle":

        total_steps = (
            config.epochs * steps_per_epoch
        )

        scheduler = OneCycleLR(

            optimizer,

            max_lr=config.learning_rate,

            total_steps=total_steps,

            pct_start=0.3,

            anneal_strategy="cos",

            div_factor=25.0,

            final_div_factor=10000.0,

        )

        return SchedulerController(

            optimizer=optimizer,

            scheduler=scheduler,

            interval="batch",

            warmup_epochs=0,

            scheduler_name="onecycle",

            total_steps=total_steps,

        )

    # --------------------------------------------------------
    # Warmup scheduler
    # --------------------------------------------------------

    warmup = build_warmup_scheduler(
        optimizer,
        warmup_epochs
    )

    # --------------------------------------------------------
    # Cosine Annealing
    # --------------------------------------------------------

    if scheduler_name == "cosine":

        main_epochs = max(
            1,
            config.epochs - warmup_epochs
        )

        scheduler = CosineAnnealingLR(

            optimizer,

            T_max=main_epochs,

            eta_min=config.min_lr,

        )

    # --------------------------------------------------------
    # StepLR
    # --------------------------------------------------------

    elif scheduler_name == "step":

        scheduler = StepLR(

            optimizer,

            step_size=config.scheduler_step_size,

            gamma=config.scheduler_gamma,

        )

    # --------------------------------------------------------
    # ReduceLROnPlateau
    # --------------------------------------------------------

    elif scheduler_name == "plateau":

        scheduler = ReduceLROnPlateau(

            optimizer,

            mode=get_monitor_mode(config),

            factor=config.scheduler_factor,

            patience=config.scheduler_patience,

            threshold=config.min_delta,

            threshold_mode="abs",

            min_lr=config.min_lr,

        )

        # Plateau requires a validation metric, so
        # warmup is controlled separately.

        return SchedulerController(

            optimizer=optimizer,

            scheduler=scheduler,

            interval="epoch",

            warmup_scheduler=warmup,

            warmup_epochs=warmup_epochs,

            requires_metric=True,

            scheduler_name="plateau",

        )

    else:

        raise RuntimeError(
            "Unexpected scheduler factory state."
        )

    # --------------------------------------------------------
    # Combine warmup and main scheduler
    # --------------------------------------------------------

    if warmup is not None:

        scheduler = SequentialLR(

            optimizer,

            schedulers=[
                warmup,
                scheduler
            ],

            milestones=[
                warmup_epochs
            ],

        )

    return SchedulerController(

        optimizer=optimizer,

        scheduler=scheduler,

        interval="epoch",

        warmup_epochs=warmup_epochs,

        scheduler_name=scheduler_name,

    )


# ============================================================
# SCHEDULER SUMMARY
# ============================================================

def get_scheduler_summary(controller, config):

    return {

        "scheduler": controller.scheduler_name,

        "interval": controller.interval,

        "initial_learning_rate": config.learning_rate,

        "minimum_learning_rate": config.min_lr,

        "total_epochs": config.epochs,

        "warmup_epochs": controller.warmup_epochs,

        "total_steps": controller.total_steps,

        "monitor": (
            config.monitor
            if controller.requires_metric
            else None
        ),

        "monitor_mode": (
            get_monitor_mode(config)
            if controller.requires_metric
            else None
        ),

        "current_learning_rates": controller.get_last_lr(),

    }


# ============================================================
# PRINT SCHEDULER REPORT
# ============================================================

def print_scheduler_report(controller, config):

    summary = get_scheduler_summary(
        controller,
        config
    )

    print("\n" + "=" * 70)

    print("LEARNING RATE SCHEDULER REPORT")

    print("=" * 70)

    for key, value in summary.items():

        print(f"{key:<35}: {value}")

    print("=" * 70)


# ============================================================
# EXECUTION TEST
# ============================================================

def main():

    from configs.training_config import TrainingConfig

    print("=" * 70)

    print("MAIZE LEAF - SCHEDULER FACTORY TEST")

    print("=" * 70)

    # --------------------------------------------------------
    # Test standard epoch schedulers
    # --------------------------------------------------------

    scheduler_names = [
        "cosine",
        "step",
        "plateau",
        "none",
    ]

    for name in scheduler_names:

        print(f"\nTesting {name.upper()}...")

        model = torch.nn.Linear(16, NUM_CLASSES_FOR_TEST)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001
        )

        config = TrainingConfig(

            model="custom_cnn",

            epochs=5,

            learning_rate=0.001,

            scheduler=name,

            warmup_epochs=0,

        )

        controller = build_scheduler(
            optimizer,
            config
        )

        initial_lr = controller.get_last_lr()[0]

        for epoch in range(2):

            optimizer.zero_grad()

            output = model(torch.randn(8, 16))

            loss = output.square().mean()

            loss.backward()

            optimizer.step()

            if name == "plateau":

                controller.step_epoch(
                    metric=0.5 - epoch * 0.01
                )

            else:

                controller.step_epoch()

        assert controller.completed_epochs == 2

        assert controller.get_last_lr()[0] > 0

        if name == "none":

            assert controller.get_last_lr()[0] == initial_lr

        print(f"[PASS] {name.upper()} epoch scheduling")

    # --------------------------------------------------------
    # Test warmup + cosine
    # --------------------------------------------------------

    print("\nTesting warmup + cosine...")

    model = torch.nn.Linear(16, NUM_CLASSES_FOR_TEST)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    config = TrainingConfig(

        model="custom_cnn",

        epochs=6,

        learning_rate=0.001,

        scheduler="cosine",

        warmup_epochs=2,

    )

    controller = build_scheduler(
        optimizer,
        config
    )

    initial_warmup_lr = controller.get_last_lr()[0]

    assert initial_warmup_lr < config.learning_rate

    for _ in range(2):

        optimizer.step()

        controller.step_epoch()

    warmup_complete_lr = controller.get_last_lr()[0]

    assert abs(
        warmup_complete_lr - config.learning_rate
    ) < 1e-8

    print("[PASS] Linear warmup reaches initial learning rate")

    # --------------------------------------------------------
    # Test OneCycleLR
    # --------------------------------------------------------

    print("\nTesting OneCycleLR batch scheduling...")

    model = torch.nn.Linear(16, NUM_CLASSES_FOR_TEST)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    config = TrainingConfig(

        model="custom_cnn",

        epochs=3,

        learning_rate=0.001,

        scheduler="onecycle",

    )

    steps_per_epoch = 4

    controller = build_scheduler(
        optimizer,
        config,
        steps_per_epoch=steps_per_epoch
    )

    assert controller.total_steps == 12

    learning_rates = []

    for _ in range(controller.total_steps):

        optimizer.zero_grad()

        output = model(torch.randn(8, 16))

        loss = output.square().mean()

        loss.backward()

        optimizer.step()

        controller.step_batch()

        learning_rates.append(
            controller.get_last_lr()[0]
        )

    assert controller.completed_batches == 12

    assert len(set(learning_rates)) > 1

    print("[PASS] OneCycleLR batch scheduling")

    # --------------------------------------------------------
    # Test OneCycleLR required steps
    # --------------------------------------------------------

    print("\nTesting OneCycleLR validation...")

    try:

        build_scheduler(
            optimizer,
            config,
            steps_per_epoch=None
        )

        raise AssertionError(
            "OneCycleLR accepted missing steps_per_epoch."
        )

    except ValueError:

        print("[PASS] OneCycleLR steps-per-epoch protection")

    # --------------------------------------------------------
    # Test Plateau metric requirement
    # --------------------------------------------------------

    print("\nTesting plateau metric requirement...")

    model = torch.nn.Linear(16, NUM_CLASSES_FOR_TEST)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    config = TrainingConfig(
        model="custom_cnn",
        scheduler="plateau",
        epochs=5
    )

    controller = build_scheduler(
        optimizer,
        config
    )

    try:

        controller.step_epoch()

        raise AssertionError(
            "Plateau accepted a missing validation metric."
        )

    except ValueError:

        print("[PASS] Plateau metric protection")

    # --------------------------------------------------------
    # Test checkpoint state
    # --------------------------------------------------------

    print("\nTesting scheduler state restoration...")

    state = controller.state_dict()

    controller.load_state_dict(state)

    print("[PASS] Scheduler state save/load")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\nSCHEDULER FACTORY TEST: PASSED")

    print("=" * 70)


NUM_CLASSES_FOR_TEST = 7


if __name__ == "__main__":

    main()
