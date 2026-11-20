
"""
Maize Leaf Classification
Early Stopping Engine

Features:
    - Monitor validation loss, accuracy or Macro-F1.
    - Automatic min/max mode detection.
    - Configurable patience.
    - Configurable minimum improvement (min_delta).
    - Early stopping enable/disable.
    - Best epoch tracking.
    - Non-finite metric protection.
    - State dictionary saving.
    - Training checkpoint restoration.
    - Reproducible resume support.

Usage:

    early_stopping = EarlyStopping(
        patience=15,
        min_delta=0.0001,
        monitor="val_macro_f1",
        mode="auto",
        enabled=True
    )

    should_stop = early_stopping.step(
        metric_value=0.85,
        epoch=1
    )

    if early_stopping.improved:
        # Save best checkpoint.
        pass

    if should_stop:
        # Stop training.
        pass
"""

import math

from pathlib import Path


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_MONITORS = {

    "val_loss",

    "val_accuracy",

    "val_macro_f1",

}

SUPPORTED_MODES = {

    "auto",

    "min",

    "max",

}


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:

    def __init__(
        self,
        patience=15,
        min_delta=0.0001,
        monitor="val_macro_f1",
        mode="auto",
        enabled=True,
        verbose=True
    ):

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if not isinstance(patience, int) or patience < 1:

            raise ValueError(
                "Patience must be an integer greater than or equal to 1."
            )

        if min_delta < 0:

            raise ValueError(
                "min_delta cannot be negative."
            )

        if monitor not in SUPPORTED_MONITORS:

            raise ValueError(
                f"Unsupported monitor: {monitor}. "
                f"Choose from {sorted(SUPPORTED_MONITORS)}."
            )

        if mode not in SUPPORTED_MODES:

            raise ValueError(
                f"Unsupported mode: {mode}. "
                f"Choose from {sorted(SUPPORTED_MODES)}."
            )

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        self.patience = patience

        self.min_delta = float(min_delta)

        self.monitor = monitor

        self.mode = self._resolve_mode(mode)

        self.enabled = bool(enabled)

        self.verbose = bool(verbose)

        # ----------------------------------------------------
        # Dynamic state
        # ----------------------------------------------------

        self.best_value = None

        self.best_epoch = None

        self.bad_epochs = 0

        self.total_epochs = 0

        self.last_epoch = 0

        self.last_value = None

        self.improved = False

        self.should_stop = False

        self.stop_epoch = None

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.improvement_count = 0

    # ========================================================
    # RESOLVE AUTOMATIC MODE
    # ========================================================

    def _resolve_mode(self, mode):

        if mode == "auto":

            if self.monitor == "val_loss":

                return "min"

            return "max"

        return mode

    # ========================================================
    # COMPARISON
    # ========================================================

    def _is_improvement(self, current_value):

        if self.best_value is None:

            return True

        if self.mode == "min":

            return (

                current_value

                <

                self.best_value - self.min_delta

            )

        return (

            current_value

            >

            self.best_value + self.min_delta

        )

    # ========================================================
    # MAIN UPDATE FUNCTION
    # ========================================================

    def step(
        self,
        metric_value,
        epoch=None
    ):
        """
        Process one completed validation epoch.

        Returns:
            True  -> Stop training.
            False -> Continue training.

        Check self.improved to determine whether
        the current epoch produced a new best result.
        """

        # ----------------------------------------------------
        # Validate metric
        # ----------------------------------------------------

        if hasattr(metric_value, "item"):

            metric_value = metric_value.item()

        try:

            metric_value = float(metric_value)

        except (TypeError, ValueError):

            raise ValueError(
                "Metric value must be a valid number."
            )

        if not math.isfinite(metric_value):

            raise ValueError(
                "Metric value cannot be NaN or infinity."
            )

        # ----------------------------------------------------
        # Resolve epoch
        # ----------------------------------------------------

        if epoch is None:

            epoch = self.last_epoch + 1

        if not isinstance(epoch, int) or epoch < 1:

            raise ValueError(
                "Epoch must be a positive integer."
            )

        if epoch <= self.last_epoch:

            raise ValueError(
                "Epoch numbers must increase monotonically."
            )

        self.last_epoch = epoch

        self.total_epochs += 1

        self.last_value = metric_value

        # Reset improvement flag.

        self.improved = False

        # ----------------------------------------------------
        # Check improvement
        # ----------------------------------------------------

        if self._is_improvement(metric_value):

            self.best_value = metric_value

            self.best_epoch = epoch

            self.bad_epochs = 0

            self.improved = True

            self.improvement_count += 1

            if self.verbose:

                print(

                    f"\n[EARLY STOPPING] "

                    f"New best {self.monitor}: "

                    f"{metric_value:.6f} "

                    f"at epoch {epoch}"

                )

        else:

            self.bad_epochs += 1

            if self.verbose:

                print(

                    f"[EARLY STOPPING] "

                    f"No improvement: "

                    f"{self.bad_epochs}/{self.patience} | "

                    f"Best: {self.best_value:.6f}"

                )

        # ----------------------------------------------------
        # Check stopping condition
        # ----------------------------------------------------

        if self.enabled:

            if self.bad_epochs >= self.patience:

                self.should_stop = True

                self.stop_epoch = epoch

                if self.verbose:

                    print(

                        f"\n[EARLY STOPPING] "

                        f"Training stopped at epoch {epoch}."

                    )

                    print(

                        f"Best epoch: {self.best_epoch}"

                    )

                    print(

                        f"Best {self.monitor}: "

                        f"{self.best_value:.6f}"

                    )

                return True

        return False

    # ========================================================
    # STATE DICTIONARY
    # ========================================================

    def state_dict(self):

        """
        Return complete state for checkpoint saving.
        """

        return {

            "config": {

                "patience": self.patience,

                "min_delta": self.min_delta,

                "monitor": self.monitor,

                "mode": self.mode,

                "enabled": self.enabled,

            },

            "state": {

                "best_value": self.best_value,

                "best_epoch": self.best_epoch,

                "bad_epochs": self.bad_epochs,

                "total_epochs": self.total_epochs,

                "last_epoch": self.last_epoch,

                "last_value": self.last_value,

                "improved": self.improved,

                "should_stop": self.should_stop,

                "stop_epoch": self.stop_epoch,

                "improvement_count": self.improvement_count,

            },

        }

    # ========================================================
    # LOAD STATE
    # ========================================================

    def load_state_dict(
        self,
        state_dict,
        strict_config=True
    ):
        """
        Restore early stopping state after interruption.

        Allows training to resume without resetting patience.
        """

        if not isinstance(state_dict, dict):

            raise TypeError(
                "Early stopping state must be a dictionary."
            )

        if "state" not in state_dict:

            raise ValueError(
                "Invalid early stopping state dictionary."
            )

        saved_config = state_dict.get(
            "config",
            {}
        )

        saved_state = state_dict["state"]

        # ----------------------------------------------------
        # Configuration validation
        # ----------------------------------------------------

        if strict_config:

            for key in (
                "monitor",
                "mode"
            ):

                if key in saved_config:

                    if getattr(self, key) != saved_config[key]:

                        raise ValueError(

                            f"Early stopping configuration "
                            f"mismatch: {key}."

                        )

        # ----------------------------------------------------
        # Restore state
        # ----------------------------------------------------

        required_keys = [

            "best_value",

            "best_epoch",

            "bad_epochs",

            "total_epochs",

            "last_epoch",

            "last_value",

            "improved",

            "should_stop",

            "stop_epoch",

            "improvement_count",

        ]

        for key in required_keys:

            if key not in saved_state:

                raise ValueError(
                    f"Missing state field: {key}"
                )

        self.best_value = saved_state["best_value"]

        self.best_epoch = saved_state["best_epoch"]

        self.bad_epochs = int(
            saved_state["bad_epochs"]
        )

        self.total_epochs = int(
            saved_state["total_epochs"]
        )

        self.last_epoch = int(
            saved_state["last_epoch"]
        )

        self.last_value = saved_state["last_value"]

        self.improved = bool(
            saved_state["improved"]
        )

        self.should_stop = bool(
            saved_state["should_stop"]
        )

        self.stop_epoch = saved_state["stop_epoch"]

        self.improvement_count = int(
            saved_state["improvement_count"]
        )

        return self

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.best_value = None

        self.best_epoch = None

        self.bad_epochs = 0

        self.total_epochs = 0

        self.last_epoch = 0

        self.last_value = None

        self.improved = False

        self.should_stop = False

        self.stop_epoch = None

        self.improvement_count = 0

        return self

    # ========================================================
    # SUMMARY
    # ========================================================

    def get_summary(self):

        return {

            "enabled": self.enabled,

            "monitor": self.monitor,

            "mode": self.mode,

            "patience": self.patience,

            "min_delta": self.min_delta,

            "best_value": self.best_value,

            "best_epoch": self.best_epoch,

            "bad_epochs": self.bad_epochs,

            "total_epochs": self.total_epochs,

            "improvement_count": self.improvement_count,

            "should_stop": self.should_stop,

            "stop_epoch": self.stop_epoch,

        }


# ============================================================
# EARLY STOPPING TEST
# ============================================================

def main():

    print("=" * 70)

    print("MAIZE LEAF - EARLY STOPPING ENGINE TEST")

    print("=" * 70)

    # --------------------------------------------------------
    # TEST 1: Validation loss minimization
    # --------------------------------------------------------

    print("\nTEST 1: VALIDATION LOSS MINIMIZATION")

    early = EarlyStopping(

        patience=3,

        min_delta=0.001,

        monitor="val_loss",

        mode="auto",

        verbose=False

    )

    assert early.mode == "min"

    sequence = [

        0.80,

        0.60,

        0.50,

        0.501,

        0.502,

        0.503,

    ]

    stop_results = []

    for epoch, value in enumerate(sequence, 1):

        stopped = early.step(
            value,
            epoch
        )

        stop_results.append(stopped)

    assert early.best_value == 0.50

    assert early.best_epoch == 3

    assert early.bad_epochs == 3

    assert stop_results[-1] is True

    print("[PASS] Validation loss minimization")

    print("[PASS] Patience counting")

    print("[PASS] Early stopping trigger")

    # --------------------------------------------------------
    # TEST 2: Macro-F1 maximization
    # --------------------------------------------------------

    print("\nTEST 2: MACRO-F1 MAXIMIZATION")

    early = EarlyStopping(

        patience=3,

        min_delta=0.001,

        monitor="val_macro_f1",

        mode="auto",

        verbose=False

    )

    assert early.mode == "max"

    values = [

        0.70,

        0.75,

        0.80,

        0.799,

        0.798,

        0.797,

    ]

    for epoch, value in enumerate(values, 1):

        early.step(
            value,
            epoch
        )

    assert early.best_value == 0.80

    assert early.best_epoch == 3

    assert early.should_stop is True

    print("[PASS] Macro-F1 maximization")

    # --------------------------------------------------------
    # TEST 3: Minimum improvement threshold
    # --------------------------------------------------------

    print("\nTEST 3: MINIMUM IMPROVEMENT")

    early = EarlyStopping(

        patience=3,

        min_delta=0.01,

        monitor="val_accuracy",

        verbose=False

    )

    early.step(0.80, 1)

    early.step(0.805, 2)

    assert early.improved is False

    assert early.bad_epochs == 1

    early.step(0.82, 3)

    assert early.improved is True

    assert early.best_value == 0.82

    print("[PASS] Minimum improvement threshold")

    # --------------------------------------------------------
    # TEST 4: Disabled early stopping
    # --------------------------------------------------------

    print("\nTEST 4: DISABLED EARLY STOPPING")

    early = EarlyStopping(

        patience=2,

        monitor="val_loss",

        enabled=False,

        verbose=False

    )

    for epoch, value in enumerate(
        [0.5, 0.6, 0.7, 0.8],
        1
    ):

        stopped = early.step(
            value,
            epoch
        )

        assert stopped is False

    assert early.best_value == 0.5

    assert early.bad_epochs == 3

    assert early.should_stop is False

    print("[PASS] Disabled early stopping")

    print("[PASS] Best metric tracking while disabled")

    # --------------------------------------------------------
    # TEST 5: State saving and restoration
    # --------------------------------------------------------

    print("\nTEST 5: CHECKPOINT STATE RESTORATION")

    original = EarlyStopping(

        patience=5,

        min_delta=0.001,

        monitor="val_macro_f1",

        verbose=False

    )

    original.step(0.70, 1)

    original.step(0.80, 2)

    original.step(0.79, 3)

    saved_state = original.state_dict()

    restored = EarlyStopping(

        patience=5,

        min_delta=0.001,

        monitor="val_macro_f1",

        verbose=False

    )

    restored.load_state_dict(
        saved_state
    )

    assert restored.best_value == original.best_value

    assert restored.best_epoch == original.best_epoch

    assert restored.bad_epochs == original.bad_epochs

    assert restored.last_epoch == original.last_epoch

    assert restored.improvement_count == (
        original.improvement_count
    )

    print("[PASS] State dictionary generation")

    print("[PASS] State restoration")

    # Continue after restoration.

    restored.step(0.78, 4)

    assert restored.bad_epochs == 2

    print("[PASS] Patience continuity after restoration")

    # --------------------------------------------------------
    # TEST 6: Invalid configuration
    # --------------------------------------------------------

    print("\nTEST 6: INVALID CONFIGURATION PROTECTION")

    invalid_configs = [

        {"patience": 0},

        {"min_delta": -0.1},

        {"monitor": "invalid_metric"},

        {"mode": "invalid_mode"},

    ]

    for config in invalid_configs:

        try:

            EarlyStopping(
                verbose=False,
                **config
            )

        except ValueError:

            continue

        raise AssertionError(
            f"Invalid configuration accepted: {config}"
        )

    print("[PASS] Invalid configuration protection")

    # --------------------------------------------------------
    # TEST 7: Non-finite metric protection
    # --------------------------------------------------------

    print("\nTEST 7: NON-FINITE METRIC PROTECTION")

    early = EarlyStopping(
        verbose=False
    )

    for invalid_value in [
        float("nan"),
        float("inf"),
        float("-inf")
    ]:

        try:

            early.step(
                invalid_value,
                epoch=1
            )

        except ValueError:

            continue

        raise AssertionError(
            "Non-finite metric was accepted."
        )

    print("[PASS] NaN and infinity protection")

    # --------------------------------------------------------
    # TEST 8: Epoch sequence protection
    # --------------------------------------------------------

    print("\nTEST 8: EPOCH SEQUENCE PROTECTION")

    early = EarlyStopping(
        verbose=False
    )

    early.step(
        0.80,
        epoch=1
    )

    try:

        early.step(
            0.81,
            epoch=1
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Duplicate epoch was accepted."
        )

    print("[PASS] Duplicate epoch protection")

    # --------------------------------------------------------
    # TEST 9: Reset
    # --------------------------------------------------------

    print("\nTEST 9: RESET")

    early.reset()

    assert early.best_value is None

    assert early.best_epoch is None

    assert early.bad_epochs == 0

    assert early.total_epochs == 0

    assert early.should_stop is False

    print("[PASS] State reset")

    # --------------------------------------------------------
    # TEST 10: Summary
    # --------------------------------------------------------

    print("\nTEST 10: SUMMARY")

    early = EarlyStopping(
        verbose=False
    )

    early.step(
        0.90,
        epoch=1
    )

    summary = early.get_summary()

    assert summary["best_value"] == 0.90

    assert summary["best_epoch"] == 1

    print("[PASS] Early stopping summary")

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\nEARLY STOPPING ENGINE TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
