import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results(log_path=None, save_path=None):
    if log_path is None:
        log_path = os.path.join(os.path.dirname(__file__), "logs", "training_log.csv")
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "logs", "training_performance.png")

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Training log not found at: {log_path}")

    df = pd.read_csv(log_path)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.suptitle("MAPPO Training Performance (20x20 Grid Exploration)", fontsize=15, fontweight="bold", y=0.98)

    runs = df["run"]

    # 1. Mean Episode Returns (with shaded std dev)
    ax1 = axes[0, 0]
    ax1.plot(runs, df["mean_return_drone"], label="Drone", color="#0284c7", lw=2, marker="o", ms=4)
    ax1.fill_between(
        runs,
        df["mean_return_drone"] - df["std_return_drone"],
        df["mean_return_drone"] + df["std_return_drone"],
        color="#0284c7", alpha=0.15
    )
    ax1.plot(runs, df["mean_return_rover"], label="Rover", color="#d97706", lw=2, marker="s", ms=4)
    ax1.fill_between(
        runs,
        df["mean_return_rover"] - df["std_return_rover"],
        df["mean_return_rover"] + df["std_return_rover"],
        color="#d97706", alpha=0.15
    )
    ax1.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.7)
    ax1.set_title("Mean Episode Return (± 1 std)", fontweight="semibold")
    ax1.set_xlabel("Training Run")
    ax1.set_ylabel("Return")
    ax1.set_xticks(runs)
    ax1.legend(frameon=True, loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Critic Loss (MSE)
    ax2 = axes[0, 1]
    ax2.plot(runs, df["critic_loss"], color="#e11d48", lw=2, marker="o", ms=4, label="Critic MSE Loss")
    ax2.set_title("Centralized Critic Loss", fontweight="semibold")
    ax2.set_xlabel("Training Run")
    ax2.set_ylabel("MSE Loss")
    ax2.set_xticks(runs)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, loc="upper right")

    # 3. Policy Entropy Decay
    ax3 = axes[1, 0]
    ax3.plot(runs, df["drone_entropy"], label="Drone Entropy", color="#0284c7", lw=2, marker="o", ms=4)
    ax3.plot(runs, df["rover_entropy"], label="Rover Entropy", color="#d97706", lw=2, marker="s", ms=4)
    ax3.axhline(1.792, color="gray", linestyle="--", lw=1, alpha=0.7, label="Uniform Max ln(6)")
    ax3.set_title("Policy Entropy Decay", fontweight="semibold")
    ax3.set_xlabel("Training Run")
    ax3.set_ylabel("Entropy")
    ax3.set_xticks(runs)
    ax3.legend(frameon=True, loc="upper right")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. Mean Episode Length
    ax4 = axes[1, 1]
    ax4.plot(runs, df["mean_ep_length"], color="#7c3aed", lw=2, marker="d", ms=4, label="Mean Ep Length")
    ax4.fill_between(
        runs,
        df["mean_ep_length"] - df["std_ep_length"],
        df["mean_ep_length"] + df["std_ep_length"],
        color="#7c3aed", alpha=0.15
    )
    ax4.axhline(300, color="gray", linestyle="--", lw=1, alpha=0.7, label="Max Steps (300)")
    ax4.set_title("Mean Episode Length", fontweight="semibold")
    ax4.set_xlabel("Training Run")
    ax4.set_ylabel("Steps")
    ax4.set_xticks(runs)
    ax4.legend(frameon=True, loc="lower right")
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot successfully saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_training_results()
