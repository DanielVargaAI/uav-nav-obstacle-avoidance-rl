from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def plot_run(
    run_dir: str = typer.Option(..., help="Pfad zum Run-Ordner mit training_metrics.csv"),
    window: int = 50,
):
    """
    Erstellt Übersichts-Plots für Episode Rewards, Success Rate, Crash Rate und Steps.
    """
    csv_path = Path(run_dir) / "training_metrics.csv"
    if not csv_path.exists():
        print(f"Keine Metriken-Datei gefunden unter: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Gleitende Mittelwerte berechnen
    df["reward_smooth"] = df["reward"].rolling(window, min_periods=1).mean()
    df["success_rate"] = df["success"].rolling(window, min_periods=1).mean() * 100
    df["collision_rate"] = df["collision"].rolling(window, min_periods=1).mean() * 100
    df["timeout_rate"] = df["timeout"].rolling(window, min_periods=1).mean() * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Trainingsauswertung: {Path(run_dir).name}", fontsize=16)

    # 1. Rewards
    axes[0, 0].plot(df["episode"], df["reward"], alpha=0.3, color="gray", label="Raw Reward")
    axes[0, 0].plot(df["episode"], df["reward_smooth"], color="blue", linewidth=2, label=f"Moving Average ({window})")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # 2. Performance (Success / Crash / Timeout Rates)
    axes[0, 1].plot(df["episode"], df["success_rate"], color="green", linewidth=2, label="Success %")
    axes[0, 1].plot(df["episode"], df["collision_rate"], color="red", linewidth=2, label="Crash %")
    axes[0, 1].plot(df["episode"], df["timeout_rate"], color="orange", linewidth=2, label="Timeout %")
    axes[0, 1].set_title(f"Performance Rates (Gleitender Schnitt over {window} Ep.)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Prozent (%)")
    axes[0, 1].set_ylim(-5, 105)
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # 3. Episode-Länge (Steps)
    axes[1, 0].plot(df["episode"], df["length"].rolling(window, min_periods=1).mean(), color="purple")
    axes[1, 0].set_title("Durchschnittliche Episode-Länge (Steps)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].grid(True)

    # 4. Curriculum Stage Verlauf
    axes[1, 1].plot(df["episode"], df["stage"], color="teal", drawstyle="steps-post")
    axes[1, 1].set_title("Curriculum Stage Verlauf")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Stage")
    axes[1, 1].grid(True)

    plt.tight_layout()
    output_path = Path(run_dir) / "training_summary.png"
    plt.savefig(output_path, dpi=300)
    print(f"Auswertung erfolgreich gespeichert unter: {output_path}")
    plt.show()


if __name__ == "__main__":
    app()

# uv run python -m uav_nav_obstacle_avoidance_rl.visualization.plotter --run-dir "reports/45.0deg_72x9/PPO_Model/run_20261018_224122"
