from agent.utils import load_model, ModelVersion
import visualkeras


if __name__ == "__main__":
    model = load_model(None, 10, ModelVersion.BCNetLSTM)
    visualkeras.layered_view(
        model,
        "model.png",
        legend=True,
        show_dimension=True,
        max_xy=300,
    )
