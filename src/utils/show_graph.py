import matplotlib.pyplot as plt


def plot_history(history: Dict[str, list], title: str = "History"):
    plt.figure(figsize=(12, 6))
    for phase in history.keys():
        plt.plot(history[phase], label=f'{phase.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.show()