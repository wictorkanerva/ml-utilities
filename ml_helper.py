import matplotlib.pyplot as plt
def plot_loss_curves(history):
    """
    Returns seperate loss curves for training and validation metrics.
    
    Args:
        history: TensorFlow History object
        
    Returns:
        Plots of trainng/validation loss and accuracy metrics.
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    
    num_epochs = range(len(history.history["loss"]))
    
    # Plot loss
    plt.plot(num_epochs, loss, label="training_loss")
    plt.plot(num_epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    # Plot accuracy
    plt.figure()
    plt.plot(num_epochs, accuracy, label="training_accuracy")
    plt.plot(num_epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()



def compare_histories(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow History objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]
    
    # Combine original history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]
    
    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]
    
    # Adjust the figure size and layout
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

    # Plot for accuracy
    axes[0].plot(total_acc, label="Training Accuracy", marker='o', linestyle='-')
    axes[0].plot(total_val_acc, label="Validation Accuracy", marker='o', linestyle='-')
    axes[0].axvline(x=initial_epochs-1, color='r', linestyle='--', label="Start Fine Tuning")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)

    # Plot for loss
    axes[1].plot(total_loss, label="Training Loss", marker='o', linestyle='-')
    axes[1].plot(total_val_loss, label="Validation Loss", marker='o', linestyle='-')
    axes[1].axvline(x=initial_epochs-1, color='r', linestyle='--', label="Start Fine Tuning")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()
    
    
    
