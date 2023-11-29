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
    
