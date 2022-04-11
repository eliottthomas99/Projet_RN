import optuna
from utils import DEVICE
from encoder_decoder import EncoderDecoder
import torch.optim as optim


def objective(trial, vocab_size, encoder_dim, n_epochs, extractor, dataset, data_loader, loss_criterion, normalise):
    """
    Objective function to be minimized.

    :param trial: optuna trial object
    :vocab_size: size of vocabulary
    :encoder_dim: size of encoder
    :n_epochs: number of epochs
    :extractor: name of CNN to use
    :dataset: object containing dataset information
    :data_loader: data loader object
    :loss_criterion: loss to use (cross entropy or other)
    :normalise: if 1, normalise images
    :return: loss
    """
    # Generate the optimizers.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True) 
    embed_size = trial.suggest_int("embed_size", 100, 500)
    attention_dim = trial.suggest_int("attention_dim", 128, 512)
    decoder_dim = trial.suggest_int("decoder_dim", 256, 1024)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    print(trial.params) # Prints the parameters to be tested for this trial. 

    model = EncoderDecoder(
        vocab_size=vocab_size, encoder_dim=encoder_dim, n_epochs=n_epochs, 
        embed_size=embed_size, 
        attention_dim=attention_dim, decoder_dim=decoder_dim, 
        normalise=normalise, extractor=extractor, 
        dropout=dropout 
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model training and evaluation
    opti_loss = model.fit(data_loader, optimizer, loss_criterion, dataset)

    return opti_loss


def optisearch(extractor, dataset, data_loader, loss, vocab_size, encoder_dim, n_epochs, normalise):
    """
    Create a study object and run the Objective function.
    """
    objective_callback = lambda trial: objective(
                                trial, vocab_size=vocab_size, 
                                encoder_dim=encoder_dim, n_epochs=n_epochs, 
                                extractor=extractor, dataset=dataset, data_loader=data_loader, 
                                loss_criterion=loss, normalise=normalise)

    study = optuna.create_study(direction="minimize") # Create a study object.
    study.optimize(objective_callback, n_trials=10) # Run the study.

    print("Number of finished trials: ", len(study.trials)) # Prints the number of finished trials.

    print("Best value:", study.best_value) # Prints the best value.
    print("Best params:", study.best_params) # Prints the best hyperparameters.
