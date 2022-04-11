import optuna
from utils import DEVICE
from encoder_decoder import EncoderDecoder
import torch.optim as optim


def objective(trial, vocab_size, encoder_dim, n_epochs, extractor, dataset, data_loader, loss_criterion, normalise):

    # Generate the optimizers.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    embed_size = trial.suggest_int("embed_size", 100, 500)
    attention_dim = trial.suggest_int("attention_dim", 128, 512)
    decoder_dim = trial.suggest_int("decoder_dim", 256, 1024)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    print(trial.params)


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

    objective_callback = lambda trial: objective(
                                trial, vocab_size=vocab_size, 
                                encoder_dim=encoder_dim, n_epochs=n_epochs, 
                                extractor=extractor, dataset=dataset, data_loader=data_loader, 
                                loss_criterion=loss, normalise=normalise)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_callback, n_trials=10)

    print("Number of finished trials: ", len(study.trials))

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
