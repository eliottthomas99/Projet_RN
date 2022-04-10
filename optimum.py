import optuna


def objective(trial, model, extractor, data_loader, optimizer, loss_criterion, dataset):
    # Generate the optimizers.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    embed_size = trial.suggest_int("embed_size", 100, 500)
    attention_dim = trial.suggest_int("attention_dim", 128, 512)
    decoder_dim = trial.suggest_int("decoder_dim", 256, 1024)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model.set_params(extractor, learning_rate, embed_size, attention_dim, decoder_dim, dropout)

    # model training and evaluation
    opti_loss = model.fit(data_loader, optimizer, loss_criterion, dataset)

    return opti_loss


def optisearch(model, extractor, dataset, data_loader, optimizer, loss):

    objective_callback = lambda trial: objective(trial, model, extractor, dataset, data_loader, optimizer, loss)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_callback, n_trials=10)

    print("Number of finished trials: ", len(study.trials))

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
